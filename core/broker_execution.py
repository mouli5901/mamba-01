"""Zerodha Kite Connect Live Execution Engine for AI Council Trading Engine.

Phase 7 requirements:
1. Kept strictly behind LIVE_TRADING flag.
2. Daily token refresh & auth lifecycle (6 AM IST daily expiry).
3. Idempotent order placement with client_order_id.
   On network timeout/5xx: check order status by client_order_id before resubmitting (never blind-retry).
   Exponential backoff (3 attempts at 1s/2s/4s); hard failure triggers kill-switch path.
4. Reconciliation job (market open + every N minutes): diff Kite actual positions against internal ledger.
5. Sole caller is core/risk_manager.py.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import KITE_API_KEY, KITE_API_SECRET, LIVE_TRADING
from core.risk_manager import ExecutionResult

logger = logging.getLogger(__name__)

SESSION_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / ".kite_session.json"


class KiteAuthError(Exception):
    """Raised when Zerodha daily authentication is missing or expired."""
    pass


class KiteAuthManager:
    """Manages Zerodha Kite Connect login, token generation, and daily session cache."""

    def __init__(
        self,
        api_key: str = KITE_API_KEY,
        api_secret: str = KITE_API_SECRET,
        session_path: Path | str = SESSION_CACHE_PATH,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session_path = Path(session_path)

    def get_login_url(self) -> str:
        """Generate official Zerodha Kite login URL for daily manual auth."""
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=self.api_key)
        return kite.login_url()

    def generate_session(self, request_token: str) -> Dict[str, Any]:
        """Exchange morning request_token for daily access_token (~6 AM IST rollover)."""
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=self.api_key)
        data = kite.generate_session(request_token, api_secret=self.api_secret)

        session_payload = {
            "access_token": data["access_token"],
            "public_token": data.get("public_token", ""),
            "user_id": data.get("user_id", ""),
            "session_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.session_path, "w", encoding="utf-8") as f:
            json.dump(session_payload, f, indent=2)

        logger.info(f"Kite session generated and saved for user {session_payload['user_id']}.")
        return session_payload

    def get_access_token(self) -> str:
        """Retrieve active access token if issued today, otherwise raise KiteAuthError."""
        if not self.session_path.exists():
            raise KiteAuthError("No Kite session found. Morning login required.")

        with open(self.session_path, "r", encoding="utf-8") as f:
            session = json.load(f)

        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if session.get("session_date") != today_str:
            raise KiteAuthError(
                f"Kite access token expired (issued on {session.get('session_date')}, today is {today_str}). "
                "Zerodha tokens expire daily at 6 AM IST. Morning login required."
            )

        return session["access_token"]


class BrokerExecutionEngine:
    """Live execution engine interacting with Zerodha Kite Connect API."""

    def __init__(
        self,
        auth_manager: Optional[KiteAuthManager] = None,
        kite_client: Optional[Any] = None,
        live_trading: bool = LIVE_TRADING,
    ):
        self.auth_manager = auth_manager or KiteAuthManager()
        self._kite_client = kite_client
        self.live_trading = live_trading

    def _get_client(self) -> Any:
        """Instantiate and authenticate KiteConnect client."""
        if self._kite_client is not None:
            return self._kite_client

        from kiteconnect import KiteConnect
        token = self.auth_manager.get_access_token()
        kite = KiteConnect(api_key=self.auth_manager.api_key)
        kite.set_access_token(token)
        return kite

    def check_order_status_by_tag(self, client_order_id: str, kite: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """Look up order on Zerodha by client_order_id tag to prevent duplicate submissions."""
        client = kite or self._get_client()
        tag = client_order_id[:20]  # Kite tags allow max 20 alphanumeric characters
        try:
            orders = client.orders()
            for o in orders:
                if o.get("tag") == tag:
                    return o
        except Exception as e:
            logger.warning(f"Error checking order status by tag {tag}: {e}")
        return None

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        client_order_id: Optional[str] = None,
        product: str = "CNC",  # CNC = Delivery, MIS = Intraday
        order_type: str = "LIMIT",
        exchange: str = "NSE",
    ) -> ExecutionResult:
        """Idempotently place order on Zerodha with retry logic and tag tracking.

        On timeout or 5xx: checks order status by client_order_id before resubmitting.
        Never blind-retries order placement.
        """
        if not self.live_trading:
            return ExecutionResult(
                success=False,
                mode="LIVE",
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                error="LIVE_TRADING is false. Live orders are blocked by ground rules.",
            )

        order_id = client_order_id or str(uuid.uuid4())
        tag = order_id[:20]
        side_norm = side.upper()

        kite = self._get_client()

        # Step 1: Pre-check if order was already submitted
        existing = self.check_order_status_by_tag(order_id, kite=kite)
        if existing:
            logger.info(f"Order {order_id} already exists on Kite: status={existing.get('status')}")
            return ExecutionResult(
                success=True,
                mode="LIVE",
                order_id=existing.get("order_id", order_id),
                symbol=symbol,
                side=side_norm,
                qty=qty,
                price=float(existing.get("average_price") or price),
            )

        # Step 2: Order execution with exponential backoff retry (1s, 2s, 4s)
        backoffs = [1.0, 2.0, 4.0]
        last_error: Optional[Exception] = None

        for attempt, delay in enumerate(backoffs, start=1):
            try:
                from kiteconnect import KiteConnect

                tx_type = (
                    KiteConnect.TRANSACTION_TYPE_BUY
                    if side_norm == "BUY"
                    else KiteConnect.TRANSACTION_TYPE_SELL
                )
                prod = KiteConnect.PRODUCT_CNC if product == "CNC" else KiteConnect.PRODUCT_MIS
                otype = KiteConnect.ORDER_TYPE_LIMIT if order_type == "LIMIT" else KiteConnect.ORDER_TYPE_MARKET

                broker_order_id = kite.place_order(
                    variety=KiteConnect.VARIETY_REGULAR,
                    exchange=exchange,
                    tradingsymbol=symbol,
                    transaction_type=tx_type,
                    quantity=qty,
                    product=prod,
                    order_type=otype,
                    price=price if order_type == "LIMIT" else None,
                    tag=tag,
                )

                logger.info(f"[Zerodha Live Order Success] {side_norm} {qty} {symbol} @ {price}, ID: {broker_order_id}")
                return ExecutionResult(
                    success=True,
                    mode="LIVE",
                    order_id=str(broker_order_id),
                    symbol=symbol,
                    side=side_norm,
                    qty=qty,
                    price=price,
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Kite order placement attempt {attempt}/{len(backoffs)} failed: {e}. "
                    f"Checking status by tag '{tag}' before retry..."
                )

                # CRITICAL SAFETY: Check if order actually succeeded server-side before retrying
                time.sleep(delay)
                existing = self.check_order_status_by_tag(order_id, kite=kite)
                if existing:
                    logger.info(f"Order found on server despite client error: {existing.get('order_id')}")
                    return ExecutionResult(
                        success=True,
                        mode="LIVE",
                        order_id=existing.get("order_id", order_id),
                        symbol=symbol,
                        side=side_norm,
                        qty=qty,
                        price=float(existing.get("average_price") or price),
                    )

        # All attempts exhausted
        logger.critical(f"Kite order placement failed after {len(backoffs)} retries: {last_error}")
        return ExecutionResult(
            success=False,
            mode="LIVE",
            symbol=symbol,
            side=side_norm,
            qty=qty,
            price=price,
            error=f"Exhausted retries on Kite Connect: {str(last_error)}",
        )

    def reconcile_positions(
        self,
        internal_positions: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Reconciliation job: Diffs Kite's actual positions against internal ledger.

        Alerts immediately on any mismatch.
        """
        kite = self._get_client()
        try:
            kite_positions_raw = kite.positions()
        except Exception as e:
            logger.error(f"Failed to fetch positions from Kite during reconciliation: {e}")
            return {"in_sync": False, "error": str(e), "mismatches": []}

        # Combine 'net' positions from Zerodha
        net_positions = kite_positions_raw.get("net", [])
        actual_positions: Dict[str, int] = {}
        for pos in net_positions:
            sym = pos.get("tradingsymbol", "")
            qty = int(pos.get("quantity", 0))
            if qty != 0:
                actual_positions[sym] = qty

        mismatches = []
        all_symbols = set(internal_positions.keys()) | set(actual_positions.keys())

        for sym in all_symbols:
            internal_qty = internal_positions.get(sym, {}).get("qty", 0)
            actual_qty = actual_positions.get(sym, 0)

            if internal_qty != actual_qty:
                mismatch = {
                    "symbol": sym,
                    "internal_qty": internal_qty,
                    "broker_actual_qty": actual_qty,
                    "drift": actual_qty - internal_qty,
                }
                mismatches.append(mismatch)
                logger.critical(
                    f"POSITION RECONCILIATION MISMATCH for {sym}! "
                    f"Internal: {internal_qty}, Broker Actual: {actual_qty}, Drift: {mismatch['drift']}"
                )

        in_sync = len(mismatches) == 0
        if in_sync:
            logger.info("Position reconciliation PASSED: internal ledger matches Zerodha broker positions.")

        return {
            "in_sync": in_sync,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "mismatches": mismatches,
            "broker_positions": actual_positions,
        }
