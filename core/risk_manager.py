"""Risk Manager for AI Council Trading Engine.

Phase 6 requirements:
1. Hard caps from config/risk_rules.json:
   - max 1-2% capital per trade
   - max concurrent positions
   - max daily loss (kill switch — halts new orders for the day)
   - max consecutive losses
2. Dead-man's switch:
   - If data engine heartbeat is stale beyond N seconds:
     - Paper mode: flatten open positions
     - Live mode: alert and halt
3. Sole execution gateway:
   - This module is the ONLY caller of paper_engine.py and broker_execution.py.
   - Enforces default Mode A (paper). Mode B (live Zerodha) requires both LIVE_TRADING=true
     AND explicit manual confirmation.
4. Test gate:
   - Rejects any proposal breaching a single hard cap regardless of council consensus.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import LIVE_TRADING
from core.paper_engine import OrderResult, PaperTradingEngine

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "risk_rules.json"


@dataclass
class RiskCheckResult:
    approved: bool
    rejection_reason: Optional[str] = None
    calculated_qty: int = 0
    trade_cost: float = 0.0
    kill_switch_triggered: bool = False


@dataclass
class ExecutionResult:
    success: bool
    mode: str  # "PAPER" or "LIVE"
    order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    qty: int = 0
    price: float = 0.0
    fees: float = 0.0
    balance_after: float = 0.0
    error: Optional[str] = None


class RiskManager:
    """Institutional Risk Management Gate and Sole Execution Controller."""

    def __init__(
        self,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        paper_engine: Optional[PaperTradingEngine] = None,
        broker_execution: Optional[Any] = None,
        live_trading: bool = LIVE_TRADING,
    ):
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)

        # Risk parameters
        self.max_capital_per_trade_pct = float(self.config.get("max_capital_per_trade_pct", 0.02))
        self.max_concurrent_positions = int(self.config.get("max_concurrent_positions", 5))
        self.max_daily_loss_pct = float(self.config.get("max_daily_loss_pct", 0.05))
        self.max_consecutive_losses = int(self.config.get("max_consecutive_losses", 4))
        self.heartbeat_stale_seconds = float(self.config.get("heartbeat_stale_seconds", 30))
        self.initial_capital = float(self.config.get("initial_capital", 100000.0))

        # Trading mode: Default Mode A (paper)
        self.live_trading = live_trading

        # State trackers
        self.daily_starting_balance = self.initial_capital
        self.current_consecutive_losses = 0
        self.kill_switch_active = False
        self.kill_switch_reason: Optional[str] = None
        self.last_heartbeat_ts: float = datetime.now(timezone.utc).timestamp()
        self.halted_due_to_stale_heartbeat = False

        # Execution engines (injected or default)
        self.paper_engine = paper_engine or PaperTradingEngine(config_path=self.config_path)
        self.broker_execution = broker_execution

    def _load_config(self, path: Path) -> Dict[str, Any]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def record_heartbeat(self, timestamp: Optional[float] = None) -> None:
        """Update data engine heartbeat timestamp."""
        self.last_heartbeat_ts = timestamp or datetime.now(timezone.utc).timestamp()
        if self.halted_due_to_stale_heartbeat:
            logger.info("Heartbeat restored; clearing heartbeat halt.")
            self.halted_due_to_stale_heartbeat = False

    def is_heartbeat_stale(self, current_ts: Optional[float] = None) -> bool:
        """Check if data engine heartbeat exceeds threshold."""
        now = current_ts or datetime.now(timezone.utc).timestamp()
        staleness = now - self.last_heartbeat_ts
        return staleness > self.heartbeat_stale_seconds

    def check_dead_mans_switch(
        self,
        current_prices: Optional[Dict[str, float]] = None,
        current_ts: Optional[float] = None,
    ) -> bool:
        """Dead-man's switch:

        If heartbeat is stale > N seconds:
        - Mode A (Paper): flatten all open positions.
        - Mode B (Live): alert and halt new orders.
        Returns True if dead-man's switch triggered.
        """
        if not self.is_heartbeat_stale(current_ts=current_ts):
            return False

        staleness = (current_ts or datetime.now(timezone.utc).timestamp()) - self.last_heartbeat_ts
        logger.critical(f"DEAD-MAN'S SWITCH TRIGGERED: Heartbeat stale for {staleness:.1f}s (> {self.heartbeat_stale_seconds}s)")
        self.halted_due_to_stale_heartbeat = True

        if not self.live_trading:
            # Mode A (Paper): Emergency liquidation of all open positions
            positions = self.paper_engine.get_positions()
            for symbol, pos in positions.items():
                qty = pos["qty"]
                if qty > 0:
                    exit_price = (current_prices or {}).get(symbol, pos["avg_price"])
                    logger.warning(f"[Dead-Man Emergency] Flattening position {symbol} qty={qty} at {exit_price}")
                    try:
                        self.paper_engine.execute_order(
                            symbol=symbol,
                            side="SELL",
                            qty=qty,
                            price=exit_price,
                            skip_risk_checks=True,
                        )
                    except Exception as e:
                        logger.error(f"Failed to flatten {symbol} during emergency liquidation: {e}")
        else:
            # Mode B (Live): Alert and halt
            logger.critical("[Dead-Man Emergency LIVE] HALTING ALL ORDERS. Stale tick data detected.")

        return True

    def calculate_position_size(
        self,
        price: float,
        balance: float,
        confidence: float = 1.0,
    ) -> int:
        """Compute position quantity capped strictly by max_capital_per_trade_pct."""
        if price <= 0 or balance <= 0:
            return 0
        allowed_capital = balance * self.max_capital_per_trade_pct
        max_qty = math.floor(allowed_capital / price)
        return max(0, max_qty)

    def evaluate_proposal(
        self,
        proposal: Dict[str, Any],
        current_balance: Optional[float] = None,
        open_positions: Optional[Dict[str, Dict[str, Any]]] = None,
        current_ts: Optional[float] = None,
    ) -> RiskCheckResult:
        """Run all hard risk checks against a council proposal.

        Rejects immediately if ANY hard cap is breached, regardless of council consensus.
        """
        symbol = proposal.get("symbol", "")
        side = str(proposal.get("side", "")).upper()
        price = float(proposal.get("requested_price", 0.0))
        confidence = float(proposal.get("confidence", 1.0))

        balance = current_balance if current_balance is not None else self.paper_engine.get_current_balance()
        positions = open_positions if open_positions is not None else self.paper_engine.get_positions()

        # Check 1: Kill switch active
        if self.kill_switch_active:
            return RiskCheckResult(
                approved=False,
                rejection_reason=f"Kill Switch Active: {self.kill_switch_reason}",
                kill_switch_triggered=True,
            )

        # Check 2: Dead-man's switch / Stale heartbeat
        if self.is_heartbeat_stale(current_ts=current_ts) or self.halted_due_to_stale_heartbeat:
            return RiskCheckResult(
                approved=False,
                rejection_reason=f"Dead-Man's Switch: Data engine heartbeat is stale (> {self.heartbeat_stale_seconds}s)",
            )

        # Check 3: Max consecutive losses cap
        if self.current_consecutive_losses >= self.max_consecutive_losses:
            self.kill_switch_active = True
            self.kill_switch_reason = f"Max consecutive losses ({self.max_consecutive_losses}) breached"
            return RiskCheckResult(
                approved=False,
                rejection_reason=self.kill_switch_reason,
                kill_switch_triggered=True,
            )

        # Check 4: Daily loss limit kill switch
        daily_loss_pct = (self.daily_starting_balance - balance) / self.daily_starting_balance
        if daily_loss_pct >= self.max_daily_loss_pct:
            self.kill_switch_active = True
            self.kill_switch_reason = (
                f"Max daily loss limit breached: {daily_loss_pct*100:.2f}% >= {self.max_daily_loss_pct*100:.2f}%"
            )
            return RiskCheckResult(
                approved=False,
                rejection_reason=self.kill_switch_reason,
                kill_switch_triggered=True,
            )

        # BUY side checks
        if side == "BUY":
            # Check 5: Max concurrent positions cap
            if symbol not in positions and len(positions) >= self.max_concurrent_positions:
                return RiskCheckResult(
                    approved=False,
                    rejection_reason=f"Max concurrent positions limit ({self.max_concurrent_positions}) reached",
                )

            # Determine proposed quantity
            requested_qty = proposal.get("qty")
            calculated_qty = (
                int(requested_qty)
                if requested_qty is not None
                else self.calculate_position_size(price, balance, confidence)
            )

            if calculated_qty <= 0:
                return RiskCheckResult(
                    approved=False,
                    rejection_reason=f"Calculated position size is 0 (price {price} vs capital cap)",
                    calculated_qty=0,
                )

            # Check 6: Hard capital per trade cap
            trade_cost = calculated_qty * price
            max_allowed_trade = balance * self.max_capital_per_trade_pct
            if trade_cost > max_allowed_trade:
                return RiskCheckResult(
                    approved=False,
                    rejection_reason=(
                        f"Trade cost {trade_cost:.2f} exceeds {self.max_capital_per_trade_pct*100}% "
                        f"capital cap ({max_allowed_trade:.2f})"
                    ),
                    calculated_qty=calculated_qty,
                    trade_cost=trade_cost,
                )

            # Check 7: Solvency
            if trade_cost > balance:
                return RiskCheckResult(
                    approved=False,
                    rejection_reason=f"Insufficient funds: trade cost {trade_cost:.2f} > balance {balance:.2f}",
                    calculated_qty=calculated_qty,
                    trade_cost=trade_cost,
                )

            return RiskCheckResult(
                approved=True,
                calculated_qty=calculated_qty,
                trade_cost=trade_cost,
            )

        # SELL side checks
        elif side == "SELL":
            current_pos = positions.get(symbol, {"qty": 0})
            held_qty = current_pos["qty"]
            requested_qty = proposal.get("qty", held_qty)
            qty_to_sell = min(held_qty, requested_qty)

            if held_qty <= 0 or qty_to_sell <= 0:
                return RiskCheckResult(
                    approved=False,
                    rejection_reason=f"Position shortage: Attempting to sell {symbol}, but held qty is {held_qty}",
                )

            return RiskCheckResult(
                approved=True,
                calculated_qty=qty_to_sell,
                trade_cost=qty_to_sell * price,
            )

        return RiskCheckResult(
            approved=False,
            rejection_reason=f"Invalid order side: {side}",
        )

    def route_and_execute(
        self,
        proposal: Dict[str, Any],
        manual_confirmation_received: bool = False,
        client_order_id: Optional[str] = None,
    ) -> ExecutionResult:
        """Sole caller of execution engines.

        Evaluates proposal against all hard caps, then dispatches to paper_engine
        or broker_execution depending on trading mode and manual confirmation.
        """
        # Step 1: Run all hard risk checks
        check = self.evaluate_proposal(proposal)
        if not check.approved:
            logger.warning(f"Proposal REJECTED by RiskManager: {check.rejection_reason}")
            return ExecutionResult(
                success=False,
                mode="LIVE" if self.live_trading else "PAPER",
                symbol=proposal.get("symbol", ""),
                side=proposal.get("side", ""),
                error=check.rejection_reason,
            )

        symbol = proposal["symbol"]
        side = proposal["side"]
        price = float(proposal["requested_price"])
        qty = check.calculated_qty
        order_id = client_order_id or proposal.get("proposal_id") or str(uuid.uuid4())

        # Step 2: Live Mode Safeguards
        if self.live_trading:
            if not manual_confirmation_received:
                reason = "LIVE_TRADING is enabled, but manual operator confirmation was NOT provided."
                logger.critical(f"Live order blocked: {reason}")
                return ExecutionResult(
                    success=False,
                    mode="LIVE",
                    symbol=symbol,
                    side=side,
                    error=reason,
                )

            if self.broker_execution is None:
                return ExecutionResult(
                    success=False,
                    mode="LIVE",
                    symbol=symbol,
                    side=side,
                    error="Broker execution module not wired.",
                )

            logger.info(f"[LIVE EXECUTION] Submitting {side} {qty} {symbol} @ {price}")
            return self.broker_execution.place_order(
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                client_order_id=order_id,
            )

        # Step 3: Default Mode A (Paper) Execution
        logger.info(f"[PAPER EXECUTION] Submitting {side} {qty} {symbol} @ {price}")
        try:
            res: OrderResult = self.paper_engine.execute_order(
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                client_order_id=order_id,
                skip_risk_checks=True,  # Risk checks already fully enforced above
            )
            return ExecutionResult(
                success=True,
                mode="PAPER",
                order_id=res.client_order_id,
                symbol=res.symbol,
                side=res.side,
                qty=res.qty,
                price=res.fill_price,
                fees=res.fees,
                balance_after=res.balance_after,
            )
        except Exception as e:
            logger.error(f"Paper execution error: {e}")
            return ExecutionResult(
                success=False,
                mode="PAPER",
                symbol=symbol,
                side=side,
                qty=qty,
                error=str(e),
            )

    def record_trade_outcome(self, pnl: float) -> None:
        """Update consecutive loss counters and daily PnL."""
        if pnl < 0:
            self.current_consecutive_losses += 1
            if self.current_consecutive_losses >= self.max_consecutive_losses:
                self.kill_switch_active = True
                self.kill_switch_reason = f"Max consecutive losses ({self.max_consecutive_losses}) hit"
                logger.critical(self.kill_switch_reason)
        else:
            self.current_consecutive_losses = 0

    def reset_daily_limits(self, starting_balance: Optional[float] = None) -> None:
        """Reset limits at start of new trading day."""
        balance = starting_balance if starting_balance is not None else self.paper_engine.get_current_balance()
        self.daily_starting_balance = balance
        self.current_consecutive_losses = 0
        self.kill_switch_active = False
        self.kill_switch_reason = None
        logger.info(f"Daily risk limits reset. Starting capital: {self.daily_starting_balance}")
