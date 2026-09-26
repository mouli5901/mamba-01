"""Paper Trading Engine for AI Council Trading Engine.

Phase 3 requirements:
1. SQLite schema: `orders`, `positions`, `ledger`, with an explicit `client_order_id` (UUID) column.
2. Slippage: 0.02% against the touched price, plus configurable bid-ask spread estimate; fees/taxes from config/risk_rules.json.
3. Defense in depth: re-verifies capital and position limits before execution.
4. Atomic writes across all 3 tables per order fill.
5. Backtest/replay mode with dedicated DB isolation (data/backtest_portfolio.db).
6. Test gate: scripted buy/sell sequence against replayed bars asserting exact hand-computed ledger balance.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "risk_rules.json"
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "paper_portfolio.db"
DEFAULT_BACKTEST_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "backtest_portfolio.db"


@dataclass
class OrderResult:
    client_order_id: str
    symbol: str
    side: str
    qty: int
    requested_price: float
    fill_price: float
    fees: float
    status: str
    created_at: str
    cash_delta: float
    balance_after: float


class PaperTradingEngine:
    """Brokerless execution engine matching orders against bars/ticks with slippage and fee modeling."""

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        initial_capital: Optional[float] = None,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config(config_path)

        if initial_capital is not None:
            self.initial_capital = float(initial_capital)
        else:
            self.initial_capital = float(self.config.get("initial_capital", 100000.0))

        self.slippage_pct = float(self.config.get("slippage_pct", 0.0002))
        self.bid_ask_spread_estimate = float(self.config.get("bid_ask_spread_estimate", 0.0001))
        self.fees_config = self.config.get("fees", {})
        self.max_capital_per_trade_pct = float(self.config.get("max_capital_per_trade_pct", 0.02))
        self.max_concurrent_positions = int(self.config.get("max_concurrent_positions", 5))

        self._init_sqlite()

    def _load_config(self, config_path: Path | str) -> Dict[str, Any]:
        p = Path(config_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
        finally:
            conn.close()

    def _init_sqlite(self) -> None:
        """Initialize SQLite schema with orders, positions, and ledger."""
        with self._connection() as conn:
            with conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS orders (
                        client_order_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        qty INTEGER NOT NULL,
                        requested_price REAL NOT NULL,
                        fill_price REAL NOT NULL,
                        fees REAL NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS positions (
                        symbol TEXT PRIMARY KEY,
                        qty INTEGER NOT NULL,
                        avg_price REAL NOT NULL,
                        unrealized_pnl REAL NOT NULL DEFAULT 0.0
                    );

                    CREATE TABLE IF NOT EXISTS ledger (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        client_order_id TEXT NOT NULL,
                        cash_delta REAL NOT NULL,
                        balance_after REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (client_order_id) REFERENCES orders(client_order_id)
                    );
                """)

    def get_current_balance(self, conn: Optional[sqlite3.Connection] = None) -> float:
        """Fetch latest running cash balance from the ledger, or initial capital if empty."""
        if conn is not None:
            row = conn.execute(
                "SELECT balance_after FROM ledger ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return float(row["balance_after"]) if row else self.initial_capital

        with self._connection() as local_conn:
            row = local_conn.execute(
                "SELECT balance_after FROM ledger ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return float(row["balance_after"]) if row else self.initial_capital

    def get_positions(self, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Dict[str, Any]]:
        """Fetch all open positions (qty > 0)."""
        def _extract(c: sqlite3.Connection):
            rows = c.execute(
                "SELECT symbol, qty, avg_price, unrealized_pnl FROM positions WHERE qty > 0"
            ).fetchall()
            return {
                r["symbol"]: {
                    "qty": int(r["qty"]),
                    "avg_price": float(r["avg_price"]),
                    "unrealized_pnl": float(r["unrealized_pnl"]),
                }
                for r in rows
            }

        if conn is not None:
            return _extract(conn)

        with self._connection() as local_conn:
            return _extract(local_conn)

    def calculate_fill_price(self, price: float, side: str) -> float:
        """Apply 0.02% slippage + half bid-ask spread against touched price.

        BUY pays higher (price * (1 + slippage + spread/2)).
        SELL receives lower (price * (1 - slippage - spread/2)).
        """
        side_norm = side.upper()
        spread_impact = self.bid_ask_spread_estimate / 2.0
        total_drag = self.slippage_pct + spread_impact

        if side_norm == "BUY":
            return round(price * (1.0 + total_drag), 4)
        elif side_norm == "SELL":
            return round(price * (1.0 - total_drag), 4)
        else:
            raise ValueError(f"Invalid order side: {side}")

    def calculate_fees(self, turnover: float, side: str) -> float:
        """Calculate complete Indian exchange taxes and fees per config/risk_rules.json.

        STT, exchange txn charges, SEBI turnover fees, stamp duty (BUY only),
        GST (18% on brokerage + exchange txn + SEBI), and per-order brokerage.
        """
        side_norm = side.upper()
        brokerage = float(self.fees_config.get("brokerage_per_order", 20.0))
        stt = turnover * float(self.fees_config.get("stt_pct", 0.001))
        exchange_txn = turnover * float(self.fees_config.get("exchange_txn_pct", 0.0000345))
        sebi_turnover = turnover * float(self.fees_config.get("sebi_turnover_pct", 0.000001))

        # Stamp duty is charged on buyer only
        stamp_duty = (turnover * float(self.fees_config.get("stamp_duty_pct", 0.00003))) if side_norm == "BUY" else 0.0

        gst_rate = float(self.fees_config.get("gst_pct", 0.18))
        gst = gst_rate * (brokerage + exchange_txn + sebi_turnover)

        total_fees = brokerage + stt + exchange_txn + sebi_turnover + stamp_duty + gst
        return round(total_fees, 4)

    def execute_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        price: float,
        client_order_id: Optional[str] = None,
        created_at: Optional[str] = None,
        skip_risk_checks: bool = False,
    ) -> OrderResult:
        """Idempotently execute an order with atomic write across orders, positions, and ledger."""
        if qty <= 0:
            raise ValueError(f"Quantity must be positive, got {qty}")
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")

        side_norm = side.upper()
        if side_norm not in ("BUY", "SELL"):
            raise ValueError(f"Side must be BUY or SELL, got {side}")

        order_id = client_order_id or str(uuid.uuid4())
        ts = created_at or datetime.now(timezone.utc).isoformat()

        with self._connection() as conn:
            with conn:
                # 1. Idempotency check: if order exists, return recorded result
                existing = conn.execute(
                    "SELECT * FROM orders WHERE client_order_id = ?", (order_id,)
                ).fetchone()
                if existing:
                    ledger_row = conn.execute(
                        "SELECT cash_delta, balance_after FROM ledger WHERE client_order_id = ?", (order_id,)
                    ).fetchone()
                    return OrderResult(
                        client_order_id=existing["client_order_id"],
                        symbol=existing["symbol"],
                        side=existing["side"],
                        qty=existing["qty"],
                        requested_price=existing["requested_price"],
                        fill_price=existing["fill_price"],
                        fees=existing["fees"],
                        status=existing["status"],
                        created_at=existing["created_at"],
                        cash_delta=ledger_row["cash_delta"] if ledger_row else 0.0,
                        balance_after=ledger_row["balance_after"] if ledger_row else 0.0,
                    )

                # Current state
                current_balance = self.get_current_balance(conn)
                positions = self.get_positions(conn)
                current_pos = positions.get(symbol, {"qty": 0, "avg_price": 0.0, "unrealized_pnl": 0.0})

                # Calculate fill price and fees
                fill_price = self.calculate_fill_price(price, side_norm)
                turnover = fill_price * qty
                fees = self.calculate_fees(turnover, side_norm)

                # 2. Defense in depth: Hard risk checks
                if not skip_risk_checks:
                    if side_norm == "BUY":
                        # Check concurrent positions cap
                        if symbol not in positions and len(positions) >= self.max_concurrent_positions:
                            raise ValueError(
                                f"Risk Breach: Max concurrent positions ({self.max_concurrent_positions}) reached"
                            )
                        # Check capital per trade cap
                        trade_cost = turnover + fees
                        max_allowed_cost = current_balance * self.max_capital_per_trade_pct
                        if trade_cost > max_allowed_cost:
                            raise ValueError(
                                f"Risk Breach: Trade cost {trade_cost:.2f} exceeds {self.max_capital_per_trade_pct*100}% capital cap ({max_allowed_cost:.2f})"
                            )
                        # Solvency check
                        if trade_cost > current_balance:
                            raise ValueError(
                                f"Insufficient Funds: Need {trade_cost:.2f}, balance is {current_balance:.2f}"
                            )
                    elif side_norm == "SELL":
                        if current_pos["qty"] < qty:
                            raise ValueError(
                                f"Position Shortage: Attempting to sell {qty} of {symbol}, holding {current_pos['qty']}"
                            )

                # 3. Compute ledger impact
                if side_norm == "BUY":
                    cash_delta = round(-(turnover + fees), 4)
                else:
                    cash_delta = round(+(turnover - fees), 4)

                balance_after = round(current_balance + cash_delta, 4)

                # 4. Atomic transaction across orders, positions, and ledger
                # Order row
                conn.execute(
                    """
                    INSERT INTO orders (client_order_id, symbol, side, qty, requested_price, fill_price, fees, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (order_id, symbol, side_norm, qty, price, fill_price, fees, "FILLED", ts),
                )

                # Position row
                if side_norm == "BUY":
                    new_qty = current_pos["qty"] + qty
                    new_avg = round(
                        ((current_pos["qty"] * current_pos["avg_price"]) + (qty * fill_price)) / new_qty,
                        4,
                    )
                    conn.execute(
                        """
                        INSERT INTO positions (symbol, qty, avg_price, unrealized_pnl)
                        VALUES (?, ?, ?, 0.0)
                        ON CONFLICT(symbol) DO UPDATE SET
                            qty = excluded.qty,
                            avg_price = excluded.avg_price
                        """,
                        (symbol, new_qty, new_avg),
                    )
                else:
                    new_qty = current_pos["qty"] - qty
                    if new_qty == 0:
                        conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
                    else:
                        conn.execute(
                            "UPDATE positions SET qty = ? WHERE symbol = ?",
                            (new_qty, symbol),
                        )

                # Ledger row
                conn.execute(
                    """
                    INSERT INTO ledger (client_order_id, cash_delta, balance_after, timestamp)
                    VALUES (?, ?, ?, ?)
                    """,
                    (order_id, cash_delta, balance_after, ts),
                )

        return OrderResult(
            client_order_id=order_id,
            symbol=symbol,
            side=side_norm,
            qty=qty,
            requested_price=price,
            fill_price=fill_price,
            fees=fees,
            status="FILLED",
            created_at=ts,
            cash_delta=cash_delta,
            balance_after=balance_after,
        )

    def update_unrealized_pnl(self, current_prices: Dict[str, float]) -> None:
        """Update unrealized PnL for open positions against latest market prices."""
        with self._connection() as conn:
            with conn:
                for symbol, price in current_prices.items():
                    row = conn.execute(
                        "SELECT qty, avg_price FROM positions WHERE symbol = ?", (symbol,)
                    ).fetchone()
                    if row and row["qty"] > 0:
                        unrealized = round((price - row["avg_price"]) * row["qty"], 4)
                        conn.execute(
                            "UPDATE positions SET unrealized_pnl = ? WHERE symbol = ?",
                            (unrealized, symbol),
                        )

    def reconcile_ledger(self) -> Tuple[bool, float, float]:
        """Verify that running balance strictly equals initial_capital + SUM(cash_delta)."""
        with self._connection() as conn:
            sum_delta = conn.execute("SELECT COALESCE(SUM(cash_delta), 0.0) as s FROM ledger").fetchone()["s"]
            latest_balance = self.get_current_balance(conn)
            expected_balance = round(self.initial_capital + float(sum_delta), 4)
            is_consistent = abs(latest_balance - expected_balance) < 1e-4
            return is_consistent, latest_balance, expected_balance

    @classmethod
    def create_backtest_engine(
        cls,
        db_path: Path | str = DEFAULT_BACKTEST_DB_PATH,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        initial_capital: float = 100000.0,
    ) -> PaperTradingEngine:
        """Factory for backtest runs with completely isolated database."""
        target_path = Path(db_path)
        if target_path.exists():
            target_path.unlink()
        return cls(db_path=target_path, config_path=config_path, initial_capital=initial_capital)
