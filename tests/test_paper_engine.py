"""Unit and Test Gate verification for Phase 3: Paper Trading Engine (core/paper_engine.py)."""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.paper_engine import PaperTradingEngine, OrderResult


class TestPaperTradingEngine(unittest.TestCase):
    """Test suite verifying Phase 3 Paper Trading Engine requirements."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_paper_portfolio.db"
        self.engine = PaperTradingEngine(
            db_path=self.db_path,
            initial_capital=100000.0,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sqlite_schema(self):
        """1. Verify SQLite schema: orders, positions, ledger with client_order_id."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Check orders table columns
            cursor.execute("PRAGMA table_info(orders);")
            orders_cols = {row[1] for row in cursor.fetchall()}
            expected_orders = {
                "client_order_id", "symbol", "side", "qty",
                "requested_price", "fill_price", "fees", "status", "created_at"
            }
            self.assertTrue(expected_orders.issubset(orders_cols))

            # Check positions table columns
            cursor.execute("PRAGMA table_info(positions);")
            positions_cols = {row[1] for row in cursor.fetchall()}
            expected_positions = {"symbol", "qty", "avg_price", "unrealized_pnl"}
            self.assertTrue(expected_positions.issubset(positions_cols))

            # Check ledger table columns
            cursor.execute("PRAGMA table_info(ledger);")
            ledger_cols = {row[1] for row in cursor.fetchall()}
            expected_ledger = {"id", "client_order_id", "cash_delta", "balance_after", "timestamp"}
            self.assertTrue(expected_ledger.issubset(ledger_cols))
        finally:
            conn.close()

    def test_idempotent_order_execution(self):
        """Verify client_order_id ensures duplicate submissions do not duplicate ledger/positions."""
        order_id = "test-uuid-12345"
        res1 = self.engine.execute_order(
            symbol="INFY",
            side="BUY",
            qty=1,
            price=1500.0,
            client_order_id=order_id,
            skip_risk_checks=True,
        )

        bal1 = self.engine.get_current_balance()
        pos1 = self.engine.get_positions()

        # Resubmit with exact same client_order_id
        res2 = self.engine.execute_order(
            symbol="INFY",
            side="BUY",
            qty=1,
            price=1500.0,
            client_order_id=order_id,
            skip_risk_checks=True,
        )

        bal2 = self.engine.get_current_balance()
        pos2 = self.engine.get_positions()

        self.assertEqual(res1.client_order_id, res2.client_order_id)
        self.assertEqual(bal1, bal2)
        self.assertEqual(pos1["INFY"]["qty"], pos2["INFY"]["qty"])

    def test_slippage_and_fee_modeling(self):
        """2. Verify slippage and Indian transaction charges on BUY and SELL."""
        # 0.02% slippage + 0.005% half spread = 0.025% drag
        buy_fill = self.engine.calculate_fill_price(1000.0, "BUY")
        self.assertEqual(buy_fill, 1000.25)

        sell_fill = self.engine.calculate_fill_price(1000.0, "SELL")
        self.assertEqual(sell_fill, 999.75)

        # Fee computation check on BUY 10000 INR turnover
        fees_buy = self.engine.calculate_fees(10000.0, "BUY")
        # Brokerage = 20, STT=10, Exch=0.345, SEBI=0.01, Stamp=0.30, GST=0.18*(20+0.345+0.01)=3.6639
        # Sum = 34.3189 -> rounded to 34.3189
        self.assertAlmostEqual(fees_buy, 34.3189, places=3)

    def test_defense_in_depth_risk_rejections(self):
        """Verify engine independently blocks risk-breaching proposals."""
        # Attempting to sell unowned stock
        with self.assertRaises(ValueError) as ctx:
            self.engine.execute_order("RELIANCE", "SELL", 5, 2500.0)
        self.assertIn("Position Shortage", str(ctx.exception))

        # Attempting trade exceeding 2% capital cap (2000 INR on 100k balance)
        with self.assertRaises(ValueError) as ctx:
            self.engine.execute_order("RELIANCE", "BUY", 10, 2500.0)  # 25,000 INR > 2,000 INR cap
        self.assertIn("Risk Breach", str(ctx.exception))

    def test_gate_scripted_replay_matches_hand_computed_balance(self):
        """3. Test gate: Scripted buy/sell sequence asserting exact hand-computed ledger balance."""
        # Initial Balance: 100,000.00
        # Trade 1: BUY 10 shares of TATAMOTORS at 1000.00 (skip_risk_checks=True for precise hand test)
        # Expected Fill Price: 1000.25, Turnover: 10002.50
        # Expected Fees: 34.3216
        # Cash Delta 1: -10036.8216
        # Expected Balance 1: 89963.1784
        res1 = self.engine.execute_order(
            symbol="TATAMOTORS",
            side="BUY",
            qty=10,
            price=1000.0,
            skip_risk_checks=True,
        )
        self.assertEqual(res1.fill_price, 1000.25)
        self.assertEqual(res1.fees, 34.3216)
        self.assertEqual(res1.cash_delta, -10036.8216)
        self.assertEqual(res1.balance_after, 89963.1784)

        # Trade 2: SELL 10 shares of TATAMOTORS at 1050.00
        # Expected Fill Price: 1049.7375, Turnover: 10497.375
        # Expected Fees: 34.5371
        # Cash Delta 2: +10462.8379
        # Expected Balance 2: 100426.0163
        res2 = self.engine.execute_order(
            symbol="TATAMOTORS",
            side="SELL",
            qty=10,
            price=1050.0,
            skip_risk_checks=True,
        )
        self.assertEqual(res2.fill_price, 1049.7375)
        self.assertEqual(res2.fees, 34.5371)
        self.assertEqual(res2.cash_delta, 10462.8379)
        self.assertEqual(res2.balance_after, 100426.0163)

        # Final ledger assertion
        final_balance = self.engine.get_current_balance()
        self.assertEqual(final_balance, 100426.0163)

        # Assert positions are fully flattened
        open_positions = self.engine.get_positions()
        self.assertNotIn("TATAMOTORS", open_positions)

        # Assert ledger reconciliation passes
        is_consistent, actual, expected = self.engine.reconcile_ledger()
        self.assertTrue(is_consistent)
        self.assertEqual(actual, 100426.0163)
        self.assertEqual(expected, 100426.0163)


if __name__ == "__main__":
    unittest.main()
