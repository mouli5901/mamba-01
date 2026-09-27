"""Unit and Test Gate verification for Phase 6: Risk Manager (core/risk_manager.py)."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from core.paper_engine import PaperTradingEngine
from core.risk_manager import ExecutionResult, RiskCheckResult, RiskManager


class TestRiskManager(unittest.TestCase):
    """Test suite verifying Phase 6 Risk Manager requirements and hard-cap test gate."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_paper_portfolio.db"
        self.paper_engine = PaperTradingEngine(
            db_path=self.db_path,
            initial_capital=100000.0,
        )
        self.risk_manager = RiskManager(
            paper_engine=self.paper_engine,
            live_trading=False,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_gate_reject_breach_capital_per_trade_cap(self):
        """4. Test gate: proposal breaching max 2% capital per trade is rejected."""
        # Balance = 100,000 INR -> Max allowed trade cost = 2,000 INR
        # Proposal requests 10 shares @ 500 INR = 5,000 INR (> 2,000 cap)
        proposal = {
            "symbol": "INFY",
            "side": "BUY",
            "requested_price": 500.0,
            "qty": 10,
            "confidence": 0.90,
        }
        res = self.risk_manager.evaluate_proposal(proposal, current_balance=100000.0)
        self.assertFalse(res.approved)
        self.assertIn("capital cap", res.rejection_reason)

    def test_gate_reject_breach_max_concurrent_positions(self):
        """4. Test gate: proposal breaching max concurrent positions (5) is rejected."""
        # Mock 5 open positions
        mock_positions = {
            f"SYM_{i}": {"qty": 10, "avg_price": 100.0, "unrealized_pnl": 0.0}
            for i in range(5)
        }
        proposal = {
            "symbol": "NEW_SYM",
            "side": "BUY",
            "requested_price": 100.0,
            "confidence": 0.95,
        }
        res = self.risk_manager.evaluate_proposal(
            proposal,
            current_balance=100000.0,
            open_positions=mock_positions,
        )
        self.assertFalse(res.approved)
        self.assertIn("Max concurrent positions limit (5) reached", res.rejection_reason)

    def test_gate_reject_breach_max_daily_loss_kill_switch(self):
        """4. Test gate: proposal when daily loss exceeds 5% is rejected via kill switch."""
        # Starting capital: 100,000. Current balance: 94,000 (6% daily drawdown > 5% limit)
        self.risk_manager.daily_starting_balance = 100000.0
        proposal = {
            "symbol": "TCS",
            "side": "BUY",
            "requested_price": 100.0,
            "qty": 10,
            "confidence": 0.85,
        }
        res = self.risk_manager.evaluate_proposal(proposal, current_balance=94000.0)
        self.assertFalse(res.approved)
        self.assertTrue(res.kill_switch_triggered)
        self.assertTrue(self.risk_manager.kill_switch_active)
        self.assertIn("Max daily loss limit breached", res.rejection_reason)

    def test_gate_reject_breach_max_consecutive_losses(self):
        """4. Test gate: proposal after 4 consecutive losses is rejected."""
        for _ in range(4):
            self.risk_manager.record_trade_outcome(pnl=-150.0)

        proposal = {
            "symbol": "HDFCBANK",
            "side": "BUY",
            "requested_price": 1500.0,
            "qty": 1,
            "confidence": 0.90,
        }
        res = self.risk_manager.evaluate_proposal(proposal, current_balance=100000.0)
        self.assertFalse(res.approved)
        self.assertTrue(res.kill_switch_triggered)
        self.assertIn("Max consecutive losses", res.rejection_reason)

    def test_dead_mans_switch_stale_heartbeat_flatten_paper(self):
        """2. Verify dead-man's switch flattens positions in paper mode when heartbeat is stale."""
        # Open a position first
        self.paper_engine.execute_order(
            symbol="SBIN",
            side="BUY",
            qty=10,
            price=800.0,
            skip_risk_checks=True,
        )
        self.assertIn("SBIN", self.paper_engine.get_positions())

        # Simulate heartbeat 45 seconds old (threshold is 30s)
        now = time.time()
        self.risk_manager.record_heartbeat(timestamp=now - 45.0)

        # Trigger dead-man's switch
        triggered = self.risk_manager.check_dead_mans_switch(
            current_prices={"SBIN": 810.0},
            current_ts=now,
        )
        self.assertTrue(triggered)

        # Open positions must be liquidated
        self.assertEqual(len(self.paper_engine.get_positions()), 0)

        # New proposal must be rejected while heartbeat is stale
        proposal = {"symbol": "SBIN", "side": "BUY", "requested_price": 800.0}
        res = self.risk_manager.evaluate_proposal(proposal, current_ts=now)
        self.assertFalse(res.approved)
        self.assertIn("Dead-Man's Switch", res.rejection_reason)

    def test_dead_mans_switch_live_mode_alerts_and_halts(self):
        """2. Verify dead-man's switch halts live trading without placing unauthorized orders."""
        mock_broker = MagicMock()
        live_rm = RiskManager(
            paper_engine=self.paper_engine,
            broker_execution=mock_broker,
            live_trading=True,
        )

        now = time.time()
        live_rm.record_heartbeat(timestamp=now - 50.0)

        triggered = live_rm.check_dead_mans_switch(current_ts=now)
        self.assertTrue(triggered)
        self.assertTrue(live_rm.halted_due_to_stale_heartbeat)
        self.assertFalse(mock_broker.place_order.called)

    def test_reject_position_shortage_on_sell(self):
        """Verify sell proposals exceeding held quantities are rejected."""
        proposal = {
            "symbol": "WIPRO",
            "side": "SELL",
            "requested_price": 500.0,
            "qty": 5,
        }
        res = self.risk_manager.evaluate_proposal(
            proposal,
            open_positions={},  # 0 held
        )
        self.assertFalse(res.approved)
        self.assertIn("Position shortage", res.rejection_reason)

    def test_live_trading_blocks_without_manual_confirmation(self):
        """Verify Mode B (live Zerodha) strictly requires manual confirmation."""
        mock_broker = MagicMock()
        live_rm = RiskManager(
            paper_engine=self.paper_engine,
            broker_execution=mock_broker,
            live_trading=True,
        )

        proposal = {
            "symbol": "NIFTY50",
            "side": "BUY",
            "requested_price": 100.0,
            "qty": 10,
        }

        # Attempt to execute live trade with manual_confirmation_received=False
        result = live_rm.route_and_execute(proposal, manual_confirmation_received=False)
        self.assertFalse(result.success)
        self.assertIn("manual operator confirmation was NOT provided", result.error)
        self.assertFalse(mock_broker.place_order.called)

        # With explicit manual confirmation
        mock_broker.place_order.return_value = ExecutionResult(
            success=True, mode="LIVE", symbol="NIFTY50", side="BUY", qty=10, price=100.0
        )
        confirmed_result = live_rm.route_and_execute(proposal, manual_confirmation_received=True)
        self.assertTrue(confirmed_result.success)
        self.assertTrue(mock_broker.place_order.called)

    def test_valid_proposal_executes_in_paper_mode(self):
        """Verify valid proposal within limits executes cleanly in default Mode A."""
        # 2% of 100k is 2000 INR. 10 shares @ 150 = 1500 INR (well within limit)
        proposal = {
            "proposal_id": "prop-001",
            "symbol": "ITC",
            "side": "BUY",
            "requested_price": 150.0,
            "qty": 10,
            "confidence": 0.85,
        }
        result = self.risk_manager.route_and_execute(proposal)
        self.assertTrue(result.success)
        self.assertEqual(result.mode, "PAPER")
        self.assertEqual(result.symbol, "ITC")
        self.assertEqual(result.qty, 10)
        self.assertIn("ITC", self.paper_engine.get_positions())


if __name__ == "__main__":
    unittest.main()
