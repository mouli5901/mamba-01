"""Unit tests for Phase 8: Retrospective Audit Agent (core/audit_agent.py)."""

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from core.audit_agent import AuditAgent, TradeAuditSummary
from core.paper_engine import PaperTradingEngine


class TestAuditAgent(unittest.TestCase):
    """Test suite verifying Phase 8 Audit Agent requirements."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_paper_portfolio.db"
        self.memory_path = Path(self.temp_dir.name) / "test_agent_memory.json"
        self.reports_dir = Path(self.temp_dir.name) / "reports"

        self.paper_engine = PaperTradingEngine(
            db_path=self.db_path,
            initial_capital=100000.0,
        )
        self.audit_agent = AuditAgent(
            db_path=self.db_path,
            memory_path=self.memory_path,
            reports_dir=self.reports_dir,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_audit_agent_read_only_nature(self):
        """2. Verify audit agent has no trade execution capability or callers."""
        self.assertFalse(hasattr(self.audit_agent, "execute_order"))
        self.assertFalse(hasattr(self.audit_agent, "place_order"))
        self.assertFalse(hasattr(self.audit_agent, "submit_trade"))

    def test_end_of_day_audit_generation_and_persistence(self):
        """2. Verify end-of-day audit review compares trade outcomes to logged reasoning."""
        # 1. Simulate Council Deliberation recorded in agent_memory.json
        sig_id = "sig-aud-001"
        council_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_id": sig_id,
            "symbol": "INFY",
            "timeframe": "5min",
            "close_price": 1500.0,
            "ml_probability": 0.65,
            "votes": {
                "claude_technical": "BUY",
                "gpt_quant": "BUY",
                "gemini_macro": "HOLD",
            },
            "consensus_reached": True,
            "consensus_action": "BUY",
            "raw_responses": {
                "claude_technical": {"reasoning": "Breakout"},
                "gpt_quant": {"reasoning": "Quant edge"},
                "gemini_macro": {"reasoning": "Waiting"},
            },
        }
        with open(self.memory_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(council_log) + "\n")

        # 2. Simulate executed round-trip trade
        # Buy 10 @ 1500
        self.paper_engine.execute_order(
            symbol="INFY",
            side="BUY",
            qty=10,
            price=1500.0,
            client_order_id=sig_id,
            skip_risk_checks=True,
        )
        # Sell 10 @ 1550 (profitable round-trip)
        self.paper_engine.execute_order(
            symbol="INFY",
            side="SELL",
            qty=10,
            price=1550.0,
            skip_risk_checks=True,
        )

        # 3. Run Audit Agent
        summary = self.audit_agent.generate_daily_audit()

        self.assertEqual(summary.total_trades, 2)
        self.assertEqual(summary.winning_trades, 1)
        self.assertEqual(summary.losing_trades, 0)
        self.assertEqual(summary.win_rate_pct, 100.0)
        self.assertGreater(summary.net_pnl, 0.0)
        self.assertGreater(summary.total_fees_paid, 0.0)

        # 4. Verify Claude and GPT got positive accuracy credit
        self.assertEqual(summary.agent_accuracy["claude_technical"], 100.0)
        self.assertEqual(summary.agent_accuracy["gpt_quant"], 100.0)

        # 5. Verify report JSON artifact was persisted to reports directory
        saved_reports = list(self.reports_dir.glob("*.json"))
        self.assertEqual(len(saved_reports), 1)


if __name__ == "__main__":
    unittest.main()
