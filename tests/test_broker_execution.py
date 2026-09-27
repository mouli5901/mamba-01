"""Unit and Test Gate verification for Phase 7: Zerodha Live Engine (core/broker_execution.py)."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

from core.broker_execution import (
    BrokerExecutionEngine,
    KiteAuthError,
    KiteAuthManager,
)
from core.risk_manager import ExecutionResult


class TestBrokerExecution(unittest.TestCase):
    """Test suite verifying Phase 7 Zerodha Live Engine requirements and test gate."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.session_path = Path(self.temp_dir.name) / ".kite_session.json"
        self.auth_manager = KiteAuthManager(
            api_key="mock_key",
            api_secret="mock_secret",
            session_path=self.session_path,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_live_trading_guard_blocks_unauthorized_runs(self):
        """1. Keep behind LIVE_TRADING: default false blocks any live order."""
        mock_kite = MagicMock()
        engine = BrokerExecutionEngine(
            auth_manager=self.auth_manager,
            kite_client=mock_kite,
            live_trading=False,
        )
        res = engine.place_order(symbol="TCS", side="BUY", qty=1, price=3500.0)
        self.assertFalse(res.success)
        self.assertIn("LIVE_TRADING is false", res.error)
        self.assertFalse(mock_kite.place_order.called)

    def test_daily_token_expiry_and_refresh(self):
        """1. Auth: tokens expire daily (~6 AM IST). Stale sessions are rejected."""
        # 1. Missing session
        with self.assertRaises(KiteAuthError):
            self.auth_manager.get_access_token()

        # 2. Yesterday's session
        yesterday_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        with open(self.session_path, "w", encoding="utf-8") as f:
            json.dump({"access_token": "old_token", "session_date": yesterday_str}, f)

        with self.assertRaises(KiteAuthError) as ctx:
            self.auth_manager.get_access_token()
        self.assertIn("expired", str(ctx.exception))

        # 3. Today's session
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with open(self.session_path, "w", encoding="utf-8") as f:
            json.dump({"access_token": "valid_token", "session_date": today_str}, f)

        token = self.auth_manager.get_access_token()
        self.assertEqual(token, "valid_token")

    def test_idempotent_order_never_blind_retries_on_timeout(self):
        """2. Idempotency: on network error/timeout, check order status by tag before resubmitting."""
        mock_kite = MagicMock()
        order_id = "test-order-uuid-999"
        tag = order_id[:20]

        # Simulate first place_order call raising a network timeout exception
        mock_kite.place_order.side_effect = TimeoutError("Connection reset / gateway timeout")

        # Pre-check sees no order; post-timeout check sees the order on server
        mock_kite.orders.side_effect = [
            [],  # Pre-check before first attempt
            [    # Check after timeout
                {
                    "order_id": "zerodha-order-777",
                    "tag": tag,
                    "status": "COMPLETE",
                    "average_price": 2500.0,
                }
            ],
        ]

        engine = BrokerExecutionEngine(
            auth_manager=self.auth_manager,
            kite_client=mock_kite,
            live_trading=True,
        )

        from unittest.mock import patch
        with patch("time.sleep"):
            res = engine.place_order(
                symbol="INFY",
                side="BUY",
                qty=5,
                price=2500.0,
                client_order_id=order_id,
            )

        # Execution succeeded because order was confirmed via status check
        self.assertTrue(res.success)
        self.assertEqual(res.order_id, "zerodha-order-777")
        # place_order was only attempted once, and status check prevented duplicate submission
        self.assertEqual(mock_kite.place_order.call_count, 1)

    def test_reconciliation_detects_position_drift(self):
        """3. Reconciliation: diff Kite's actual positions against internal ledger, alert on drift."""
        mock_kite = MagicMock()
        mock_kite.positions.return_value = {
            "net": [
                {"tradingsymbol": "RELIANCE", "quantity": 15},  # Broker holds 15
                {"tradingsymbol": "INFY", "quantity": 10},      # Broker holds 10
            ]
        }

        engine = BrokerExecutionEngine(
            auth_manager=self.auth_manager,
            kite_client=mock_kite,
            live_trading=True,
        )

        # Internal ledger holds 10 RELIANCE (drift of +5) and 10 INFY (in sync)
        internal_positions = {
            "RELIANCE": {"qty": 10, "avg_price": 2800.0},
            "INFY": {"qty": 10, "avg_price": 1500.0},
        }

        recon = engine.reconcile_positions(internal_positions)
        self.assertFalse(recon["in_sync"])
        self.assertEqual(len(recon["mismatches"]), 1)
        mismatch = recon["mismatches"][0]
        self.assertEqual(mismatch["symbol"], "RELIANCE")
        self.assertEqual(mismatch["internal_qty"], 10)
        self.assertEqual(mismatch["broker_actual_qty"], 15)
        self.assertEqual(mismatch["drift"], 5)

    def test_gate_paper_mode_only_safeguard(self):
        """4. Test gate: system enforces paper-mode-only when LIVE_TRADING is false."""
        engine = BrokerExecutionEngine(
            auth_manager=self.auth_manager,
            live_trading=False,
        )
        self.assertFalse(engine.live_trading)


if __name__ == "__main__":
    unittest.main()
