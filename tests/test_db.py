"""Unit and schema tests for Phase 2: Database schema (core/db.py)."""

import unittest
from unittest.mock import MagicMock
from datetime import datetime, timezone
import pandas as pd

from core.db import (
    DDL_CREATE_TICKS_TABLE,
    DDL_CREATE_HYPERTABLE,
    DDL_COMPOSITE_INDEX,
    DDL_CONTINUOUS_AGGREGATES,
    DDL_AGGREGATE_INDEXES,
    DDL_REFRESH_POLICIES,
    build_signal_window_query,
    insert_ticks_batch,
    explain_signal_query,
)


class TestDatabaseSchema(unittest.TestCase):
    """Test suite verifying Phase 2 schema specifications."""

    def test_hypertable_specification(self):
        """1. Hypertable ticks(time, symbol, ltp, volume) partitioned on time, chunk_time_interval => '1 day'."""
        self.assertIn("ticks", DDL_CREATE_TICKS_TABLE)
        self.assertIn("time TIMESTAMPTZ NOT NULL", DDL_CREATE_TICKS_TABLE)
        self.assertIn("symbol VARCHAR(32) NOT NULL", DDL_CREATE_TICKS_TABLE)
        self.assertIn("ltp DOUBLE PRECISION NOT NULL", DDL_CREATE_TICKS_TABLE)
        self.assertIn("volume BIGINT NOT NULL", DDL_CREATE_TICKS_TABLE)

        # Check create_hypertable call
        self.assertIn("create_hypertable", DDL_CREATE_HYPERTABLE)
        self.assertIn("'ticks'", DDL_CREATE_HYPERTABLE)
        self.assertIn("'time'", DDL_CREATE_HYPERTABLE)
        self.assertIn("INTERVAL '1 day'", DDL_CREATE_HYPERTABLE)

    def test_composite_index_specification(self):
        """2. Composite index (symbol, time DESC)."""
        self.assertIn("CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time_desc", DDL_COMPOSITE_INDEX)
        self.assertIn("(symbol, time DESC)", DDL_COMPOSITE_INDEX)

    def test_continuous_aggregates_specification(self):
        """3. Continuous aggregates ohlcv_1min, _5min, _15min rolling up ticks."""
        timeframes = ["1min", "5min", "15min"]
        for tf in timeframes:
            self.assertIn(tf, DDL_CONTINUOUS_AGGREGATES)
            ddl = DDL_CONTINUOUS_AGGREGATES[tf]
            self.assertIn(f"ohlcv_{tf}", ddl)
            self.assertIn("WITH (timescaledb.continuous)", ddl)
            self.assertIn("time_bucket", ddl)
            self.assertIn("first(ltp, time) AS open", ddl)
            self.assertIn("max(ltp) AS high", ddl)
            self.assertIn("min(ltp) AS low", ddl)
            self.assertIn("last(ltp, time) AS close", ddl)
            self.assertIn("sum(volume) AS volume", ddl)
            self.assertIn("GROUP BY bucket, symbol", ddl)

            # Continuous aggregate index on (symbol, bucket DESC)
            self.assertIn(tf, DDL_AGGREGATE_INDEXES)
            self.assertIn("(symbol, bucket DESC)", DDL_AGGREGATE_INDEXES[tf])

            # Refresh policy
            self.assertIn(tf, DDL_REFRESH_POLICIES)
            self.assertIn("add_continuous_aggregate_policy", DDL_REFRESH_POLICIES[tf])

    def test_signal_window_query_uses_continuous_aggregates(self):
        """Verify downstream query reads from continuous aggregates, not raw ticks."""
        q_1min = build_signal_window_query("NIFTY50", "1min", 100)
        self.assertIn("FROM ohlcv_1min", q_1min)
        self.assertIn("WHERE symbol = :symbol", q_1min)
        self.assertIn("ORDER BY bucket DESC", q_1min)
        self.assertIn("LIMIT :limit", q_1min)

        q_5min = build_signal_window_query("BANKNIFTY", "5min", 50)
        self.assertIn("FROM ohlcv_5min", q_5min)

        q_15min = build_signal_window_query("RELIANCE", "15min", 200)
        self.assertIn("FROM ohlcv_15min", q_15min)

    def test_batch_insert_formatting(self):
        """Verify batch tick insertion supports multiple timestamp formats."""
        mock_session = MagicMock()
        ticks = [
            {"time": "2026-09-26T09:15:00Z", "symbol": "NIFTY50", "ltp": 25000.5, "volume": 1000},
            {"time": 1727334900, "symbol": "NIFTY50", "ltp": 25002.0, "volume": 1200},
            {"time": datetime(2026, 9, 26, 9, 15, 2, tzinfo=timezone.utc), "symbol": "NIFTY50", "ltp": 25001.0, "volume": 500},
        ]
        inserted_count = insert_ticks_batch(ticks, session=mock_session)
        self.assertEqual(inserted_count, 3)
        self.assertTrue(mock_session.execute.called)

    def test_explain_analyze_test_gate_index_scan(self):
        """4. Test gate: plan with Index Scan passes validation."""
        mock_session = MagicMock()
        # Mock EXPLAIN (ANALYZE, FORMAT JSON) plan returning Index Scan
        mock_plan = [{
            "Plan": {
                "Node Type": "Index Scan",
                "Relation Name": "ohlcv_1min",
                "Index Name": "idx_ohlcv_1min_symbol_bucket",
                "Plans": [],
            }
        }]
        mock_session.execute.return_value.scalar.return_value = mock_plan

        result = explain_signal_query("NIFTY50", "1min", 100, session=mock_session)
        self.assertTrue(result["has_index_scan"])
        self.assertFalse(result["has_seq_scan"])
        self.assertTrue(result["passed"])

    def test_explain_analyze_test_gate_seq_scan_fails(self):
        """4. Test gate: plan with Seq Scan is detected and rejected."""
        mock_session = MagicMock()
        # Mock EXPLAIN plan returning Seq Scan
        mock_plan = [{
            "Plan": {
                "Node Type": "Seq Scan",
                "Relation Name": "ohlcv_1min",
                "Plans": [],
            }
        }]
        mock_session.execute.return_value.scalar.return_value = mock_plan

        result = explain_signal_query("NIFTY50", "1min", 100, session=mock_session)
        self.assertFalse(result["has_index_scan"])
        self.assertTrue(result["has_seq_scan"])
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
