"""Unit and Test Gate verification for Phase 4: Technical Analyst & ML Signal Engine (core/ml_signal.py)."""

import unittest
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

from core.ml_signal import (
    DEFAULT_MODEL_PATH,
    MLSignalEngine,
    backtest_ml_signal,
    compute_technical_indicators,
    extract_features,
)


class TestMLSignalEngine(unittest.TestCase):
    """Test suite verifying Phase 4 Technical Analyst & ML Signal Engine requirements."""

    @classmethod
    def setUpClass(cls):
        # Generate 200 deterministic OHLCV bars
        rng = np.random.default_rng(123)
        n = 200
        initial_price = 20000.0
        returns = rng.normal(0.0002, 0.005, n)
        prices = initial_price * np.exp(np.cumsum(returns))
        highs = prices * (1.0 + rng.uniform(0.001, 0.005, n))
        lows = prices * (1.0 - rng.uniform(0.001, 0.005, n))
        opens = np.roll(prices, 1)
        opens[0] = initial_price
        volumes = rng.integers(1000, 10000, n)
        start_date = datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc)
        timestamps = [start_date + timedelta(minutes=5 * i) for i in range(n)]

        cls.sample_df = pd.DataFrame(
            {
                "open": np.round(opens, 2),
                "high": np.round(highs, 2),
                "low": np.round(lows, 2),
                "close": np.round(prices, 2),
                "volume": volumes,
            },
            index=pd.DatetimeIndex(timestamps, name="time"),
        )
        cls.engine = MLSignalEngine(model_path=DEFAULT_MODEL_PATH)

    def test_technical_indicators_calculation(self):
        """1. Verify RSI, EMA 20/50/200, MACD, Bollinger Bands, ATR via pandas_ta."""
        data_ind = compute_technical_indicators(self.sample_df)

        # RSI(14)
        self.assertIn("rsi_14", data_ind.columns)
        valid_rsi = data_ind["rsi_14"].dropna()
        self.assertTrue(len(valid_rsi) > 0)
        self.assertTrue((valid_rsi >= 0).all() and (valid_rsi <= 100).all())

        # EMAs
        self.assertIn("ema_20", data_ind.columns)
        self.assertIn("ema_50", data_ind.columns)
        self.assertIn("ema_200", data_ind.columns)

        # MACD
        self.assertIn("macd", data_ind.columns)
        self.assertIn("macd_signal", data_ind.columns)
        self.assertIn("macd_hist", data_ind.columns)

        # Bollinger Bands
        self.assertIn("bb_lower", data_ind.columns)
        self.assertIn("bb_upper", data_ind.columns)
        self.assertIn("bb_bandwidth", data_ind.columns)

        # ATR(14)
        self.assertIn("atr_14", data_ind.columns)
        valid_atr = data_ind["atr_14"].dropna()
        self.assertTrue((valid_atr > 0).all())

    def test_committed_model_artifact_loaded_without_retraining(self):
        """2. Verify model artifact exists on disk and is loaded without retraining in hot path."""
        self.assertTrue(DEFAULT_MODEL_PATH.exists())
        self.assertGreater(DEFAULT_MODEL_PATH.stat().st_size, 1000)
        self.assertIsNotNone(self.engine.classifier)

    def test_output_probability_vector_structure(self):
        """3. Verify output returns structured payload with probability vector to feed council."""
        payload = self.engine.generate_signal(
            symbol="NIFTY50",
            timeframe="5min",
            ohlcv_df=self.sample_df,
        )

        self.assertEqual(payload["symbol"], "NIFTY50")
        self.assertEqual(payload["timeframe"], "5min")
        self.assertEqual(payload["status"], "READY")

        # Probability vector
        p_vec = payload["probability_vector"]
        self.assertIn("down", p_vec)
        self.assertIn("neutral", p_vec)
        self.assertIn("up", p_vec)
        total_p = sum(p_vec.values())
        self.assertAlmostEqual(total_p, 1.0, places=2)

        # ML win probability
        ml_prob = payload["ml_probability"]
        self.assertTrue(0.0 <= ml_prob <= 1.0)

        # Indicators dictionary for council reasoning
        ind = payload["indicators"]
        expected_keys = {"rsi_14", "ema_20", "ema_50", "ema_200", "macd", "bb_lower", "bb_upper", "atr_14"}
        self.assertTrue(expected_keys.issubset(ind.keys()))

    def test_gate_backtest_signal_baseline(self):
        """4. Test gate: backtest signal alone against historical bars and verify baseline metrics."""
        results = backtest_ml_signal(
            ohlcv_df=self.sample_df,
            threshold=0.45,
            holding_period=5,
            model_path=DEFAULT_MODEL_PATH,
        )
        self.assertIn("total_trades", results)
        self.assertIn("hit_rate", results)
        self.assertIn("avg_return_pct", results)
        self.assertIn("profit_factor", results)
        self.assertGreaterEqual(results["hit_rate"], 0.0)
        self.assertLessEqual(results["hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
