"""Unit and Test Gate verification for Phase 5: LangGraph AI Council (core/agent_council.py)."""

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from core.agent_council import AgentCouncil
from core.ml_signal import MLSignalEngine


class TestAgentCouncil(unittest.TestCase):
    """Test suite verifying Phase 5 LangGraph AI Council requirements."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.memory_path = Path(self.temp_dir.name) / "agent_memory.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_gate_mocked_slow_llm_timeout_and_consensus(self):
        """4. Test gate: replay signal with a mocked slow LLM.

        Confirm that:
        1. Slow LLM times out and falls back to ABSTAIN without blocking.
        2. 2-of-3 consensus logic still passes the proposal.
        """
        async def run_test():
            # Fast Agent 1: Votes BUY
            def fast_claude(ctx):
                return {"vote": "BUY", "confidence": 0.85, "reasoning": "Strong momentum breakout."}

            # Fast Agent 2: Votes BUY
            def fast_gpt(ctx):
                return {"vote": "BUY", "confidence": 0.80, "reasoning": "High ML probability confirms edge."}

            # Slow Agent 3: Simulates stuck LLM taking 5.0 seconds
            def slow_gemini(ctx):
                time.sleep(5.0)
                return {"vote": "BUY", "confidence": 0.90, "reasoning": "Late response."}

            # Hard timeout set to 0.3s for rapid test execution
            council = AgentCouncil(
                claude_agent=fast_claude,
                gpt_agent=fast_gpt,
                gemini_agent=slow_gemini,
                timeout=0.3,
                memory_path=self.memory_path,
            )

            signal_payload = {
                "signal_id": "test-sig-001",
                "symbol": "NIFTY50",
                "timeframe": "5min",
                "close": 24200.0,
                "ml_probability": 0.65,
                "probability_vector": {"down": 0.15, "neutral": 0.20, "up": 0.65},
                "indicators": {"rsi_14": 42.5, "ema_20": 24150.0},
            }

            t0 = time.time()
            result = await council.evaluate_signal(signal_payload)
            elapsed = time.time() - t0

            # 1. Total execution time must be bounded by timeout (well under the 5s sleep)
            self.assertLess(elapsed, 2.0)

            # 2. Check that the slow agent timed out and fell back to ABSTAIN
            gemini_result = result["agent_results"]["gemini_macro"]
            self.assertTrue(gemini_result["timed_out"])
            self.assertEqual(gemini_result["vote"], "ABSTAIN")
            self.assertIn("timed out", gemini_result["reasoning"])

            # 3. Check 2-of-3 consensus succeeded because Claude and GPT both voted BUY
            self.assertTrue(result["consensus_reached"])
            self.assertEqual(result["consensus_action"], "BUY")
            self.assertIsNotNone(result["proposal"])

            proposal = result["proposal"]
            self.assertEqual(proposal["side"], "BUY")
            self.assertEqual(proposal["requested_price"], 24200.0)
            self.assertAlmostEqual(proposal["confidence"], 0.825, places=3)
            self.assertEqual(proposal["votes"]["gemini_macro"], "ABSTAIN")

            # 4. Confirm audit log recorded all 3 raw responses
            self.assertTrue(self.memory_path.exists())
            with open(self.memory_path, "r", encoding="utf-8") as f:
                logs = [json.loads(line) for line in f]
            self.assertEqual(len(logs), 1)
            self.assertEqual(logs[0]["consensus_action"], "BUY")
            self.assertIn("error", logs[0]["raw_responses"]["gemini_macro"])

        asyncio.run(run_test())

    def test_no_consensus_creates_no_proposal(self):
        """Verify that when 2-of-3 consensus is not reached, no proposal is emitted."""
        async def run_test():
            def agent1(ctx):
                return {"vote": "BUY", "confidence": 0.70, "reasoning": "Bullish"}

            def agent2(ctx):
                return {"vote": "SELL", "confidence": 0.75, "reasoning": "Bearish divergence"}

            def agent3(ctx):
                return {"vote": "HOLD", "confidence": 0.50, "reasoning": "Neutral"}

            council = AgentCouncil(
                claude_agent=agent1,
                gpt_agent=agent2,
                gemini_agent=agent3,
                timeout=1.0,
                memory_path=self.memory_path,
            )

            signal_payload = {
                "symbol": "BANKNIFTY",
                "close": 51000.0,
                "ml_probability": 0.48,
            }

            result = await council.evaluate_signal(signal_payload)
            self.assertFalse(result["consensus_reached"])
            self.assertIsNone(result["consensus_action"])
            self.assertIsNone(result["proposal"])

        asyncio.run(run_test())

    def test_sell_consensus_triggers_sell_proposal(self):
        """Verify 2-of-3 SELL votes trigger a SELL proposal."""
        async def run_test():
            def agent1(ctx):
                return {"vote": "SELL", "confidence": 0.80, "reasoning": "Overbought breakdown"}

            def agent2(ctx):
                return {"vote": "SELL", "confidence": 0.85, "reasoning": "Downside quant probability"}

            def agent3(ctx):
                return {"vote": "HOLD", "confidence": 0.60, "reasoning": "Waiting"}

            council = AgentCouncil(
                claude_agent=agent1,
                gpt_agent=agent2,
                gemini_agent=agent3,
                timeout=1.0,
                memory_path=self.memory_path,
            )

            signal_payload = {
                "symbol": "RELIANCE",
                "close": 2900.0,
                "ml_probability": 0.35,
            }

            result = await council.evaluate_signal(signal_payload)
            self.assertTrue(result["consensus_reached"])
            self.assertEqual(result["consensus_action"], "SELL")
            self.assertIsNotNone(result["proposal"])
            self.assertEqual(result["proposal"]["side"], "SELL")

        asyncio.run(run_test())

    def test_replay_phase4_signals_through_council(self):
        """Verify seamless pipeline integration between Phase 4 MLSignalEngine and Phase 5 Council."""
        async def run_test():
            # Create synthetic bars
            rng = np.random.default_rng(99)
            prices = 1000.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.003, 100)))
            df = pd.DataFrame(
                {
                    "open": prices,
                    "high": prices * 1.002,
                    "low": prices * 0.998,
                    "close": prices,
                    "volume": [5000] * 100,
                },
                index=pd.date_range("2026-01-01", periods=100, freq="5min"),
            )

            # Generate Phase 4 signal
            engine = MLSignalEngine()
            signal = engine.generate_signal("TATAMOTORS", "5min", ohlcv_df=df)

            # Feed directly into Council
            council = AgentCouncil(
                timeout=1.0,
                memory_path=self.memory_path,
            )
            result = await council.evaluate_signal(signal)

            self.assertEqual(result["symbol"], "TATAMOTORS")
            self.assertIn("votes", result)
            self.assertEqual(len(result["votes"]), 3)
            self.assertTrue(result["audit_logged"])

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
