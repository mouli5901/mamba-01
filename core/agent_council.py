"""LangGraph AI Council for Trading Decisions.

Phase 5 requirements:
1. Fan the three LLM calls out with `asyncio.gather` — concurrent, non-blocking execution.
2. Hard per-call timeout (~8s) with an "abstain" fallback on timeout.
3. Consensus: 2-of-3 required to pass a proposal to risk_manager.py.
4. Log all three raw responses regardless of outcome for audit_agent.py in data/agent_memory.json.
5. Zero direct coupling to paper_engine.py or broker_execution.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, START, StateGraph

from config.settings import ANTHROPIC_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_PATH = Path(__file__).resolve().parent.parent / "data" / "agent_memory.json"
DEFAULT_TIMEOUT_SECONDS = 8.0


class AgentVote(TypedDict):
    agent: str
    vote: str  # "BUY", "SELL", "HOLD", "ABSTAIN"
    confidence: float  # 0.0 - 1.0
    reasoning: str
    timed_out: bool
    raw_response: Optional[Dict[str, Any]]


class CouncilProposal(TypedDict):
    proposal_id: str
    signal_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    requested_price: float
    confidence: float
    votes: Dict[str, str]
    timestamp: str
    reasoning_summary: str


class CouncilState(TypedDict):
    signal_id: str
    symbol: str
    timeframe: str
    candle_ts: str
    close_price: float
    indicators: Dict[str, Any]
    ml_probability: float
    probability_vector: Dict[str, float]
    raw_responses: Dict[str, Any]
    votes: Dict[str, str]
    agent_results: Dict[str, AgentVote]
    consensus_reached: bool
    consensus_action: Optional[str]
    proposal: Optional[CouncilProposal]
    audit_logged: bool


# ── Agent Prompt Builders & Default LLM Implementations ──────


def build_system_prompt(role: str) -> str:
    return (
        f"You are the {role} in an institutional AI Trading Council. "
        "Analyze the provided technical indicators, ML win-probability vector, and price action. "
        "Return a valid JSON object with EXACTLY three fields:\n"
        '1. "vote": one of "BUY", "SELL", "HOLD"\n'
        '2. "confidence": float between 0.0 and 1.0\n'
        '3. "reasoning": concise explanation (under 50 words).\n'
        "Do NOT include markdown formatting or backticks, only pure JSON."
    )


def build_user_context(context: Dict[str, Any]) -> str:
    return (
        f"Symbol: {context.get('symbol')}\n"
        f"Timeframe: {context.get('timeframe')}\n"
        f"Candle Close Price: {context.get('close')}\n"
        f"ML Win Probability: {context.get('ml_probability')}\n"
        f"ML Probability Vector: {context.get('probability_vector')}\n"
        f"Technical Indicators: {json.dumps(context.get('indicators', {}), indent=2)}"
    )


def _parse_llm_json(raw_text: str) -> Dict[str, Any]:
    """Parse JSON output from LLM, stripping any accidental markdown code fences."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())


class ClaudeTechnicalSpecialist:
    """Agent 1: Technical & Momentum Specialist (Anthropic Claude)."""

    def __init__(self, api_key: str = ANTHROPIC_API_KEY):
        self.api_key = api_key

    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key or self.api_key.startswith("sk-ant-XXXX"):
            # Fallback heuristic if API key is not configured
            rsi = context.get("indicators", {}).get("rsi_14", 50.0)
            ml_p = context.get("ml_probability", 0.5)
            if rsi < 35 and ml_p > 0.50:
                return {"vote": "BUY", "confidence": 0.75, "reasoning": "RSI oversold with favorable ML win probability."}
            elif rsi > 70 and ml_p < 0.45:
                return {"vote": "SELL", "confidence": 0.70, "reasoning": "RSI overbought with downward momentum."}
            return {"vote": "HOLD", "confidence": 0.60, "reasoning": "Indicators neutral."}

        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model_name="claude-3-5-sonnet-20241022", api_key=self.api_key, max_tokens=150)
        prompt = f"{build_system_prompt('Technical Specialist')}\n\n{build_user_context(context)}"
        response = llm.invoke(prompt)
        parsed = _parse_llm_json(response.content)
        parsed["raw"] = response.content
        return parsed


class GPTQuantSpecialist:
    """Agent 2: Quantitative & Risk Edge Specialist (OpenAI GPT)."""

    def __init__(self, api_key: str = OPENAI_API_KEY):
        self.api_key = api_key

    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key or self.api_key.startswith("sk-XXXX"):
            ml_p = context.get("ml_probability", 0.5)
            if ml_p >= 0.52:
                return {"vote": "BUY", "confidence": ml_p, "reasoning": f"Quant edge: win prob {ml_p:.2f} > threshold."}
            elif ml_p <= 0.42:
                return {"vote": "SELL", "confidence": 1.0 - ml_p, "reasoning": f"Quant risk: win prob {ml_p:.2f} indicates down probability."}
            return {"vote": "HOLD", "confidence": 0.50, "reasoning": "Win probability within neutral band."}

        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=self.api_key, max_tokens=150)
        prompt = f"{build_system_prompt('Quantitative Specialist')}\n\n{build_user_context(context)}"
        response = llm.invoke(prompt)
        parsed = _parse_llm_json(response.content)
        parsed["raw"] = response.content
        return parsed


class GeminiMacroSpecialist:
    """Agent 3: Macro & Volatility Filter Specialist (Google Gemini)."""

    def __init__(self, api_key: str = GOOGLE_API_KEY):
        self.api_key = api_key

    def __call__(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key or self.api_key.startswith("AIzaXXXX"):
            bb_upper = context.get("indicators", {}).get("bb_upper", 0.0)
            close = context.get("close", 0.0)
            ml_p = context.get("ml_probability", 0.5)
            if close <= bb_upper and ml_p > 0.50:
                return {"vote": "BUY", "confidence": 0.70, "reasoning": "Macro/volatility confirms headroom within Bollinger bands."}
            elif close >= bb_upper:
                return {"vote": "HOLD", "confidence": 0.65, "reasoning": "Price pressing upper Bollinger band, awaiting pullback."}
            return {"vote": "HOLD", "confidence": 0.55, "reasoning": "Macro stance neutral."}

        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=self.api_key, max_output_tokens=150)
        prompt = f"{build_system_prompt('Macro & Volatility Specialist')}\n\n{build_user_context(context)}"
        response = llm.invoke(prompt)
        parsed = _parse_llm_json(response.content)
        parsed["raw"] = response.content
        return parsed


# ── Council Coordinator Engine ───────────────────────────────


class AgentCouncil:
    """Coordinates the 3-member AI Council using asyncio.gather and LangGraph."""

    def __init__(
        self,
        claude_agent: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        gpt_agent: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        gemini_agent: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        memory_path: Path | str = DEFAULT_MEMORY_PATH,
    ):
        self.claude_agent = claude_agent or ClaudeTechnicalSpecialist()
        self.gpt_agent = gpt_agent or GPTQuantSpecialist()
        self.gemini_agent = gemini_agent or GeminiMacroSpecialist()
        self.timeout = timeout
        self.memory_path = Path(memory_path)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.graph = self._build_graph()

    async def _execute_agent_with_timeout(
        self,
        agent_name: str,
        agent_callable: Callable[[Dict[str, Any]], Dict[str, Any]],
        context: Dict[str, Any],
    ) -> AgentVote:
        """Run blocking LLM SDK in worker thread with hard timeout (~8s) and 'ABSTAIN' fallback."""
        loop = asyncio.get_running_loop()
        try:
            # Enforce non-blocking synchronous execution via run_in_executor
            future = loop.run_in_executor(None, agent_callable, context)
            result = await asyncio.wait_for(future, timeout=self.timeout)

            vote = str(result.get("vote", "ABSTAIN")).upper()
            if vote not in ("BUY", "SELL", "HOLD"):
                vote = "ABSTAIN"

            return {
                "agent": agent_name,
                "vote": vote,
                "confidence": float(result.get("confidence", 0.5)),
                "reasoning": str(result.get("reasoning", "")),
                "timed_out": False,
                "raw_response": result,
            }

        except asyncio.TimeoutError:
            logger.warning(f"Agent '{agent_name}' timed out after {self.timeout}s. Fallback to ABSTAIN.")
            return {
                "agent": agent_name,
                "vote": "ABSTAIN",
                "confidence": 0.0,
                "reasoning": f"Decision timed out (>{self.timeout}s) - falling back to abstain",
                "timed_out": True,
                "raw_response": {"error": "TimeoutError", "timeout_seconds": self.timeout},
            }
        except Exception as e:
            logger.error(f"Agent '{agent_name}' error: {e}. Fallback to ABSTAIN.")
            return {
                "agent": agent_name,
                "vote": "ABSTAIN",
                "confidence": 0.0,
                "reasoning": f"Execution error: {str(e)}",
                "timed_out": False,
                "raw_response": {"error": str(e)},
            }

    async def _fanout_llm_calls_node(self, state: CouncilState) -> Dict[str, Any]:
        """Node 1: Fan out the 3 LLM calls concurrently using asyncio.gather."""
        context = {
            "symbol": state["symbol"],
            "timeframe": state["timeframe"],
            "close": state["close_price"],
            "ml_probability": state["ml_probability"],
            "probability_vector": state["probability_vector"],
            "indicators": state["indicators"],
        }

        # Fan out all 3 LLM calls concurrently via asyncio.gather
        results: List[AgentVote] = await asyncio.gather(
            self._execute_agent_with_timeout("claude_technical", self.claude_agent, context),
            self._execute_agent_with_timeout("gpt_quant", self.gpt_agent, context),
            self._execute_agent_with_timeout("gemini_macro", self.gemini_agent, context),
        )

        agent_results = {res["agent"]: res for res in results}
        votes = {res["agent"]: res["vote"] for res in results}
        raw_responses = {res["agent"]: res["raw_response"] for res in results}

        return {
            "agent_results": agent_results,
            "votes": votes,
            "raw_responses": raw_responses,
        }

    def _evaluate_consensus_node(self, state: CouncilState) -> Dict[str, Any]:
        """Node 2: Evaluate 2-of-3 consensus required to emit a proposal to risk_manager.py."""
        votes = list(state["votes"].values())
        buy_count = votes.count("BUY")
        sell_count = votes.count("SELL")

        consensus_action: Optional[str] = None
        if buy_count >= 2:
            consensus_action = "BUY"
        elif sell_count >= 2:
            consensus_action = "SELL"

        if consensus_action is not None:
            # Consensus reached: create formal trade proposal for risk_manager.py
            matching_confidences = [
                res["confidence"]
                for res in state["agent_results"].values()
                if res["vote"] == consensus_action
            ]
            avg_conf = sum(matching_confidences) / len(matching_confidences) if matching_confidences else 0.5
            reasonings = [
                f"{res['agent']}: {res['reasoning']}"
                for res in state["agent_results"].values()
                if res["vote"] == consensus_action
            ]

            proposal: CouncilProposal = {
                "proposal_id": str(uuid.uuid4()),
                "signal_id": state["signal_id"],
                "symbol": state["symbol"],
                "side": consensus_action,
                "requested_price": state["close_price"],
                "confidence": round(avg_conf, 4),
                "votes": state["votes"],
                "timestamp": state["candle_ts"],
                "reasoning_summary": " | ".join(reasonings),
            }
            logger.info(f"Consensus REACHED: {consensus_action} on {state['symbol']} (2-of-3 votes).")
            return {
                "consensus_reached": True,
                "consensus_action": consensus_action,
                "proposal": proposal,
            }
        else:
            logger.info(f"Consensus NOT reached on {state['symbol']}. Votes: {state['votes']}.")
            return {
                "consensus_reached": False,
                "consensus_action": None,
                "proposal": None,
            }

    def _audit_logging_node(self, state: CouncilState) -> Dict[str, Any]:
        """Node 3: Log all 3 raw responses and final verdict into data/agent_memory.json."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_id": state["signal_id"],
            "symbol": state["symbol"],
            "timeframe": state["timeframe"],
            "candle_ts": state["candle_ts"],
            "close_price": state["close_price"],
            "ml_probability": state["ml_probability"],
            "votes": state["votes"],
            "consensus_reached": state["consensus_reached"],
            "consensus_action": state["consensus_action"],
            "raw_responses": state["raw_responses"],
            "proposal": state["proposal"],
        }

        try:
            with open(self.memory_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to append to audit log {self.memory_path}: {e}")

        return {"audit_logged": True}

    def _build_graph(self):
        """Construct LangGraph decision workflow."""
        builder = StateGraph(CouncilState)
        builder.add_node("fanout_llm_calls", self._fanout_llm_calls_node)
        builder.add_node("evaluate_consensus", self._evaluate_consensus_node)
        builder.add_node("audit_logging", self._audit_logging_node)

        builder.add_edge(START, "fanout_llm_calls")
        builder.add_edge("fanout_llm_calls", "evaluate_consensus")
        builder.add_edge("evaluate_consensus", "audit_logging")
        builder.add_edge("audit_logging", END)

        return builder.compile()

    async def evaluate_signal(self, signal_payload: Dict[str, Any]) -> CouncilState:
        """Run the AI Council on an incoming ML signal payload from Phase 4."""
        initial_state: CouncilState = {
            "signal_id": signal_payload.get("signal_id", str(uuid.uuid4())),
            "symbol": signal_payload["symbol"],
            "timeframe": signal_payload.get("timeframe", "5min"),
            "candle_ts": signal_payload.get("candle_ts", datetime.now(timezone.utc).isoformat()),
            "close_price": float(signal_payload["close"]),
            "indicators": signal_payload.get("indicators", {}),
            "ml_probability": float(signal_payload.get("ml_probability", 0.5)),
            "probability_vector": signal_payload.get("probability_vector", {}),
            "raw_responses": {},
            "votes": {},
            "agent_results": {},
            "consensus_reached": False,
            "consensus_action": None,
            "proposal": None,
            "audit_logged": False,
        }

        final_state: CouncilState = await self.graph.ainvoke(initial_state)
        return final_state
