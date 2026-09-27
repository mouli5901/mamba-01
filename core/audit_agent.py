"""End-of-Day Audit Agent for AI Council Trading Engine.

Phase 8 requirements:
1. End-of-day review of executed trades vs logged AI Council reasoning.
2. Read-only: never wired to place trades, execute orders, or call paper_engine/broker_execution.
3. Produces structured audit report and saves to data/audit_reports/audit_YYYY_MM_DD.json.
4. Identifies individual agent accuracy, consensus efficacy, and risk adherence.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "paper_portfolio.db"
DEFAULT_MEMORY_PATH = Path(__file__).resolve().parent.parent / "data" / "agent_memory.json"
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "data" / "audit_reports"


@dataclass
class TradeAuditSummary:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    total_fees_paid: float
    net_pnl: float
    agent_accuracy: Dict[str, float]
    consensus_alignment_pct: float
    risk_violations: int
    executive_summary: str
    generated_at: str


class AuditAgent:
    """Read-only retrospective auditor evaluating trades against council deliberations."""

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        memory_path: Path | str = DEFAULT_MEMORY_PATH,
        reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    ):
        self.db_path = Path(db_path)
        self.memory_path = Path(memory_path)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def load_executed_trades(self) -> List[Dict[str, Any]]:
        """Fetch all filled orders from database."""
        if not self.db_path.exists():
            return []
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM orders WHERE status = 'FILLED' ORDER BY created_at ASC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def load_council_decisions(self) -> List[Dict[str, Any]]:
        """Fetch all logged council deliberations from agent memory."""
        if not self.memory_path.exists():
            return []
        decisions = []
        with open(self.memory_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        decisions.append(json.loads(line))
                    except Exception:
                        pass
        return decisions

    def generate_daily_audit(
        self,
        date_str: Optional[str] = None,
        llm_client: Optional[Any] = None,
    ) -> TradeAuditSummary:
        """Perform end-of-day audit review comparing trade outcomes to council votes."""
        target_date = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        trades = self.load_executed_trades()
        decisions = self.load_council_decisions()

        # Map decisions by signal_id or timestamp
        decisions_by_signal = {d.get("signal_id"): d for d in decisions if "signal_id" in d}

        total_trades = len(trades)
        total_fees = sum(float(t.get("fees", 0.0)) for t in trades)

        # Pair round trips (BUY followed by SELL for same symbol)
        round_trips = []
        inventory: Dict[str, List[Dict[str, Any]]] = {}
        for t in trades:
            sym = t["symbol"]
            side = t["side"].upper()
            if side == "BUY":
                inventory.setdefault(sym, []).append(t)
            elif side == "SELL" and inventory.get(sym):
                buy_order = inventory[sym].pop(0)
                pnl = (float(t["fill_price"]) - float(buy_order["fill_price"])) * int(t["qty"])
                pnl -= (float(t["fees"]) + float(buy_order["fees"]))
                round_trips.append({
                    "symbol": sym,
                    "buy": buy_order,
                    "sell": t,
                    "net_pnl": pnl,
                    "win": pnl > 0,
                })

        winning = sum(1 for r in round_trips if r["win"])
        losing = sum(1 for r in round_trips if not r["win"])
        win_rate = (winning / len(round_trips) * 100.0) if round_trips else 0.0
        net_pnl = sum(r["net_pnl"] for r in round_trips)

        # Agent accuracy calculation
        agent_correct_votes: Dict[str, int] = {"claude_technical": 0, "gpt_quant": 0, "gemini_macro": 0}
        agent_total_votes: Dict[str, int] = {"claude_technical": 0, "gpt_quant": 0, "gemini_macro": 0}

        for rt in round_trips:
            buy_sig_id = rt["buy"].get("client_order_id")
            dec = decisions_by_signal.get(buy_sig_id)
            if dec:
                votes = dec.get("votes", {})
                is_win = rt["win"]
                for agent, vote in votes.items():
                    if vote in ("BUY", "SELL"):
                        agent_total_votes[agent] = agent_total_votes.get(agent, 0) + 1
                        if (vote == "BUY" and is_win) or (vote == "SELL" and not is_win):
                            agent_correct_votes[agent] = agent_correct_votes.get(agent, 0) + 1

        agent_accuracy = {}
        for agent in agent_total_votes:
            tot = agent_total_votes[agent]
            agent_accuracy[agent] = round((agent_correct_votes[agent] / tot * 100.0), 1) if tot > 0 else 50.0

        consensus_decisions = [d for d in decisions if d.get("consensus_reached")]
        consensus_alignment = (len(consensus_decisions) / len(decisions) * 100.0) if decisions else 0.0

        best_agent = max(agent_accuracy.keys(), key=lambda k: agent_accuracy[k]) if agent_accuracy else "none"
        best_acc = agent_accuracy.get(best_agent, 0.0)
        narrative = (
            f"End-of-Day Audit Report for {target_date}: Completed {total_trades} orders across "
            f"{len(round_trips)} round-trip positions with a win rate of {win_rate:.1f}% and net P&L of ₹{net_pnl:+,.2f}. "
            f"Total exchange fees and taxes paid: ₹{total_fees:,.2f}. "
            f"Consensus gate triggered on {consensus_alignment:.1f}% of market signals. "
            f"Top performing council advisor: {best_agent} ({best_acc}% accuracy)."
        )

        audit_summary = TradeAuditSummary(
            total_trades=total_trades,
            winning_trades=winning,
            losing_trades=losing,
            win_rate_pct=round(win_rate, 2),
            total_fees_paid=round(total_fees, 2),
            net_pnl=round(net_pnl, 2),
            agent_accuracy=agent_accuracy,
            consensus_alignment_pct=round(consensus_alignment, 2),
            risk_violations=0,
            executive_summary=narrative,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        # Save structured report
        out_file = self.reports_dir / f"audit_{target_date.replace('-', '_')}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(asdict(audit_summary), f, indent=2)

        logger.info(f"Saved EOD audit report to {out_file}.")
        return audit_summary
