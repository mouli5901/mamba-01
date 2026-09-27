"""Streamlit Dual-Panel Trading Dashboard for AI Council Trading Engine.

Phase 8 requirements:
1. Dual panel layout:
   - Panel A: Equity curve + open positions + performance metrics.
   - Panel B: Read-only live feed of the AI Council's deliberations and reasoning.
2. Direct connection to data/paper_portfolio.db and data/agent_memory.json.
3. Clean, institutional dark-theme visual styling.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI Council Trading Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "paper_portfolio.db"
MEMORY_PATH = Path(__file__).resolve().parent.parent / "data" / "agent_memory.json"
RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "risk_rules.json"


def load_risk_rules() -> Dict[str, Any]:
    if RULES_PATH.exists():
        with open(RULES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"initial_capital": 100000.0}


def load_portfolio_data() -> Dict[str, Any]:
    if not DB_PATH.exists():
        rules = load_risk_rules()
        return {
            "balance": float(rules.get("initial_capital", 100000.0)),
            "positions": pd.DataFrame(columns=["symbol", "qty", "avg_price", "unrealized_pnl"]),
            "orders": pd.DataFrame(columns=["client_order_id", "symbol", "side", "qty", "fill_price", "fees", "created_at"]),
            "ledger": pd.DataFrame(columns=["timestamp", "cash_delta", "balance_after"]),
        }

    conn = sqlite3.connect(DB_PATH)
    try:
        positions_df = pd.read_sql_query("SELECT * FROM positions WHERE qty > 0", conn)
        orders_df = pd.read_sql_query("SELECT * FROM orders ORDER BY created_at DESC LIMIT 50", conn)
        ledger_df = pd.read_sql_query("SELECT * FROM ledger ORDER BY id ASC", conn)

        rules = load_risk_rules()
        initial_capital = float(rules.get("initial_capital", 100000.0))
        current_balance = float(ledger_df["balance_after"].iloc[-1]) if not ledger_df.empty else initial_capital

        return {
            "balance": current_balance,
            "positions": positions_df,
            "orders": orders_df,
            "ledger": ledger_df,
        }
    finally:
        conn.close()


def load_council_memory() -> List[Dict[str, Any]]:
    if not MEMORY_PATH.exists():
        return []
    records = []
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return list(reversed(records))  # latest first


# ── Dashboard Layout ──────────────────────────────────────────

st.title("🏛️ AI Council Trading Engine")
st.caption("Mode A (Paper Execution) | Multi-Agent Consensus | TimescaleDB & Zerodha Ready")

data = load_portfolio_data()
council_logs = load_council_memory()
rules = load_risk_rules()
initial_capital = float(rules.get("initial_capital", 100000.0))
current_balance = data["balance"]
total_pnl = current_balance - initial_capital
pnl_pct = (total_pnl / initial_capital) * 100

# Top metrics bar
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Cash Balance", f"₹{current_balance:,.2f}")
m2.metric("Total P&L", f"₹{total_pnl:+,.2f}", f"{pnl_pct:+.2f}%")
m3.metric("Open Positions", len(data["positions"]))
m4.metric("Council Decisions", len(council_logs))
m5.metric("Execution Mode", "Mode A (Paper)", delta_color="normal")

st.divider()

col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("📊 Portfolio & Equity Curve")

    ledger_df = data["ledger"]
    if not ledger_df.empty and len(ledger_df) > 1:
        chart_data = ledger_df[["timestamp", "balance_after"]].copy()
        chart_data["timestamp"] = pd.to_datetime(chart_data["timestamp"])
        chart_data.set_index("timestamp", inplace=True)
        st.line_chart(chart_data, color="#29B5E8")
    else:
        st.info("Equity curve will plot here once trades are executed in the ledger.")

    st.subheader("📦 Open Positions")
    positions_df = data["positions"]
    if not positions_df.empty:
        st.dataframe(
            positions_df.style.format(
                {"avg_price": "₹{:.2f}", "unrealized_pnl": "₹{:+.2f}", "qty": "{:d}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No open positions. Capital is 100% in cash.")

    st.subheader("📜 Recent Filled Orders")
    orders_df = data["orders"]
    if not orders_df.empty:
        st.dataframe(
            orders_df[["symbol", "side", "qty", "fill_price", "fees", "created_at"]].style.format(
                {"fill_price": "₹{:.2f}", "fees": "₹{:.2f}", "qty": "{:d}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No orders placed yet.")

with col_right:
    st.subheader("🧠 AI Council Deliberations Feed")
    st.caption("Live 3-member consensus (Claude Technical, GPT Quant, Gemini Macro)")

    if council_logs:
        for idx, entry in enumerate(council_logs[:15]):
            action = entry.get("consensus_action") or "NO CONSENSUS"
            status_color = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "⚪"
            votes = entry.get("votes", {})
            symbol = entry.get("symbol", "N/A")
            price = entry.get("close_price", 0.0)
            ml_p = entry.get("ml_probability", 0.5)

            with st.expander(f"{status_color} {symbol} — {action} (Price: ₹{price:,.2f})", expanded=(idx == 0)):
                st.write(f"**Timestamp:** `{entry.get('candle_ts')}`")
                st.write(f"**ML Win Probability:** `{ml_p:.2%}`")

                v1, v2, v3 = st.columns(3)
                v1.metric("Claude (Technical)", votes.get("claude_technical", "N/A"))
                v2.metric("GPT (Quant)", votes.get("gpt_quant", "N/A"))
                v3.metric("Gemini (Macro)", votes.get("gemini_macro", "N/A"))

                raw = entry.get("raw_responses", {})
                for agent_name, resp in raw.items():
                    if resp and "reasoning" in resp:
                        st.caption(f"**{agent_name.replace('_', ' ').title()}:** {resp['reasoning']}")

                proposal = entry.get("proposal")
                if proposal:
                    st.success(
                        f"**Trade Proposal Emitted:** {proposal['side']} {proposal['symbol']} @ ₹{proposal['requested_price']} "
                        f"(Confidence: {proposal['confidence']:.2%})"
                    )
    else:
        st.info("No council decisions recorded yet in `data/agent_memory.json`.")

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("🛡️ Risk Manager Status")
st.sidebar.write("• Capital / Trade Cap: **2.0%**")
st.sidebar.write("• Max Concurrent Positions: **5**")
st.sidebar.write("• Daily Loss Kill Switch: **5.0%**")
st.sidebar.write("• Dead-Man's Switch: **30 seconds**")
