"""FastAPI Application for AI Council Trading Engine.

Provides REST and WebSocket endpoints for dashboard, health monitoring,
council feeds, and risk management status.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config.settings import LIVE_TRADING
from core.db import get_recent_ohlcv
from core.paper_engine import PaperTradingEngine
from core.risk_manager import RiskManager

app = FastAPI(
    title="AI Council Trading Engine API",
    description="Institutional AI Council Trading Engine with LangGraph consensus, TimescaleDB, and Risk Gates.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

paper_engine = PaperTradingEngine()
risk_manager = RiskManager(paper_engine=paper_engine, live_trading=LIVE_TRADING)
MEMORY_PATH = Path(__file__).resolve().parent.parent / "data" / "agent_memory.json"


@app.get("/")
def root():
    return {
        "status": "online",
        "system": "AI Council Trading Engine",
        "mode": "LIVE" if LIVE_TRADING else "PAPER",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "live_trading": LIVE_TRADING,
        "kill_switch_active": risk_manager.kill_switch_active,
        "heartbeat_stale": risk_manager.is_heartbeat_stale(),
    }


@app.get("/portfolio/balance")
def get_balance():
    return {
        "balance": paper_engine.get_current_balance(),
        "mode": "PAPER",
    }


@app.get("/portfolio/positions")
def get_positions():
    return {
        "positions": paper_engine.get_positions(),
    }


@app.get("/council/decisions")
def get_council_decisions(limit: int = 20):
    if not MEMORY_PATH.exists():
        return {"decisions": []}
    records = []
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return {"decisions": list(reversed(records))[:limit]}


@app.get("/risk/status")
def get_risk_status():
    return {
        "kill_switch_active": risk_manager.kill_switch_active,
        "kill_switch_reason": risk_manager.kill_switch_reason,
        "consecutive_losses": risk_manager.current_consecutive_losses,
        "max_consecutive_losses": risk_manager.max_consecutive_losses,
        "max_capital_per_trade_pct": risk_manager.max_capital_per_trade_pct,
        "max_concurrent_positions": risk_manager.max_concurrent_positions,
        "max_daily_loss_pct": risk_manager.max_daily_loss_pct,
    }
