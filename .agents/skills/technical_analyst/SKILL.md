---
name: technical-analyst
description: Computes technical indicators (RSI, EMA, MACD, Bollinger Bands, ATR via pandas_ta) and the XGBoost win-probability vector that feeds the LangGraph council. Use whenever a new candle closes and a signal needs to be generated, or when asked to backtest/tune the ML signal in isolation from the LLM council.
---

# Technical Analyst

## When this runs
Triggered on each new candle close (1min/5min/15min, per `config/settings.py`) once `core/data_engine.py` confirms the bar is complete. Its output feeds `core/agent_council.py` as one input alongside the raw candle context — it is not a standalone trading decision.

## Responsibilities
1. Read OHLCV from the TimescaleDB continuous aggregate for the relevant timeframe. Never recompute rollups from raw ticks per candle — that's exactly what the composite `(symbol, time DESC)` index and the aggregate exist to avoid.
2. Compute via `pandas_ta`: RSI(14), EMA(20/50/200), MACD, Bollinger Bands(20,2), ATR(14).
3. Feed the indicator vector into the pre-trained XGBoost classifier (trained offline — this module loads a committed model artifact, it does not retrain in the hot path) to produce a win-probability score.
4. Return a structured payload: `{symbol, timeframe, indicators: {...}, ml_probability: float, candle_ts}`. This is shared context for all three council members — keep its shape stable across changes so a schema tweak here doesn't silently break council parsing.

## Model lifecycle — keep separate from the hot path
- Retraining is a scheduled/offline job (e.g. weekly), never triggered by a live candle close.
- Version the model artifact (filename or metadata field) so `audit_agent.py` can later attribute a historical decision to the model version that made it.

## Backtesting this skill in isolation
Before wiring into the LLM council, backtest the raw `ml_probability` output alone against 3+ months of historical data and log a baseline hit rate / Sharpe. This baseline is what later tells you whether the LLM council adds real value over the statistical signal, or just adds latency.

## Common mistakes to avoid
- Computing indicators on a still-forming candle — confirm the bar has actually closed first.
- Letting a missing or late tick silently produce a stale indicator read. If `data_engine.py`'s heartbeat has lagged, flag the signal as low-confidence rather than emitting it as normal.
