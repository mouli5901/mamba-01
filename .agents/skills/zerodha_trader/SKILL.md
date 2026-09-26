---
name: zerodha-trader
description: Live order execution against Zerodha Kite Connect — auth/token refresh, idempotent order placement with retry logic, and position reconciliation. Use only when LIVE_TRADING=true and a trade has cleared risk_manager; also use when asked about Kite Connect auth, rate limits, or the personal-vs-paid API tiers.
---

# Zerodha Trader

## Before wiring this up: confirm the tier matches the need
Kite Connect's **Personal tier is free**, but it covers order placement, GTT, alerts, and portfolio/margin APIs only — it does **not** include live market quotes or historical candle data. If `core/data_engine.py` is meant to pull ticks from Kite's WebSocket, that requires the paid **Connect tier (₹500/month per API key)**. The cheaper, more common pattern: keep a free data source (an NSE feed, `tvDatafeed`, etc.) powering `data_engine.py` and `ml_signal.py`, and reserve Kite Connect in this module purely for authenticated order execution and portfolio state.

## When this runs
Called by `risk_manager.py`, and only when the `LIVE_TRADING` env var is true. If it isn't, redirect the trade proposal to the `paper-trader` skill instead — never place a real order because a flag was ambiguous or unset.

## Auth flow
1. Kite's access token expires daily (~6 AM IST rollover). Implement the login → request token → generate session flow each morning; never hardcode or cache a token across days.
2. Read the API key/secret from environment variables only — never in code, logs, or committed config.

## Idempotent order placement
1. Every order carries the same `client_order_id` schema the `paper-trader` skill uses.
2. On any network error, timeout, or 5xx from Kite's API: **check order status by `client_order_id` before resubmitting.** A blind retry on a request that timed out but actually succeeded server-side is how duplicate positions happen.
3. Exponential backoff (e.g. 3 attempts at 1s/2s/4s). If all attempts are exhausted, surface a hard failure to `risk_manager.py`'s kill-switch path — don't fail silently.

## Reconciliation
Run a job at market open and on an interval (e.g. every 15 minutes) that pulls Kite's actual `positions()` and `orders()` and diffs them against the internal ledger (the same schema `paper-trader` writes to). Alert immediately on any mismatch — silent drift between what Kite actually holds and what the bot thinks it holds is the single most dangerous failure mode in this system.

## Rate limits and safety
- Kite Connect enforces per-second and per-day request caps. Batch position/order status checks rather than polling per-symbol in a loop.
- This module has no callers except `risk_manager.py`. If asked to wire it up anywhere else, push back and route the request through risk checks first.

## Testing this skill
There is no sandbox environment for Kite Connect — every live-mode test is a real trade. Test with the smallest possible real position size, or restrict testing to read-only endpoints (`positions`, `margins`) outside trading hours before ever placing a live order.
