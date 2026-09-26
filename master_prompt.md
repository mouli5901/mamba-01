# Master Build Prompt — AI Council Trading Engine (Google Antigravity)

Paste this into Antigravity's Manager view as the top-level task. It assumes Docker, Python 3.11, and Node are available in the workspace, and that the three skill files below have already been added under `.agents/skills/`.

## Ground rules — enforce in every phase, not just at the end
- Default execution mode is **Mode A (paper)**. Mode B (live Zerodha) requires an explicit `LIVE_TRADING=true` env var *and* the manual confirmation step in `risk_manager.py` — never auto-enable it from a script or a passing test.
- Build vertically, one working slice at a time. Don't scaffold every file up front and fill in logic later — each phase below should end with something you can actually run and test before the next phase starts.
- Treat `kiteconnect`, the LLM SDK calls, and any other blocking network I/O as synchronous. Run them in Celery workers or via `run_in_executor` — never directly inside an `async def` FastAPI route or on the asyncio event loop.
- No trade — paper or live — is placed without passing through `risk_manager.py`'s hard checks first. There should be no code path where `agent_council.py` calls `paper_engine.py` or `broker_execution.py` directly.

## Phase 0 — Scaffolding & environment
1. Create the directory tree from the blueprint.
2. `requirements.txt`: `fastapi`, `uvicorn[standard]`, `celery[redis]`, `redis`, `sqlalchemy`, `asyncpg`, `psycopg2-binary`, `pandas`, `pandas-ta`, `xgboost`, `langgraph`, `langchain-anthropic`, `langchain-openai`, `langchain-google-genai`, `kiteconnect`, `streamlit`, `pytest`, `pytest-asyncio`, `python-dotenv`.
3. `docker-compose.yml` with services: `redis`, `timescaledb` (image `timescale/timescaledb-ha:pg16`), `api` (FastAPI), `worker` (Celery), `dashboard` (Streamlit). Mount `./data` as a volume for `paper_portfolio.db` and `agent_memory.json`.
4. `.env.example` with placeholders for all three LLM API keys, `KITE_API_KEY`, `KITE_API_SECRET`, DB URL, Redis URL, and `LIVE_TRADING=false`.
5. **Test gate:** `docker compose up -d redis timescaledb`, confirm both are reachable, before writing any application code.

## Phase 1 — Data engine (`core/data_engine.py`)
1. Implement a WebSocket/REST tick client. Start with a free source (`tvDatafeed`, an NSE feed) so ticks are flowing before any broker account is wired in.
2. Buffer ticks into Redis as a capped list/stream — not unbounded.
3. Background task flushes buffered ticks into TimescaleDB in batches (100–500 rows); never one `INSERT` per tick.
4. **Kite Connect note:** Zerodha's Personal (free) tier is order/portfolio APIs only — no live quotes, no historical candles. If Kite is meant to be the tick source, that needs the paid Connect tier (₹500/month/API key). Otherwise keep a free source here and reserve Kite Connect for execution only, in Phase 7.
5. **Test gate:** run 10 minutes during market hours; tick count in TimescaleDB matches the Redis stream count (no silent drops).

## Phase 2 — Database schema (`core/db.py`)
1. Hypertable `ticks(time, symbol, ltp, volume)`, partitioned on `time`, `chunk_time_interval => interval '1 day'`.
2. Composite index `(symbol, time DESC)` — every downstream query in `ml_signal.py` uses this; without it you get a seq scan on every signal check.
3. Continuous aggregates `ohlcv_1min` / `_5min` / `_15min` rolling up `ticks`. `pandas_ta` and XGBoost read from these, never recompute OHLCV from raw ticks per signal.
4. **Test gate:** `EXPLAIN ANALYZE` the signal-window query and confirm it hits the index, not a seq scan.

## Phase 3 — Paper trading engine (`core/paper_engine.py`)
Build and stabilize fully before touching Phase 4 or 5 — everything downstream depends on its schema.
1. SQLite schema: `orders`, `positions`, `ledger`, with an explicit `client_order_id` (UUID) column for idempotency — the exact pattern `broker_execution.py` reuses later.
2. Slippage: 0.02% against the touched price, plus a configurable bid-ask spread estimate; fees/taxes from `config/risk_rules.json`.
3. **Test gate:** scripted buy/sell sequence against replayed historical bars; assert the ledger balance matches a hand-computed value.

## Phase 4 — Technical analyst + ML signal (`core/ml_signal.py`)
1. RSI, EMA 20/50/200, MACD, Bollinger Bands, ATR via `pandas_ta`, off the continuous aggregates.
2. Train the XGBoost classifier offline on historical data; commit the artifact. Don't retrain in the hot path.
3. Output a probability vector — one input to the council, not a standalone buy/sell decision.
4. **Test gate:** backtest the signal alone (no LLM council) against 3 months of historical data; log a baseline hit rate before adding the council on top.

## Phase 5 — LangGraph council (`core/agent_council.py`)
1. Fan the three LLM calls out with `asyncio.gather` — sequential calls make every decision cycle 3x slower for no benefit.
2. Hard per-call timeout (~8s) with an "abstain" fallback on timeout. A stuck LLM call should never stall a trading decision.
3. Consensus: 2-of-3 required to pass a proposal to `risk_manager.py`. Log all three raw responses regardless of outcome, for `audit_agent.py` later.
4. **Test gate:** replay Phase 4's signals through the council with a mocked slow LLM; confirm both the consensus logic and the timeout fallback work.

## Phase 6 — Risk manager (`core/risk_manager.py`)
1. Hard caps from `config/risk_rules.json`: max 1–2% capital per trade, max concurrent positions, max daily loss (kill switch — halts new orders for the day), max consecutive losses.
2. A dead-man's switch: if the data engine's heartbeat goes stale beyond N seconds, flatten open positions (paper) or alert-and-halt (live) rather than act on stale data.
3. This module is the *only* caller of `paper_engine.py` / `broker_execution.py` — enforce it in review, not just by convention.
4. **Test gate:** unit tests asserting a proposal that breaches any single hard cap is rejected regardless of what the council decided.

## Phase 7 — Zerodha live engine (`core/broker_execution.py`)
Build last, keep behind `LIVE_TRADING`.
1. Auth: Kite's access token expires daily — implement login + token-refresh, never hardcode a token.
2. Every order carries the `client_order_id` from Phase 3's schema. On retry (timeout, 5xx): check order status by that ID before resubmitting — never blind-retry a placement.
3. Reconciliation job (market open + every N minutes): pull Kite's actual positions/orders, diff against the internal ledger, alert on mismatch.
4. **Test gate:** smallest possible real position size, or paper-mode-only until Phases 3–6 have run against live market data (not just backtests) for at least a few weeks.

## Phase 8 — Dashboard, audit agent, containerization
1. Streamlit dual panel: equity curve + open positions, plus a read-only feed of the council's reasoning per decision.
2. `audit_agent.py`: end-of-day LLM review of trades vs. logged council reasoning — read-only, never wired to place trades.
3. Finalize `docker-compose.yml`; confirm a clean `docker compose up` from empty volumes reproduces schema + all services.
4. Run Mode A (paper) continuously against live market data for at least a few weeks before ever setting `LIVE_TRADING=true`.
