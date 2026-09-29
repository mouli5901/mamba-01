# AI Council Trading Engine (Google Antigravity)

An AI-driven trading engine built with FastAPI, Celery, LangGraph, Streamlit, TimescaleDB, and Redis.

## Overview

The AI Council Trading Engine is a robust, multi-phase trading system designed to aggregate market data, generate technical and machine learning signals, and use a consensus of Large Language Models (LLMs) to make trading decisions. The system is designed with strict risk management controls and supports both a paper-trading engine (Mode A) and live trading execution via Zerodha (Mode B).

## Architecture & Components

The engine is built vertically in phases, ensuring each component is testable and robust before moving on:

- **Phase 1: Data Engine** (`core/data_engine.py`)
  - A WebSocket/REST tick client (e.g., tvDatafeed) that buffers ticks into Redis as a capped stream.
  - A background task flushes these ticks into TimescaleDB in batches.
- **Phase 2: Database Schema** (`core/db.py`)
  - Uses TimescaleDB hypertables for tick data partitioned by time.
  - Employs continuous aggregates to roll up raw ticks into OHLCV (1m, 5m, 15m) bars, offloading computation.
- **Phase 3: Paper Trading Engine** (`core/paper_engine.py`)
  - A local SQLite-based execution engine simulating real orders with slippage, bid-ask spread estimates, and taxes/fees.
  - Maintains strict state across `orders`, `positions`, and `ledger`.
- **Phase 4: Technical Analyst & ML Signal** (`core/ml_signal.py`)
  - Computes technical indicators (RSI, EMAs, MACD, Bollinger Bands, ATR) via `pandas_ta` against continuous aggregates.
  - Uses offline-trained XGBoost classifiers to output a probability vector as a signal.
- **Phase 5: LangGraph Council** (`core/agent_council.py`)
  - Leverages Anthropic, OpenAI, and Google GenAI LLMs in parallel (`asyncio.gather`) to evaluate the ML signals.
  - Requires a 2-of-3 consensus to pass proposals to the risk manager, with strict timeouts and abstention fallbacks.
- **Phase 6: Risk Manager** (`core/risk_manager.py`)
  - The central gatekeeper enforcing hard caps (max capital per trade, max concurrent positions, daily loss kill switch).
  - Implements a dead-man's switch to handle stale data. This is the **only** module authorized to call execution engines.
- **Phase 7: Live Execution** (`core/broker_execution.py`)
  - Integrates with Zerodha Kite Connect for live trading.
  - Implements robust order status checking, reconciliation, and token management. Disabled by default (`LIVE_TRADING=false`).
- **Phase 8: Dashboard & Audit**
  - Streamlit dashboard for monitoring equity curves, open positions, and the AI council's reasoning.
  - `audit_agent.py` performs end-of-day LLM reviews of trades against council reasoning.

## Prerequisites

- Docker and Docker Compose
- Python 3.11 (for local development/testing)
- Node.js (if required by specific workspace tools)

## Setup & Installation

1. **Clone the repository**

2. **Environment Variables**
   Copy the example environment file and fill in your API keys and configuration details:
   ```bash
   cp .env.example .env
   ```
   *Note: Ensure `LIVE_TRADING=false` unless you explicitly want to trade with real funds and have tested thoroughly in paper mode.*

3. **Start the Infrastructure**
   Before running the application, start the underlying databases:
   ```bash
   docker compose up -d redis timescaledb
   ```
   *Verify both services are reachable before proceeding.*

4. **Start the Application Services**
   Start the API, Celery worker, and Dashboard:
   ```bash
   docker compose up -d api worker dashboard
   ```

## Development & Testing

- **Testing**: Run the test suite using `pytest`. Each phase has specific "Test gates" defined in the project blueprint that must pass.
  ```bash
  pytest
  ```
- **Local Dev**: The API is exposed on port `8000`, the Streamlit dashboard on port `8501`, Redis on `6379`, and TimescaleDB on `5432`. Data is persisted locally in the `./data` directory.

## Important Ground Rules

- **Paper Trading First**: Default execution mode is always Mode A (paper). Mode B (live) requires explicit manual confirmation.
- **Asynchronous Safety**: Blocking I/O (like KiteConnect or LLM SDK calls) must be run in Celery workers or via `run_in_executor`. Never block the FastAPI event loop.
- **Strict Risk Enforcement**: No trade is placed without passing through `risk_manager.py`. There are no bypasses.
