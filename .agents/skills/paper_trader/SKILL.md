---
name: paper-trader
description: Brokerless paper-trading execution for the AI council engine — order matching against live bars, slippage/fee modeling, and SQLite ledger persistence. Use whenever a trade proposal has cleared risk_manager and LIVE_TRADING is not set, or when asked to backtest/replay historical bars against the paper ledger.
---

# Paper Trader

## When this runs
Called by `risk_manager.py` after a trade proposal (`symbol`, `side`, `qty`, `signal_id`) has passed every hard risk check. Never called directly by `agent_council.py`.

## Responsibilities
1. Look up the current price from the TimescaleDB continuous aggregate (not raw ticks) for `symbol`.
2. Compute a fill price: apply 0.02% slippage against the touched price, plus the configured bid-ask spread estimate for the instrument.
3. Apply STT, exchange transaction charges, and turnover tax from `config/risk_rules.json` — not optional, and material to realized P&L on NIFTY options specifically.
4. Generate a `client_order_id` (UUID) before writing anything. This is the idempotency key `broker_execution.py` reuses in live mode — keep the schema identical between paper and live so switching modes doesn't require a schema migration.
5. Write to three SQLite tables in `data/paper_portfolio.db`:
   - `orders(client_order_id, symbol, side, qty, requested_price, fill_price, fees, status, created_at)`
   - `positions(symbol, qty, avg_price, unrealized_pnl)`
   - `ledger(client_order_id, cash_delta, balance_after, timestamp)`
6. Re-check the same position/capital caps `risk_manager.py` already applied — defense in depth, not redundant permission-asking.

## Backtest / replay mode
When asked to replay historical bars: iterate strictly in timestamp order, never look ahead (no reading bar N+1 while deciding on bar N), and write to a separate `data/backtest_portfolio.db` so replay runs never pollute the live paper ledger.

## Common mistakes to avoid
- Filling at the exact quoted price with zero slippage — this is the most common way a paper track record silently diverges from what live trading would actually produce.
- Recomputing the full ledger balance from order history on every trade. Maintain a running balance; only recompute from scratch during the periodic reconciliation check.
- Partial writes across the three tables (e.g. order row written, ledger row failed) — wrap each fill in one transaction so positions and cash never drift out of sync.

## Testing this skill
Run a scripted sequence of known buys/sells against a fixed historical bar sequence and assert the final ledger balance against a hand-computed value. Re-run this fixture after any change to the slippage/fee logic before trusting the result.
