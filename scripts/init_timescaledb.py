"""TimescaleDB Schema Initialization and Test Gate Runner.

Usage:
    python -m scripts.init_timescaledb [--test-gate]
"""

import sys
import argparse
import logging
from core.db import init_db, explain_signal_query, get_sync_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Initialize TimescaleDB hypertable and continuous aggregates.")
    parser.add_argument("--test-gate", action="store_true", help="Run EXPLAIN ANALYZE test gate on signal window query")
    parser.add_argument("--symbol", default="NIFTY50", help="Symbol to test with")
    parser.add_argument("--timeframe", default="1min", choices=["1min", "5min", "15min"], help="Timeframe to test")
    args = parser.parse_args()

    engine = get_sync_engine()
    logger.info("Initializing schema...")
    try:
        init_db(engine)
        logger.info("Schema initialization finished successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)

    if args.test_gate:
        logger.info(f"Running EXPLAIN ANALYZE test gate for {args.symbol} on {args.timeframe}...")
        try:
            res = explain_signal_query(args.symbol, args.timeframe, 100)
            logger.info(f"Scans detected: {res['scan_types']}")
            logger.info(f"Has Index Scan: {res['has_index_scan']}, Has Seq Scan: {res['has_seq_scan']}")
            if res["passed"]:
                logger.info("TEST GATE PASSED: Query hits index and avoids sequential scan.")
            else:
                logger.error("TEST GATE FAILED: Query did not hit index or performed sequential scan.")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Test gate execution failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
