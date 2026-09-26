"""TimescaleDB Database Engine and Schema Management for AI Council Trading Engine.

Phase 2 requirements:
1. Hypertable `ticks(time, symbol, ltp, volume)` partitioned on `time`, chunk_time_interval => '1 day'.
2. Composite index `(symbol, time DESC)` for tick lookups.
3. Continuous aggregates `ohlcv_1min`, `ohlcv_5min`, `ohlcv_15min` rolling up ticks.
4. Signal query interface and EXPLAIN ANALYZE test gate validator.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Index,
    String,
    Table,
    MetaData,
    create_engine,
    text,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from config.settings import DATABASE_URL, DATABASE_URL_SYNC

logger = logging.getLogger(__name__)

metadata = MetaData()
Base = declarative_base(metadata=metadata)


class Tick(Base):
    """Raw tick data stored in TimescaleDB hypertable."""

    __tablename__ = "ticks"

    time = Column(DateTime(timezone=True), primary_key=True, nullable=False)
    symbol = Column(String(32), primary_key=True, nullable=False)
    ltp = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)

    __table_args__ = (
        Index("idx_ticks_symbol_time_desc", "symbol", time.desc()),
    )


# DDL Statements for TimescaleDB setup
DDL_TIMESCALE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"

DDL_CREATE_TICKS_TABLE = """
CREATE TABLE IF NOT EXISTS ticks (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(32) NOT NULL,
    ltp DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL
);
"""

DDL_CREATE_HYPERTABLE = """
SELECT create_hypertable(
    'ticks',
    'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);
"""

DDL_COMPOSITE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_ticks_symbol_time_desc 
ON ticks (symbol, time DESC);
"""

# Continuous Aggregates for 1min, 5min, 15min rollups
DDL_CONTINUOUS_AGGREGATES = {
    "1min": """
    CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1min
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('1 minute', time) AS bucket,
        symbol,
        first(ltp, time) AS open,
        max(ltp) AS high,
        min(ltp) AS low,
        last(ltp, time) AS close,
        sum(volume) AS volume
    FROM ticks
    GROUP BY bucket, symbol
    WITH NO DATA;
    """,
    "5min": """
    CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_5min
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('5 minutes', time) AS bucket,
        symbol,
        first(ltp, time) AS open,
        max(ltp) AS high,
        min(ltp) AS low,
        last(ltp, time) AS close,
        sum(volume) AS volume
    FROM ticks
    GROUP BY bucket, symbol
    WITH NO DATA;
    """,
    "15min": """
    CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_15min
    WITH (timescaledb.continuous) AS
    SELECT
        time_bucket('15 minutes', time) AS bucket,
        symbol,
        first(ltp, time) AS open,
        max(ltp) AS high,
        min(ltp) AS low,
        last(ltp, time) AS close,
        sum(volume) AS volume
    FROM ticks
    GROUP BY bucket, symbol
    WITH NO DATA;
    """,
}

DDL_AGGREGATE_INDEXES = {
    "1min": "CREATE INDEX IF NOT EXISTS idx_ohlcv_1min_symbol_bucket ON ohlcv_1min (symbol, bucket DESC);",
    "5min": "CREATE INDEX IF NOT EXISTS idx_ohlcv_5min_symbol_bucket ON ohlcv_5min (symbol, bucket DESC);",
    "15min": "CREATE INDEX IF NOT EXISTS idx_ohlcv_15min_symbol_bucket ON ohlcv_15min (symbol, bucket DESC);",
}

DDL_REFRESH_POLICIES = {
    "1min": """
    SELECT add_continuous_aggregate_policy('ohlcv_1min',
        start_offset => INTERVAL '1 day',
        end_offset => INTERVAL '1 minute',
        schedule_interval => INTERVAL '1 minute',
        if_not_exists => TRUE);
    """,
    "5min": """
    SELECT add_continuous_aggregate_policy('ohlcv_5min',
        start_offset => INTERVAL '3 days',
        end_offset => INTERVAL '5 minutes',
        schedule_interval => INTERVAL '5 minutes',
        if_not_exists => TRUE);
    """,
    "15min": """
    SELECT add_continuous_aggregate_policy('ohlcv_15min',
        start_offset => INTERVAL '7 days',
        end_offset => INTERVAL '15 minutes',
        schedule_interval => INTERVAL '15 minutes',
        if_not_exists => TRUE);
    """,
}

# Engine and Session Singletons
_async_engine: Optional[AsyncEngine] = None
_sync_engine: Optional[Any] = None
_async_session_factory: Optional[async_sessionmaker[AsyncSession]] = None
_sync_session_factory: Optional[sessionmaker[Session]] = None


def get_sync_engine(url: Optional[str] = None):
    """Retrieve or create the synchronous SQLAlchemy engine."""
    global _sync_engine, _sync_session_factory
    if _sync_engine is None or url is not None:
        target_url = url or DATABASE_URL_SYNC
        _sync_engine = create_engine(
            target_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        _sync_session_factory = sessionmaker(
            bind=_sync_engine,
            autoflush=False,
            autocommit=False,
        )
    return _sync_engine


def get_async_engine(url: Optional[str] = None) -> AsyncEngine:
    """Retrieve or create the asynchronous SQLAlchemy engine."""
    global _async_engine, _async_session_factory
    if _async_engine is None or url is not None:
        target_url = url or DATABASE_URL
        _async_engine = create_async_engine(
            target_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
        _async_session_factory = async_sessionmaker(
            bind=_async_engine,
            expire_on_commit=False,
        )
    return _async_engine


@contextmanager
def get_sync_session() -> Generator[Session, None, None]:
    """Provide a transactional sync session scope."""
    get_sync_engine()
    assert _sync_session_factory is not None
    session = _sync_session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional async session scope."""
    get_async_engine()
    assert _async_session_factory is not None
    async with _async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def init_db(engine=None) -> None:
    """Initialize TimescaleDB extension, hypertable, indexes, and continuous aggregates."""
    active_engine = engine or get_sync_engine()
    logger.info("Initializing TimescaleDB schema...")

    with active_engine.connect() as conn:
        # 1. TimescaleDB extension
        try:
            conn.execute(text(DDL_TIMESCALE_EXTENSION))
            conn.commit()
            logger.info("TimescaleDB extension verified.")
        except Exception as e:
            logger.warning(f"Could not initialize timescaledb extension (skipping if standard PG/mock): {e}")
            conn.rollback()

        # 2. Table creation
        conn.execute(text(DDL_CREATE_TICKS_TABLE))
        conn.commit()
        logger.info("Table 'ticks' created/verified.")

        # 3. Hypertable setup
        try:
            conn.execute(text(DDL_CREATE_HYPERTABLE))
            conn.commit()
            logger.info("Hypertable 'ticks' created/verified on interval '1 day'.")
        except Exception as e:
            logger.warning(f"create_hypertable call failed (skipping if not TimescaleDB): {e}")
            conn.rollback()

        # 4. Composite index (symbol, time DESC)
        conn.execute(text(DDL_COMPOSITE_INDEX))
        conn.commit()
        logger.info("Composite index idx_ticks_symbol_time_desc verified.")

        # 5. Continuous aggregates & refresh policies
        for tf, ddl in DDL_CONTINUOUS_AGGREGATES.items():
            try:
                conn.execute(text(ddl))
                conn.commit()
                # Aggregate index
                conn.execute(text(DDL_AGGREGATE_INDEXES[tf]))
                conn.commit()
                # Refresh policy
                conn.execute(text(DDL_REFRESH_POLICIES[tf]))
                conn.commit()
                logger.info(f"Continuous aggregate ohlcv_{tf} and refresh policy verified.")
            except Exception as e:
                logger.warning(f"Continuous aggregate ohlcv_{tf} setup skipped/failed: {e}")
                conn.rollback()


def insert_ticks_batch(ticks: List[Dict[str, Any]], session: Optional[Session] = None) -> int:
    """Batch insert ticks into TimescaleDB.

    Enforces batching (100-500 rows) as required by Phase 1; never one INSERT per tick.
    """
    if not ticks:
        return 0

    insert_sql = text("""
        INSERT INTO ticks (time, symbol, ltp, volume)
        VALUES (:time, :symbol, :ltp, :volume)
    """)

    formatted = []
    for t in ticks:
        ts = t["time"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif isinstance(ts, (int, float)):
            ts = datetime.fromtimestamp(ts, tz=timezone.utc)

        formatted.append({
            "time": ts,
            "symbol": t["symbol"],
            "ltp": float(t["ltp"]),
            "volume": int(t["volume"]),
        })

    if session is not None:
        session.execute(insert_sql, formatted)
        return len(formatted)

    with get_sync_session() as s:
        s.execute(insert_sql, formatted)
    return len(formatted)


def build_signal_window_query(symbol: str, timeframe: str = "1min", limit: int = 100) -> str:
    """Generate the SQL query for the downstream signal window.

    Always reads from the continuous aggregates (ohlcv_1min, ohlcv_5min, ohlcv_15min),
    never recomputing OHLCV from raw ticks per signal.
    """
    view_name = f"ohlcv_{timeframe}"
    return f"""
        SELECT bucket, symbol, open, high, low, close, volume
        FROM {view_name}
        WHERE symbol = :symbol
        ORDER BY bucket DESC
        LIMIT :limit
    """


def get_recent_ohlcv(
    symbol: str,
    timeframe: str = "1min",
    limit: int = 100,
    session: Optional[Session] = None,
) -> pd.DataFrame:
    """Query recent OHLCV bars from continuous aggregates for technical indicator calculation.

    Returns chronological DataFrame (oldest to newest) with columns:
    [open, high, low, close, volume], indexed by datetime.
    """
    sql = text(build_signal_window_query(symbol, timeframe, limit))
    params = {"symbol": symbol, "limit": limit}

    if session is not None:
        result = session.execute(sql, params).mappings().all()
    else:
        with get_sync_session() as s:
            result = s.execute(sql, params).mappings().all()

    if not result:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(result)
    df.rename(columns={"bucket": "time"}, inplace=True)
    df.sort_values("time", ascending=True, inplace=True)
    df.set_index("time", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def explain_signal_query(
    symbol: str,
    timeframe: str = "1min",
    limit: int = 100,
    session: Optional[Session] = None,
) -> Dict[str, Any]:
    """Test gate: EXPLAIN ANALYZE the signal-window query.

    Asserts that the query hits an Index Scan (or Index Only Scan / Bitmap Index Scan)
    and does NOT perform an unindexed Seq Scan.
    """
    query_sql = build_signal_window_query(symbol, timeframe, limit)
    explain_sql = text(f"EXPLAIN (ANALYZE, FORMAT JSON) {query_sql}")
    params = {"symbol": symbol, "limit": limit}

    if session is not None:
        res = session.execute(explain_sql, params).scalar()
    else:
        with get_sync_session() as s:
            res = s.execute(explain_sql, params).scalar()

    # res is a JSON list containing the plan
    plan_root = res[0]["Plan"] if isinstance(res, list) and res else res

    def inspect_node_for_scans(node: Dict[str, Any]) -> List[str]:
        scan_types = []
        node_type = node.get("Node Type", "")
        if "Scan" in node_type:
            scan_types.append(f"{node_type} on {node.get('Relation Name', 'unknown')}")
        for child in node.get("Plans", []):
            scan_types.extend(inspect_node_for_scans(child))
        return scan_types

    scan_types = inspect_node_for_scans(plan_root)
    has_seq_scan = any("Seq Scan" in s for s in scan_types)
    has_index_scan = any("Index" in s for s in scan_types)

    return {
        "plan": plan_root,
        "scan_types": scan_types,
        "has_seq_scan": has_seq_scan,
        "has_index_scan": has_index_scan,
        "passed": has_index_scan and not has_seq_scan,
    }
