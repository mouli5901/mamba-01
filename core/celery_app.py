"""Celery Application for Background Tasks in AI Council Trading Engine.

Handles asynchronous tick batch flushing, EOD audit runs, and periodic reconciliation.
"""

from __future__ import annotations

import logging
from celery import Celery

from config.settings import CELERY_BROKER_URL, CELERY_RESULT_BACKEND

logger = logging.getLogger(__name__)

celery_app = Celery(
    "kite_trader",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


@celery_app.task(name="flush_ticks_batch_task")
def flush_ticks_batch_task(ticks):
    """Background task to batch-insert ticks into TimescaleDB without blocking the tick stream."""
    from core.db import insert_ticks_batch
    count = insert_ticks_batch(ticks)
    logger.info(f"[Celery Worker] Flushed {count} ticks to TimescaleDB.")
    return count


@celery_app.task(name="reconciliation_task")
def reconciliation_task():
    """Periodic job running every N minutes to diff broker positions against internal ledger."""
    from core.paper_engine import PaperTradingEngine
    from core.broker_execution import BrokerExecutionEngine
    paper = PaperTradingEngine()
    broker = BrokerExecutionEngine()
    result = broker.reconcile_positions(paper.get_positions())
    logger.info(f"[Celery Worker] Reconciliation complete: in_sync={result.get('in_sync')}")
    return result


@celery_app.task(name="daily_audit_task")
def daily_audit_task():
    """Scheduled task running at market close to generate retrospective audit report."""
    from core.audit_agent import AuditAgent
    auditor = AuditAgent()
    summary = auditor.generate_daily_audit()
    logger.info(f"[Celery Worker] Daily audit report generated: win_rate={summary.win_rate_pct}%")
    return {"win_rate_pct": summary.win_rate_pct, "net_pnl": summary.net_pnl}
