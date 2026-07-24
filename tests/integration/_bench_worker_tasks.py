"""Test-only Celery task, loaded into the worker via --include for the prefork test.
Not collected by pytest (no test_ prefix) and not part of the production app.

It proves a forked prefork child can open its own database connection after the
parent's preflight, using the process-global engine — created lazily in the child
process, which is the real runtime path (the parent never populates it)."""

from sqlalchemy import text

from src.celery import celery_app
from src.common.postgres import get_postgres_engine
from src.config import get_settings


@celery_app.task(name="_bench_store_ping")
def bench_store_ping() -> int:
    engine = get_postgres_engine(get_settings())
    with engine.connect() as conn:
        return int(conn.execute(text("SELECT 1")).scalar() or 0)
