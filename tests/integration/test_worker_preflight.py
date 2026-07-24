import subprocess
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from testcontainers.postgres import PostgresContainer  # type: ignore
from testcontainers.redis import RedisContainer  # type: ignore

import src.common.postgres as pg_engine_module
from src.config import Settings
from src.document_store.preflight import MANIFEST_TABLE, run_preflight

REPO_ROOT = Path(__file__).resolve().parents[2]
NS = "document_store"


def _settings(pg_url: str, **overrides) -> Settings:
    base: dict = dict(
        OPENAI_API_KEY="test-key", POSTGRES_URL=pg_url, DOCUMENT_STORE_BACKEND="postgres"
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def _drop_store(pg_url: str) -> None:
    engine = create_engine(pg_url)
    try:
        with engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{NS}" CASCADE'))
            conn.execute(text(f"DROP TABLE IF EXISTS {MANIFEST_TABLE}"))
    finally:
        engine.dispose()


def _run_worker(redis_url: str, pg_url: str, **env_overrides) -> subprocess.CompletedProcess:
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "PYTHONPATH": str(REPO_ROOT),
        "OPENAI_API_KEY": "test-key",
        "REDIS_URL": redis_url,
        "DOCUMENT_STORE_BACKEND": "postgres",
        "POSTGRES_URL": pg_url,
    }
    env.update(env_overrides)
    return subprocess.run(
        [
            sys.executable, "-m", "celery", "-A", "src.celery.celery_app", "worker",
            "--pool=solo", "-c", "1", "-l", "error",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_worker_exits_nonzero_and_reports_the_preflight_failure(
    redis_container: RedisContainer, postgres_container: PostgresContainer
) -> None:
    """A worker must fail startup *because of a document-store preflight failure* (not
    merely any failure). Seed a store+manifest for the default model, then start a
    worker configured for a different embedding model so preflight raises a distinctive
    PreflightError — and assert that specific message reached the worker's output."""
    redis_url = (
        f"redis://{redis_container.get_container_host_ip()}:"
        f"{redis_container.get_exposed_port(6379)}"
    )
    pg_url = postgres_container.get_connection_url()
    _drop_store(pg_url)
    pg_engine_module.dispose_postgres_engine()
    run_preflight(_settings(pg_url))  # creates the store + manifest (text-embedding-3-small)

    try:
        proc = _run_worker(
            redis_url,
            pg_url,
            EMBEDDING_MODEL="text-embedding-3-large",  # different embedding space, same dims
            EMBEDDING_DIMENSIONS="1536",
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode != 0, output
        # Distinctive marker from our compare_manifests — proves it was the document-store
        # preflight (not an import/broker/CLI failure) that stopped startup.
        assert "Embedding identity changed" in output, output
    finally:
        _drop_store(pg_url)


def test_preflight_does_not_populate_the_global_engine(
    postgres_container: PostgresContainer,
) -> None:
    """Prefork-safety contract: preflight uses a short-lived engine it disposes, so it
    never leaves the process-global SQLAlchemy engine populated. This is what lets
    Celery prefork children establish their own connections instead of inheriting the
    parent's pool. (Also serves as a successful-preflight smoke test.)"""
    pg_url = postgres_container.get_connection_url()
    _drop_store(pg_url)
    pg_engine_module.dispose_postgres_engine()
    assert pg_engine_module._engine is None

    run_preflight(_settings(pg_url))  # succeeds against a reachable DB

    assert pg_engine_module._engine is None
    _drop_store(pg_url)
