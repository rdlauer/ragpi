import subprocess
import sys
from pathlib import Path

from testcontainers.redis import RedisContainer  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_worker_exits_nonzero_when_preflight_fails(redis_container: RedisContainer) -> None:
    """A real Celery worker must fail to start (nonzero exit) when the document-store
    preflight fails — proving the bootstep exception propagates rather than being
    logged and swallowed. Broker is the live Redis container; the document store points
    at a refused Postgres port so preflight fails fast during worker startup."""
    redis_url = (
        f"redis://{redis_container.get_container_host_ip()}:"
        f"{redis_container.get_exposed_port(6379)}"
    )
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "PYTHONPATH": str(REPO_ROOT),
        "OPENAI_API_KEY": "test-key",
        "REDIS_URL": redis_url,
        "DOCUMENT_STORE_BACKEND": "postgres",
        "POSTGRES_URL": "postgresql+psycopg2://ragpi:ragpi@127.0.0.1:1/ragpi",  # refused
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "celery",
            "-A",
            "src.celery.celery_app",
            "worker",
            "--pool=solo",
            "-c",
            "1",
            "-l",
            "error",
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode != 0, (
        f"worker should have failed preflight but exited 0.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
