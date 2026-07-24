"""Standalone half-precision retrieval benchmark.

Populates an uneven multi-source 3072-dim corpus and sweeps candidate_multiplier
(with auto ef_search), reporting recall@10 vs exhaustive float32 and p50/p95 query
latency so you can pick the smallest config that meets the recall threshold.

Usage (spins up a throwaway pgvector container):
    poetry run python scripts/benchmark_halfvec_retrieval.py

Or against an existing database:
    POSTGRES_URL=postgresql+psycopg2://user:pass@host:5432/db \
        poetry run python scripts/benchmark_halfvec_retrieval.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine  # noqa: E402

from tests.integration.halfvec_bench_utils import run_sweep  # noqa: E402


def main() -> None:
    url = os.getenv("POSTGRES_URL")
    if url:
        engine = create_engine(url)
        try:
            run_sweep(engine)
        finally:
            engine.dispose()
        return

    from testcontainers.postgres import PostgresContainer  # type: ignore

    with PostgresContainer("pgvector/pgvector:pg17") as postgres:
        engine = create_engine(postgres.get_connection_url())
        try:
            run_sweep(engine)
        finally:
            engine.dispose()


if __name__ == "__main__":
    main()
