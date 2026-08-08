"""Standalone half-precision retrieval benchmark.

Populates an uneven multi-source 3072-dim corpus and sweeps candidate_multiplier
(with auto ef_search), reporting recall@10 vs exhaustive float32 and p50/p95 query
latency so you can pick the smallest config that meets the recall threshold.

Usage (spins up a throwaway pgvector container):
    poetry run python scripts/benchmark_halfvec_retrieval.py

Against an existing database, and/or enforcing a p95 latency budget:
    POSTGRES_URL=postgresql+psycopg2://user:pass@host:5432/db BENCH_P95_MS=50 \
        poetry run python scripts/benchmark_halfvec_retrieval.py

The corpus here is a synthetic clustered proxy for embedding structure. For a
definitive production guarantee, point run_sweep() at a corpus embedded with the
real target model (e.g. text-embedding-3-large) on representative hardware.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine  # noqa: E402

from tests.integration.halfvec_bench_utils import run_sweep  # noqa: E402


def main() -> None:
    p95_env = os.getenv("BENCH_P95_MS")
    p95_budget_ms = float(p95_env) if p95_env else None
    url = os.getenv("POSTGRES_URL")
    if url:
        engine = create_engine(url)
        try:
            run_sweep(engine, p95_budget_ms=p95_budget_ms)
        finally:
            engine.dispose()
        return

    from testcontainers.postgres import PostgresContainer  # type: ignore

    with PostgresContainer("pgvector/pgvector:pg17") as postgres:
        engine = create_engine(postgres.get_connection_url())
        try:
            run_sweep(engine, p95_budget_ms=p95_budget_ms)
        finally:
            engine.dispose()


if __name__ == "__main__":
    main()
