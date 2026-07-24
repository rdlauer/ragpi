import pytest
from sqlalchemy import create_engine, text
from testcontainers.postgres import PostgresContainer  # type: ignore

from tests.integration.halfvec_bench_utils import (
    TABLE,
    create_bench_table,
    generate_corpus,
    generate_queries,
    insert_corpus,
    measure_config,
)

TOP_K = 10
# Acceptance threshold for the >2000-dim half-precision path vs exhaustive float32.
RECALL_THRESHOLD = 0.98


def test_default_config_meets_recall_threshold(postgres_container: PostgresContainer) -> None:
    """Sources are larger than the candidate count (top_k * default multiplier = 100),
    so the halfvec ANN genuinely filters and this measures real recall (not a trivial
    fetch-everything). Recall@10 vs exhaustive float32 must clear the threshold on both
    the aggregate and the worst source, at the shipped default multiplier of 10."""
    url = create_engine(postgres_container.get_connection_url())
    try:
        corpus = generate_corpus({"big": 500, "medium": 200}, seed=7)
        create_bench_table(url)
        insert_corpus(url, corpus)
        queries = generate_queries(corpus, seed=8, per_source=20)

        result = measure_config(
            url, corpus, queries, top_k=TOP_K, multiplier=10, ef_search=None
        )
        assert result["recall_mean"] >= RECALL_THRESHOLD, result
        assert result["recall_worst_source"] >= RECALL_THRESHOLD, result
    finally:
        with url.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {TABLE}"))
        url.dispose()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
