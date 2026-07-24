"""Shared utilities for the >2000-dim half-precision retrieval benchmark.

Used by the acceptance test (tests/integration/test_halfvec_recall.py) and the
standalone sweep script (scripts/benchmark_halfvec_retrieval.py). Not collected by
pytest itself (no ``test_`` prefix).

Recall is measured against *exhaustive float32 cosine* over the retained vectors —
the accuracy reference — so we can confirm the halfvec candidate + float32 rerank
path meets a recall threshold, and pick the smallest candidate_multiplier/ef_search
that passes. Embeddings are clustered (not uniform random) to mimic the near-neighbor
structure real embeddings have.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import Mock

import numpy as np
from openai.types.create_embedding_response import CreateEmbeddingResponse, Usage
from openai.types.embedding import Embedding
from pgvector.sqlalchemy import Vector  # type: ignore
from sqlalchemy import DateTime, Engine, String, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from src.document_store.postgres.store import PostgresDocumentStore

DIMS = 3072
TABLE = "bench_docs"


class _Base(DeclarativeBase):
    pass


class BenchDoc(_Base):
    __tablename__ = TABLE

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, index=True)
    title: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(String)
    url: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    embedding: Mapped[Any] = mapped_column(Vector(DIMS))


def _unit(v: Any) -> Any:
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(norm == 0, 1.0, norm)


def generate_corpus(
    source_sizes: dict[str, int], *, seed: int, n_clusters: int = 8, noise: float = 0.15
) -> dict[str, list[tuple[str, Any]]]:
    """Per source: cluster centers + noisy members, all unit vectors."""
    rng = np.random.default_rng(seed)
    corpus: dict[str, list[tuple[str, Any]]] = {}
    for source, size in source_sizes.items():
        centers = _unit(rng.standard_normal((n_clusters, DIMS)))
        docs: list[tuple[str, Any]] = []
        for i in range(size):
            center = centers[i % n_clusters]
            docs.append((f"{source}-{i}", _unit(center + noise * rng.standard_normal(DIMS))))
        corpus[source] = docs
    return corpus


def generate_queries(
    corpus: dict[str, list[tuple[str, Any]]], *, seed: int, per_source: int
) -> list[tuple[str, Any]]:
    """Queries near existing docs (doc vector + small noise) so they have genuine
    nearest neighbours. Returns (source, query_vec) in a fixed order."""
    rng = np.random.default_rng(seed)
    queries: list[tuple[str, Any]] = []
    for source, docs in corpus.items():
        idx = rng.integers(0, len(docs), size=per_source)
        for j in idx:
            base = docs[int(j)][1]
            queries.append((source, _unit(base + 0.1 * rng.standard_normal(DIMS))))
    return queries


def exhaustive_topk(docs: list[tuple[str, Any]], query: Any, k: int) -> list[str]:
    ids = [doc_id for doc_id, _ in docs]
    mat = np.stack([vec for _, vec in docs])  # (n, dims), unit vectors
    sims = mat @ np.asarray(query, dtype=np.float64)  # cosine similarity
    order = np.argsort(-sims)[:k]
    return [ids[int(j)] for j in order]


def recall_at_k(predicted: list[str], truth: list[str]) -> float:
    if not truth:
        return 1.0
    return len(set(predicted) & set(truth)) / len(truth)


def create_bench_table(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text(f"DROP TABLE IF EXISTS {TABLE}"))
    _Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                f"CREATE INDEX {TABLE}_embedding_halfvec_idx ON {TABLE} "
                f"USING hnsw ((embedding::halfvec({DIMS})) halfvec_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
        )


def insert_corpus(engine: Engine, corpus: dict[str, list[tuple[str, Any]]]) -> float:
    """Insert all docs; returns wall-clock insert seconds."""
    start = time.perf_counter()
    with Session(engine) as session:
        objs = [
            BenchDoc(
                id=doc_id,
                source=source,
                title=doc_id,
                content=doc_id,
                url=f"http://{doc_id}",
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                embedding=vec.tolist(),
            )
            for source, docs in corpus.items()
            for doc_id, vec in docs
        ]
        session.bulk_save_objects(objs)
        session.commit()
    return time.perf_counter() - start


def _make_store(
    engine: Engine, query_vecs: list[Any], multiplier: int, ef_search: int | None
) -> PostgresDocumentStore:
    client = Mock()
    client.embeddings.create.side_effect = [
        CreateEmbeddingResponse(
            data=[Embedding(embedding=list(v), index=0, object="embedding")],
            model="bench",
            usage=Usage(prompt_tokens=0, total_tokens=0),
            object="list",
        )
        for v in query_vecs
    ]
    store = PostgresDocumentStore(
        engine=engine,
        openai_client=client,
        embedding_model="bench",
        embedding_dimensions=DIMS,
        candidate_multiplier=multiplier,
        hnsw_ef_search=ef_search,
    )
    store.DocumentModel = BenchDoc  # type: ignore[assignment]
    return store


def measure_config(
    engine: Engine,
    corpus: dict[str, list[tuple[str, Any]]],
    queries: list[tuple[str, Any]],
    *,
    top_k: int,
    multiplier: int,
    ef_search: int | None,
) -> dict[str, Any]:
    """Run every query through the store and compute recall vs exhaustive float32
    and query latency. Returns aggregate + per-source recall and p50/p95 latency."""
    store = _make_store(engine, [q for _, q in queries], multiplier, ef_search)
    latencies: list[float] = []
    recalls: list[float] = []
    per_source: dict[str, list[float]] = {}
    for source, query_vec in queries:
        truth = exhaustive_topk(corpus[source], query_vec, top_k)
        start = time.perf_counter()
        results = store.semantic_search(source, "bench-query", top_k)
        latencies.append((time.perf_counter() - start) * 1000.0)
        r = recall_at_k([d.id for d in results], truth)
        recalls.append(r)
        per_source.setdefault(source, []).append(r)
    lat = sorted(latencies)
    return {
        "multiplier": multiplier,
        "ef_search": ef_search,
        "recall_mean": float(np.mean(recalls)),
        "recall_worst_source": min(float(np.mean(v)) for v in per_source.values()),
        "p50_ms": lat[len(lat) // 2],
        "p95_ms": lat[min(len(lat) - 1, int(len(lat) * 0.95))],
    }


def run_sweep(engine: Engine, *, seed: int = 1234) -> list[dict[str, Any]]:
    """Populate a larger uneven multi-source corpus and sweep candidate_multiplier /
    ef_search, printing a table and recommending the smallest passing config."""
    source_sizes = {"big": 2000, "medium": 800, "small": 150}
    top_k = 10
    corpus = generate_corpus(source_sizes, seed=seed)
    queries = generate_queries(corpus, seed=seed + 1, per_source=40)

    build_start = time.perf_counter()
    create_bench_table(engine)
    insert_s = insert_corpus(engine, corpus)
    setup_s = time.perf_counter() - build_start

    print(
        f"\nCorpus: {source_sizes} ({sum(source_sizes.values())} docs, {DIMS}d), "
        f"{len(queries)} queries, top_k={top_k}. setup={setup_s:.1f}s (insert={insert_s:.1f}s)"
    )
    print(f"{'mult':>5} {'ef':>6} {'recall':>8} {'worst':>8} {'p50 ms':>9} {'p95 ms':>9}")
    results: list[dict[str, Any]] = []
    for multiplier in (1, 2, 5, 10, 20):
        m = measure_config(
            engine, corpus, queries, top_k=top_k, multiplier=multiplier, ef_search=None
        )
        results.append(m)
        ef = m["ef_search"] if m["ef_search"] is not None else "auto"
        print(
            f"{multiplier:>5} {str(ef):>6} {m['recall_mean']:>8.4f} "
            f"{m['recall_worst_source']:>8.4f} {m['p50_ms']:>9.2f} {m['p95_ms']:>9.2f}"
        )

    threshold = 0.98
    passing = [r for r in results if r["recall_worst_source"] >= threshold]
    if passing:
        best = min(passing, key=lambda r: r["multiplier"])
        print(
            f"\nRecommended (smallest passing recall_worst >= {threshold}): "
            f"multiplier={best['multiplier']} (recall_mean={best['recall_mean']:.4f}, "
            f"p95={best['p95_ms']:.2f} ms)"
        )
    else:
        print(f"\nNo config met recall_worst >= {threshold}; inspect the table above.")
    return results
