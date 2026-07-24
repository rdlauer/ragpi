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

import math
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
    # Normalize in float64, then quantize to float32 — pgvector stores single precision,
    # so the exhaustive "truth" must be computed over the same float32 values.
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / np.where(norm == 0, 1.0, norm)).astype(np.float32)


def generate_corpus(
    source_sizes: dict[str, int], *, seed: int, n_clusters: int = 8, noise_norm: float = 1.0
) -> dict[str, list[tuple[str, Any]]]:
    """Per source: cluster centers (unit) + members = center + noise, where the noise
    is scaled by 1/sqrt(DIMS) so its expected norm is ~noise_norm (comparable to the
    unit center), giving genuine clusters. With noise_norm=1.0 the intra-cluster cosine
    is ~0.5 — anisotropic/clustered structure like real embeddings, not near-orthogonal
    random vectors (which unscaled per-dimension noise produces in 3072 dims)."""
    rng = np.random.default_rng(seed)
    sigma = noise_norm / math.sqrt(DIMS)
    corpus: dict[str, list[tuple[str, Any]]] = {}
    for source, size in source_sizes.items():
        centers = _unit(rng.standard_normal((n_clusters, DIMS)))
        docs: list[tuple[str, Any]] = []
        for i in range(size):
            center = centers[i % n_clusters]
            docs.append((f"{source}-{i}", _unit(center + sigma * rng.standard_normal(DIMS))))
        corpus[source] = docs
    return corpus


def generate_queries(
    corpus: dict[str, list[tuple[str, Any]]],
    *,
    seed: int,
    per_source: int,
    query_noise_norm: float = 0.3,
) -> list[tuple[str, Any]]:
    """Queries near existing docs (doc + small noise, scaled by 1/sqrt(DIMS)) so they
    have genuine nearest neighbours. Returns (source, query_vec) in a fixed order."""
    rng = np.random.default_rng(seed)
    sigma = query_noise_norm / math.sqrt(DIMS)
    queries: list[tuple[str, Any]] = []
    for source, docs in corpus.items():
        idx = rng.integers(0, len(docs), size=per_source)
        for j in idx:
            base = docs[int(j)][1]
            queries.append((source, _unit(base + sigma * rng.standard_normal(DIMS))))
    return queries


def exhaustive_topk(docs: list[tuple[str, Any]], query: Any, k: int) -> list[str]:
    ids = [doc_id for doc_id, _ in docs]
    # Values are float32 (matching pgvector storage); accumulate in float64 for a stable
    # ranking, so the truth reflects the same stored values the ANN path reranks over.
    mat = np.stack([vec for _, vec in docs]).astype(np.float64)
    sims = mat @ np.asarray(query, dtype=np.float64)  # cosine similarity (unit vectors)
    order = np.argsort(-sims)[:k]
    return [ids[int(j)] for j in order]


def recall_at_k(predicted: list[str], truth: list[str]) -> float:
    if not truth:
        return 1.0
    return len(set(predicted) & set(truth)) / len(truth)


def create_bench_table(engine: Engine) -> None:
    """Create the table only. The vector index is built separately (see
    build_vector_index) so its build time can be measured over a populated table."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text(f"DROP TABLE IF EXISTS {TABLE}"))
    _Base.metadata.create_all(engine)


def build_vector_index(engine: Engine) -> float:
    """Create the halfvec HNSW expression index; returns build seconds (meaningful only
    over an already-populated table)."""
    start = time.perf_counter()
    with engine.begin() as conn:
        conn.execute(
            text(
                f"CREATE INDEX {TABLE}_embedding_halfvec_idx ON {TABLE} "
                f"USING hnsw ((embedding::halfvec({DIMS})) halfvec_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
        )
    return time.perf_counter() - start


def vector_index_size_bytes(engine: Engine) -> int:
    with engine.connect() as conn:
        return int(
            conn.execute(
                text(f"SELECT pg_relation_size('{TABLE}_embedding_halfvec_idx')")
            ).scalar()
            or 0
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


def run_sweep(
    engine: Engine,
    *,
    seed: int = 1234,
    recall_threshold: float = 0.98,
    p95_budget_ms: float | None = None,
) -> list[dict[str, Any]]:
    """Populate an uneven multi-source corpus, then sweep candidate_multiplier x
    ef_search — reporting recall@k vs exhaustive float32, p50/p95 latency, HNSW build
    time (over the populated table), insertion throughput, and index size — and
    recommend the smallest (multiplier, ef_search) meeting the recall (and optional
    p95) thresholds. This implements the plan's performance acceptance gate; run it on
    representative hardware (p95 is machine-dependent, so it is not asserted in CI)."""
    source_sizes = {"big": 2000, "medium": 800, "small": 150}
    top_k = 10
    corpus = generate_corpus(source_sizes, seed=seed)
    queries = generate_queries(corpus, seed=seed + 1, per_source=40)
    n_docs = sum(source_sizes.values())

    create_bench_table(engine)
    insert_s = insert_corpus(engine, corpus)
    build_s = build_vector_index(engine)  # over the populated table
    index_mb = vector_index_size_bytes(engine) / (1024 * 1024)

    print(
        f"\nCorpus: {source_sizes} ({n_docs} docs, {DIMS}d), {len(queries)} queries, "
        f"top_k={top_k}."
    )
    print(
        f"insert={insert_s:.1f}s ({n_docs / insert_s:,.0f} docs/s), "
        f"hnsw build={build_s:.1f}s, index={index_mb:.1f} MB"
    )
    print(f"{'mult':>5} {'ef':>6} {'recall':>8} {'worst':>8} {'p50 ms':>9} {'p95 ms':>9}")

    results: list[dict[str, Any]] = []
    for multiplier in (1, 2, 5, 10):
        for ef_search in (None, 100, 250, 500):
            m = measure_config(
                engine, corpus, queries, top_k=top_k, multiplier=multiplier, ef_search=ef_search
            )
            results.append(m)
            ef_label = "auto" if ef_search is None else str(ef_search)
            print(
                f"{multiplier:>5} {ef_label:>6} {m['recall_mean']:>8.4f} "
                f"{m['recall_worst_source']:>8.4f} {m['p50_ms']:>9.2f} {m['p95_ms']:>9.2f}"
            )

    def _passes(r: dict[str, Any]) -> bool:
        if r["recall_worst_source"] < recall_threshold:
            return False
        return p95_budget_ms is None or r["p95_ms"] <= p95_budget_ms

    passing = [r for r in results if _passes(r)]
    budget = "n/a" if p95_budget_ms is None else f"{p95_budget_ms:.0f} ms"
    if passing:
        best = min(passing, key=lambda r: (r["multiplier"], r["ef_search"] or 0))
        ef_label = "auto" if best["ef_search"] is None else best["ef_search"]
        print(
            f"\nRecommended smallest config (recall_worst >= {recall_threshold}, p95 <= {budget}): "
            f"multiplier={best['multiplier']}, ef_search={ef_label} "
            f"(recall_mean={best['recall_mean']:.4f}, p95={best['p95_ms']:.2f} ms)"
        )
    else:
        print(
            f"\nNo config met recall_worst >= {recall_threshold} and p95 <= {budget}; "
            "inspect the table above."
        )
    return results
