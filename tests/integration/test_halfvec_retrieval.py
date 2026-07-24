import math
from datetime import datetime, timezone
from typing import Any, Generator
from unittest.mock import Mock

import pytest
from openai.types.create_embedding_response import CreateEmbeddingResponse, Usage
from openai.types.embedding import Embedding
from sqlalchemy import DateTime, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector  # type: ignore
from testcontainers.postgres import PostgresContainer  # type: ignore

from src.document_store.postgres.store import PostgresDocumentStore

DIMS = 3072
N_ROWS = 12
TABLE = "doc3072"


class _Base(DeclarativeBase):
    pass


class Doc3072(_Base):
    __tablename__ = TABLE

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(String)
    url: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    embedding: Mapped[Any] = mapped_column(Vector(DIMS))


def _vec(angle: float) -> list[float]:
    """A 3072-dim vector whose direction is set by `angle` (rest zero) so cosine
    distances between rows are distinct and easy to reason about."""
    v = [0.0] * DIMS
    v[0] = math.cos(angle)
    v[1] = math.sin(angle)
    v[2] = 0.1
    return v


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / (na * nb)


@pytest.fixture
def populated(postgres_container: PostgresContainer) -> Generator[dict, None, None]:
    url = postgres_container.get_connection_url()
    engine = create_engine(url)
    rows = [(f"doc{i}", _vec(i * 0.35)) for i in range(N_ROWS)]
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
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        for doc_id, vec in rows:
            session.add(
                Doc3072(
                    id=doc_id,
                    source="s1",
                    title=doc_id,
                    content=doc_id,
                    url=f"http://{doc_id}",
                    created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    embedding=vec,
                )
            )
        session.commit()

    yield {"url": url, "engine": engine, "rows": rows}

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {TABLE}"))
    engine.dispose()


def _store(engine, query_vec: list[float]) -> PostgresDocumentStore:
    client = Mock()
    client.embeddings.create.return_value = CreateEmbeddingResponse(
        data=[Embedding(embedding=query_vec, index=0, object="embedding")],
        model="test-embedding-3-large",
        usage=Usage(prompt_tokens=0, total_tokens=0),
        object="list",
    )
    store = PostgresDocumentStore(
        engine=engine,
        openai_client=client,
        embedding_model="test-embedding-3-large",
        embedding_dimensions=DIMS,
        candidate_multiplier=10,  # candidate_count = 30 >= N_ROWS => exact rerank
    )
    store.DocumentModel = Doc3072  # type: ignore[assignment]
    return store


def test_two_stage_returns_exact_float32_nearest(populated: dict) -> None:
    query_vec = _vec(0.55)  # between rows 1 (0.35) and 2 (0.70)
    store = _store(populated["engine"], query_vec)

    results = store.semantic_search("s1", "irrelevant-mocked", top_k=3)

    expected = [
        doc_id
        for doc_id, _ in sorted(
            populated["rows"], key=lambda r: _cosine_distance(query_vec, r[1])
        )
    ][:3]
    assert [d.id for d in results] == expected


def test_halfvec_expression_index_is_used_with_seqscan_off(populated: dict) -> None:
    # Contract check: with sequential scans disabled the planner must be able to use
    # the halfvec expression index for the candidate ordering (proves expression/query
    # compatibility). Realistic unforced planner behavior is measured in the benchmark.
    query_vec = _vec(0.55)
    q = "[" + ",".join(str(x) for x in query_vec) + "]"
    engine = populated["engine"]
    with engine.connect() as conn:
        conn.execute(text("SELECT '[1]'::vector"))  # load pgvector GUCs
        conn.execute(text("SET LOCAL enable_seqscan = off"))
        conn.execute(text("SET LOCAL hnsw.ef_search = 30"))
        plan = "\n".join(
            r[0]
            for r in conn.execute(
                text(
                    f"EXPLAIN SELECT id FROM {TABLE} "
                    f"ORDER BY embedding::halfvec({DIMS}) <=> CAST(:q AS halfvec({DIMS})) "
                    "LIMIT 30"
                ),
                {"q": q},
            ).fetchall()
        )
    assert f"{TABLE}_embedding_halfvec_idx" in plan, plan
