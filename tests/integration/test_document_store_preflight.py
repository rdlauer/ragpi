import json
from typing import Generator

import pytest
from sqlalchemy import create_engine, text
from testcontainers.postgres import PostgresContainer  # type: ignore

from src.config import Settings
from src.document_store.manifest import StoreManifest
from src.document_store.preflight import (
    MANIFEST_TABLE,
    PreflightError,
    _create_indexes,
    build_configured_manifest,
    run_preflight,
)

NS = "document_store"  # default namespace matches the import-time model binding


def _settings(url: str, **overrides) -> Settings:
    base: dict = dict(
        OPENAI_API_KEY="test-key",
        POSTGRES_URL=url,
        DOCUMENT_STORE_BACKEND="postgres",
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def _drop_all(url: str) -> None:
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(f'DROP TABLE IF EXISTS "{NS}" CASCADE'))
            conn.execute(text(f"DROP TABLE IF EXISTS {MANIFEST_TABLE}"))
            conn.execute(text("DROP TABLE IF EXISTS big_vec"))
    finally:
        engine.dispose()


@pytest.fixture
def pg_url(postgres_container: PostgresContainer) -> Generator[str, None, None]:
    url = postgres_container.get_connection_url()
    _drop_all(url)
    yield url
    _drop_all(url)


def _read_manifest(url: str, namespace: str = NS) -> StoreManifest | None:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT manifest FROM {MANIFEST_TABLE} WHERE namespace = :ns"),
                {"ns": namespace},
            ).scalar()
    finally:
        engine.dispose()
    if row is None:
        return None
    return StoreManifest.from_dict(row if isinstance(row, dict) else json.loads(row))


def _index_names(url: str, table: str = NS) -> set[str]:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT indexname FROM pg_indexes WHERE tablename = :t"), {"t": table}
            ).fetchall()
    finally:
        engine.dispose()
    return {r[0] for r in rows}


def test_fresh_create_writes_table_indexes_and_manifest(pg_url: str) -> None:
    settings = _settings(pg_url)
    run_preflight(settings)

    engine = create_engine(pg_url)
    try:
        with engine.connect() as conn:
            assert conn.execute(text("SELECT to_regclass(:q)"), {"q": NS}).scalar() is not None
    finally:
        engine.dispose()

    names = _index_names(pg_url)
    assert f"{NS}_embedding_idx" in names  # namespace-aware ivfflat index
    assert f"{NS}_fts_vector_idx" in names

    assert _read_manifest(pg_url) == build_configured_manifest(settings)


def test_preflight_is_idempotent(pg_url: str) -> None:
    settings = _settings(pg_url)
    run_preflight(settings)
    run_preflight(settings)  # manifest present + compatible → no error, single row (namespace PK)
    assert _read_manifest(pg_url) == build_configured_manifest(settings)


def test_adopts_legacy_default_store_without_manifest(pg_url: str) -> None:
    settings = _settings(pg_url)
    run_preflight(settings)  # creates table + index + manifest

    # Simulate a pre-manifest deployment: drop the manifest table, keep the store.
    engine = create_engine(pg_url)
    try:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE {MANIFEST_TABLE}"))
    finally:
        engine.dispose()

    run_preflight(settings)  # legacy default → adopt + rewrite manifest, no error
    assert _read_manifest(pg_url) == build_configured_manifest(settings)


def test_manifest_model_mismatch_fails(pg_url: str) -> None:
    settings = _settings(pg_url)
    run_preflight(settings)

    # Overwrite the manifest with a different embedding model at the SAME dimensions —
    # the exact footgun the manifest exists to catch.
    other = build_configured_manifest(
        _settings(pg_url, EMBEDDING_MODEL="text-embedding-3-large", EMBEDDING_DIMENSIONS=1536)
    )
    engine = create_engine(pg_url)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    f"INSERT INTO {MANIFEST_TABLE} (namespace, manifest, updated_at) "
                    "VALUES (:ns, CAST(:m AS jsonb), now()) ON CONFLICT (namespace) "
                    "DO UPDATE SET manifest = EXCLUDED.manifest"
                ),
                {"ns": NS, "m": json.dumps(other.to_dict())},
            )
    finally:
        engine.dispose()

    with pytest.raises(PreflightError, match="Embedding identity changed"):
        run_preflight(settings)


def test_nondefault_legacy_adoption_requires_explicit_flag(pg_url: str) -> None:
    run_preflight(_settings(pg_url))  # table is vector(1536), then drop manifest
    engine = create_engine(pg_url)
    try:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE {MANIFEST_TABLE}"))
    finally:
        engine.dispose()

    # Non-default config (large at 1536) over an unmarked store: shape matches but the
    # model can't be verified → must require the explicit adoption flag.
    nondefault = _settings(
        pg_url, EMBEDDING_MODEL="text-embedding-3-large", EMBEDDING_DIMENSIONS=1536
    )
    with pytest.raises(PreflightError, match="EMBEDDING_ADOPT_EXISTING"):
        run_preflight(nondefault)

    adopt = _settings(
        pg_url,
        EMBEDDING_MODEL="text-embedding-3-large",
        EMBEDDING_DIMENSIONS=1536,
        EMBEDDING_ADOPT_EXISTING=True,
    )
    run_preflight(adopt)  # now allowed
    assert _read_manifest(pg_url) == build_configured_manifest(adopt)


def test_halfvec_hnsw_expression_index_builds_on_3072(pg_url: str) -> None:
    # Validates the pgvector 0.8 half-precision HNSW expression index (the >2000-dim
    # path) against the real server, using the exact DDL preflight emits.
    engine = create_engine(pg_url)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE big_vec (id text PRIMARY KEY, "
                    "embedding vector(3072), fts_vector tsvector)"
                )
            )
            _create_indexes(conn, "big_vec", 3072)
        with engine.connect() as conn:
            defs = " ".join(
                r[0]
                for r in conn.execute(
                    text("SELECT indexdef FROM pg_indexes WHERE tablename = 'big_vec'")
                ).fetchall()
            )
    finally:
        engine.dispose()

    assert "hnsw" in defs
    assert "halfvec" in defs
    assert "big_vec_embedding_halfvec_idx" in defs
