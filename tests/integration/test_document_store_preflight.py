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
    _resolve_table_oid,
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


# --------------------------------------------------------------------------- #
# Physical drift after manifest creation (manifest != proof objects still exist)
# --------------------------------------------------------------------------- #
def _exec(url: str, *statements: str) -> None:
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
    finally:
        engine.dispose()


def test_compatible_manifest_but_missing_table_fails(pg_url: str) -> None:
    settings = _settings(pg_url)
    run_preflight(settings)
    _exec(pg_url, f'DROP TABLE "{NS}" CASCADE')  # manifest row remains
    with pytest.raises(PreflightError, match="table is missing"):
        run_preflight(settings)


def test_compatible_manifest_but_missing_vector_index_fails(pg_url: str) -> None:
    settings = _settings(pg_url)
    run_preflight(settings)
    _exec(pg_url, f'DROP INDEX "{NS}_embedding_idx"')
    with pytest.raises(PreflightError, match="no valid vector index"):
        run_preflight(settings)


def test_compatible_manifest_but_missing_column_fails(pg_url: str) -> None:
    settings = _settings(pg_url)
    run_preflight(settings)
    _exec(pg_url, f'ALTER TABLE "{NS}" DROP COLUMN url')  # not part of fts_vector expr
    with pytest.raises(PreflightError, match="missing required column"):
        run_preflight(settings)


def test_legacy_adoption_requires_a_vector_index(pg_url: str) -> None:
    settings = _settings(pg_url)
    run_preflight(settings)
    # Simulate a legacy store with the vector index missing and no manifest.
    _exec(pg_url, f"DROP TABLE {MANIFEST_TABLE}", f'DROP INDEX "{NS}_embedding_idx"')
    with pytest.raises(PreflightError, match="no valid vector index"):
        run_preflight(settings)


def test_legacy_adoption_rejects_wrong_opclass(pg_url: str) -> None:
    settings = _settings(pg_url)
    run_preflight(settings)
    # Legacy store whose vector index uses the wrong (l2) opclass, no manifest.
    _exec(
        pg_url,
        f"DROP TABLE {MANIFEST_TABLE}",
        f'DROP INDEX "{NS}_embedding_idx"',
        f'CREATE INDEX "{NS}_embedding_idx" ON "{NS}" '
        "USING ivfflat (embedding vector_l2_ops) WITH (lists = 100)",
    )
    with pytest.raises(PreflightError, match="no valid vector index"):
        run_preflight(settings)


def test_adopted_legacy_index_name_recorded_and_survives_restart(pg_url: str) -> None:
    settings = _settings(pg_url)
    run_preflight(settings)
    # Simulate a pre-manifest deployment with the historical (non-namespaced) index name.
    _exec(
        pg_url,
        f'ALTER INDEX "{NS}_embedding_idx" RENAME TO embedding_idx',
        f"DROP TABLE {MANIFEST_TABLE}",
    )

    run_preflight(settings)  # adopts, records the legacy physical index name
    manifest = _read_manifest(pg_url)
    assert manifest is not None
    assert manifest.index_schema.name == "embedding_idx"

    run_preflight(settings)  # restart: compatible manifest + physical validation passes
    assert _read_manifest(pg_url).index_schema.name == "embedding_idx"  # type: ignore[union-attr]


def test_rebuilt_index_without_with_clause_is_accepted(pg_url: str) -> None:
    # The README's "rebuild the vector index" flow may omit WITH — pgvector's defaults
    # (lists=100) equal the manifest values, so validation must accept the rebuilt index
    # rather than loop startup failures on a semantically identical index.
    settings = _settings(pg_url)
    run_preflight(settings)
    _exec(
        pg_url,
        f'DROP INDEX "{NS}_embedding_idx"',
        f'CREATE INDEX "{NS}_embedding_idx" ON "{NS}" '
        "USING ivfflat (embedding vector_cosine_ops)",  # no WITH clause
    )
    run_preflight(settings)  # must pass


def test_concurrent_preflights_initialize_exactly_once(pg_url: str) -> None:
    # Plan-promised coverage: API + worker starting together on a fresh database must
    # serialize under the startup locks — both succeed, one schema/manifest results.
    from concurrent.futures import ThreadPoolExecutor

    settings = _settings(pg_url)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_preflight, settings) for _ in range(2)]
        for future in futures:
            future.result(timeout=60)  # raises if either preflight failed

    manifest = _read_manifest(pg_url)
    assert manifest == build_configured_manifest(settings)
    engine = create_engine(pg_url)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(f"SELECT count(*) FROM {MANIFEST_TABLE} WHERE namespace = :ns"),
                {"ns": NS},
            ).scalar()
    finally:
        engine.dispose()
    assert rows == 1


def test_resolve_table_oid_is_exact_and_case_sensitive(pg_url: str) -> None:
    # to_regclass() on a bound string case-folds ('MixedCase' -> mixedcase) and cannot
    # parse names like 'customer-docs'; our resolver matches pg_class.relname verbatim.
    engine = create_engine(pg_url)
    try:
        with engine.begin() as conn:
            conn.execute(text('CREATE TABLE "MixedCase" (id text)'))
            conn.execute(text('CREATE TABLE "customer-docs" (id text)'))
            assert conn.execute(text("SELECT to_regclass('MixedCase')")).scalar() is None
            assert _resolve_table_oid(conn, "MixedCase") is not None
            assert _resolve_table_oid(conn, "mixedcase") is None  # exact, not folded
            assert _resolve_table_oid(conn, "customer-docs") is not None  # hyphens allowed now
            conn.execute(text('DROP TABLE "MixedCase"'))
            conn.execute(text('DROP TABLE "customer-docs"'))
    finally:
        engine.dispose()


def test_compatible_manifest_but_wrong_index_build_params_fails(pg_url: str) -> None:
    settings = _settings(pg_url)
    run_preflight(settings)  # ivfflat lists=100 + manifest
    # Recreate the vector index with different build params (lists=10), same algo/opclass.
    _exec(
        pg_url,
        f'DROP INDEX "{NS}_embedding_idx"',
        f'CREATE INDEX "{NS}_embedding_idx" ON "{NS}" '
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10)",
    )
    with pytest.raises(PreflightError, match="no valid vector index"):
        run_preflight(settings)


def test_vector_index_on_wrong_column_is_rejected(pg_url: str) -> None:
    settings = _settings(pg_url)
    run_preflight(settings)
    # Otherwise-identical ivfflat index, but on a different column — must be rejected
    # (a textual "embedding" substring check would wrongly accept "other_embedding").
    _exec(
        pg_url,
        f'ALTER TABLE "{NS}" ADD COLUMN other_embedding vector(1536)',
        f'DROP INDEX "{NS}_embedding_idx"',
        f'CREATE INDEX "{NS}_embedding_idx" ON "{NS}" '
        "USING ivfflat (other_embedding vector_cosine_ops) WITH (lists = 100)",
    )
    with pytest.raises(PreflightError, match="no valid vector index"):
        run_preflight(settings)


def test_fts_index_on_wrong_column_is_rejected(pg_url: str) -> None:
    settings = _settings(pg_url)
    run_preflight(settings)
    _exec(
        pg_url,
        f'ALTER TABLE "{NS}" ADD COLUMN other_fts tsvector',
        f'DROP INDEX "{NS}_fts_vector_idx"',
        f'CREATE INDEX "{NS}_fts_vector_idx" ON "{NS}" USING gin (other_fts)',
    )
    with pytest.raises(PreflightError, match="full-text GIN index"):
        run_preflight(settings)


def test_partial_fts_index_is_rejected(pg_url: str) -> None:
    settings = _settings(pg_url)
    run_preflight(settings)
    # A partial predicate can leave some sources without usable FTS indexing.
    _exec(
        pg_url,
        f'DROP INDEX "{NS}_fts_vector_idx"',
        f'CREATE INDEX "{NS}_fts_vector_idx" ON "{NS}" USING gin (fts_vector) '
        "WHERE id IS NOT NULL",
    )
    with pytest.raises(PreflightError, match="full-text GIN index"):
        run_preflight(settings)
