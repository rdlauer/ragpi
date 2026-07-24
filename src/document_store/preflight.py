"""Document-store startup preflight.

Runs ONCE per process (API lifespan / Celery worker bootstep), never per request,
and in **validate-before-mutate** order under a startup lock:

  lock -> extension -> detect existing -> read manifest -> validate -> adopt/fail
       -> create only if fresh -> write manifest -> commit (release lock)

A mismatch is a *deployment* misconfiguration, so it raises ``PreflightError`` and
crashes startup — it is never surfaced as an HTTP 400. Preflight only validates,
adopts, or creates-fresh; it never drops or rebuilds an existing index, and it only
updates the pgvector extension when explicitly authorized.

This slice implements the Postgres backend (the default). Redis preflight is added
in a following slice; until then the Redis store keeps its existing self-init.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
import secrets
import time
from contextlib import contextmanager
from typing import Any, Iterator

from sqlalchemy import Engine, create_engine, text

from src.config import Settings
from src.document_store.manifest import (
    MANIFEST_VERSION,
    EmbeddingIdentity,
    IndexSchema,
    StorageSchema,
    StoreManifest,
    compare_manifests,
    normalize_endpoint_origin,
)
from src.llm_providers.constants import EmbeddingProvider

logger = logging.getLogger(__name__)

# The >2000-dim path needs halfvec (0.7.0) + iterative_scan (0.8.0), but requires
# 0.8.2: it fixed a buffer overflow in *parallel* HNSW index construction, and we
# build the HNSW index with parallel maintenance workers enabled.
MIN_PGVECTOR_FOR_LARGE = (0, 8, 2)
MANIFEST_TABLE = "ragpi_store_manifest"
PREFLIGHT_LOCK_TIMEOUT_S = 30.0
REDIS_LOCK_TTL_S = 60
# Acquisition timeout must exceed the TTL so a process can outwait a lock abandoned by
# a crashed initializer (rather than failing before the stale lock expires).
REDIS_LOCK_ACQUIRE_TIMEOUT_S = REDIS_LOCK_TTL_S + 15
# Only delete a lock we still own (guards against deleting a replacement's lock).
_REDIS_UNLOCK_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) "
    "else return 0 end"
)
# Fields the Redis index must physically have (mirrors redis/fields.py).
_REDIS_REQUIRED_FIELDS = {
    "id",
    "source",
    "content",
    "url",
    "created_at",
    "title",
    "embedding",
}

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_COLTYPE_RE = re.compile(r"^([A-Za-z_]+)\s*(?:\((\d+)\))?$")


class PreflightError(Exception):
    """Fatal startup misconfiguration; crashes the process (not an HTTP 400)."""


# --------------------------------------------------------------------------- #
# Deterministic identifiers (never Python's randomized hash())                #
# --------------------------------------------------------------------------- #
def advisory_lock_key(*parts: str) -> int:
    """Stable signed 64-bit advisory-lock key from a scope, identical across processes."""
    scope = "|".join(parts)
    digest = hashlib.sha256(scope.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, byteorder="big", signed=True)


def parse_version(value: str | None) -> tuple[int, ...]:
    """Parse an extension version semantically (numeric components), never lexicographically."""
    nums = re.findall(r"\d+", value or "")
    return tuple(int(n) for n in nums[:3]) if nums else (0,)


def _validate_ident(name: str) -> str:
    if not _IDENT_RE.match(name or ""):
        raise PreflightError(f"Unsafe SQL identifier: {name!r}")
    return name


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _index_name(namespace: str, suffix: str) -> str:
    """Namespace-aware index name that respects PostgreSQL's 63-byte identifier limit."""
    name = f"{namespace}_{suffix}"
    if len(name.encode("utf-8")) <= 63:
        return name
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    prefix = name[: 63 - len(digest) - 1]
    return f"{prefix}_{digest}"


def _parse_column_type(fmt: str) -> tuple[str, int | None]:
    m = _COLTYPE_RE.match(fmt.strip())
    if not m:
        return fmt.strip().lower(), None
    return m.group(1).lower(), int(m.group(2)) if m.group(2) else None


# --------------------------------------------------------------------------- #
# Configured manifest (what the current settings imply)                       #
# --------------------------------------------------------------------------- #
def _endpoint_space_id(settings: Settings) -> str | None:
    if settings.EMBEDDING_SPACE_ID:
        return settings.EMBEDDING_SPACE_ID
    if settings.EMBEDDING_PROVIDER == EmbeddingProvider.OLLAMA:
        return normalize_endpoint_origin(settings.OLLAMA_BASE_URL)
    if settings.EMBEDDING_PROVIDER == EmbeddingProvider.OPENAI_COMPATIBLE:
        return normalize_endpoint_origin(settings.EMBEDDING_OPENAI_COMPATIBLE_BASE_URL)
    return None  # standard OpenAI: provider identity is sufficient


def build_configured_manifest(
    settings: Settings, backend: str | None = None
) -> StoreManifest:
    backend = backend or settings.DOCUMENT_STORE_BACKEND
    dims = settings.EMBEDDING_DIMENSIONS
    identity = EmbeddingIdentity(
        provider=settings.EMBEDDING_PROVIDER.value,
        model=settings.EMBEDDING_MODEL,
        dimensions=dims,
        distance_metric="cosine",
        space_id=_endpoint_space_id(settings),
    )
    if backend == "redis":
        return StoreManifest(
            MANIFEST_VERSION,
            identity,
            StorageSchema(column_type="float32", dimensions=dims),
            IndexSchema(
                algorithm="hnsw",
                opclass="cosine",
                expression=None,
                build_params={"datatype": "float32"},
            ),
        )
    storage = StorageSchema(column_type="vector", dimensions=dims)
    if dims > 2000:
        index = IndexSchema(
            algorithm="hnsw",
            opclass="halfvec_cosine_ops",
            expression=f"embedding::halfvec({dims})",
            build_params={"m": 16, "ef_construction": 64},
        )
    else:
        index = IndexSchema(
            algorithm="ivfflat",
            opclass="vector_cosine_ops",
            expression=None,
            build_params={"lists": 100},
        )
    return StoreManifest(MANIFEST_VERSION, identity, storage, index)


def _is_legacy_default(m: StoreManifest) -> bool:
    """True if the embedding identity matches the historical pre-manifest defaults,
    so an unmarked existing store can be safely assumed to match it. Checks identity
    only (model/dims/provider) — the backend-specific storage/index representation is
    implied by the identity and validated separately."""
    ei = m.embedding_identity
    return (
        ei.provider == "openai"
        and ei.model == "text-embedding-3-small"
        and ei.dimensions == 1536
        and ei.space_id is None
    )


def _fingerprint(manifest: StoreManifest) -> str:
    return hashlib.sha256(
        json.dumps(manifest.to_dict(), sort_keys=True).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------- #
# Postgres preflight                                                          #
# --------------------------------------------------------------------------- #
def _acquire_pg_lock(conn, key: int) -> None:
    deadline = time.monotonic() + PREFLIGHT_LOCK_TIMEOUT_S
    while True:
        got = conn.execute(
            text("SELECT pg_try_advisory_xact_lock(:k)"), {"k": key}
        ).scalar()
        if got:
            return
        if time.monotonic() >= deadline:
            raise PreflightError(
                "Timed out acquiring the document-store startup lock; another process "
                "may be initializing the store. Retry, or check for a stuck initializer."
            )
        time.sleep(0.5)


def _ensure_manifest_table(conn) -> None:
    conn.execute(
        text(
            f"CREATE TABLE IF NOT EXISTS {MANIFEST_TABLE} ("
            "namespace text PRIMARY KEY, "
            "manifest jsonb NOT NULL, "
            "updated_at timestamptz NOT NULL DEFAULT now())"
        )
    )


def _read_manifest(conn, namespace: str) -> StoreManifest | None:
    row = conn.execute(
        text(f"SELECT manifest FROM {MANIFEST_TABLE} WHERE namespace = :ns"),
        {"ns": namespace},
    ).scalar()
    if row is None:
        return None
    data = row if isinstance(row, dict) else json.loads(row)
    return StoreManifest.from_dict(data)


def _write_manifest(conn, namespace: str, manifest: StoreManifest) -> None:
    conn.execute(
        text(
            f"INSERT INTO {MANIFEST_TABLE} (namespace, manifest, updated_at) "
            "VALUES (:ns, CAST(:m AS jsonb), now()) "
            "ON CONFLICT (namespace) DO UPDATE "
            "SET manifest = EXCLUDED.manifest, updated_at = now()"
        ),
        {"ns": namespace, "m": json.dumps(manifest.to_dict())},
    )


def _ensure_extension_version(conn, settings: Settings, dims: int) -> None:
    version = conn.execute(
        text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    ).scalar()
    if dims <= 2000:
        return
    if parse_version(version) >= MIN_PGVECTOR_FOR_LARGE:
        return
    if settings.PG_UPDATE_VECTOR_EXTENSION:
        logger.warning(
            "Updating the pgvector extension via ALTER EXTENSION vector UPDATE "
            "(affects the entire database)."
        )
        conn.execute(text("ALTER EXTENSION vector UPDATE"))
        version = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
        if parse_version(version) < MIN_PGVECTOR_FOR_LARGE:
            raise PreflightError(
                f"pgvector is still {version} after ALTER EXTENSION UPDATE, below the "
                "0.8.2 required for the >2000-dim index path. Upgrade the Postgres image/server "
                "(the pgvector/pgvector:pg17 image ships >= 0.8.2)."
            )
    else:
        raise PreflightError(
            f"EMBEDDING_DIMENSIONS={dims} (>2000) requires the pgvector server extension >= 0.8.2 "
            "(half-precision index + iterative scans, and the 0.8.2 fix for a buffer overflow in "
            f"parallel HNSW index builds), but the database reports {version!r}. Updating the "
            "Docker image/package does NOT upgrade an extension in an existing database. Set "
            "PG_UPDATE_VECTOR_EXTENSION=true to run 'ALTER EXTENSION vector UPDATE' at startup — "
            "note this upgrades the extension for the WHOLE database: back it up, check other "
            "pgvector-dependent apps, and revalidate them afterward."
        )


def _create_indexes(conn, namespace: str, dims: int) -> str:
    """Create the FTS GIN index and the dimension-appropriate vector index with
    namespace-aware names. Returns the physical vector-index name."""
    qns = _quote(namespace)
    fts_idx = _quote(_index_name(namespace, "fts_vector_idx"))
    conn.execute(
        text(f"CREATE INDEX IF NOT EXISTS {fts_idx} ON {qns} USING gin (fts_vector)")
    )
    if dims > 2000:
        emb_idx_name = _index_name(namespace, "embedding_halfvec_idx")
        conn.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS {_quote(emb_idx_name)} ON {qns} "
                f"USING hnsw ((embedding::halfvec({dims})) halfvec_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
        )
    else:
        emb_idx_name = _index_name(namespace, "embedding_idx")
        conn.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS {_quote(emb_idx_name)} ON {qns} "
                "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            )
        )
    return emb_idx_name


def _table_exists(conn, namespace: str) -> bool:
    return bool(
        conn.execute(
            text("SELECT to_regclass(:q) IS NOT NULL"), {"q": namespace}
        ).scalar()
    )


# Columns the document-store table must physically have (mirrors the ORM model).
_REQUIRED_COLUMNS = {
    "id",
    "source",
    "title",
    "content",
    "url",
    "created_at",
    "embedding",
    "fts_vector",
}


def _table_columns(conn, namespace: str) -> dict[str, str]:
    rows = conn.execute(
        text(
            "SELECT a.attname, format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
            "WHERE a.attrelid = to_regclass(:q) AND a.attnum > 0 AND NOT a.attisdropped"
        ),
        {"q": namespace},
    ).fetchall()
    return {row[0]: row[1] for row in rows}


def _table_indexes(conn, namespace: str) -> list[tuple[str, str]]:
    """(index name, indexdef) for every index on the table, resolved by the table's
    oid so it is schema-accurate."""
    rows = conn.execute(
        text(
            "SELECT c.relname, pg_get_indexdef(i.indexrelid) FROM pg_index i "
            "JOIN pg_class c ON c.oid = i.indexrelid WHERE i.indrelid = to_regclass(:q)"
        ),
        {"q": namespace},
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def _find_vector_index(indexes: list[tuple[str, str]]) -> tuple[str, str] | None:
    for name, indexdef in indexes:
        low = indexdef.lower()
        if "using ivfflat" in low or "using hnsw" in low:
            return name, indexdef
    return None


def _vector_index_matches(indexdef: str, index_schema: IndexSchema) -> bool:
    low = indexdef.lower()
    if f"using {index_schema.algorithm}" not in low:
        return False
    if index_schema.opclass not in low:
        return False
    if index_schema.expression:
        # e.g. "embedding::halfvec(3072)" -> require the halfvec(dims) cast.
        cast = index_schema.expression.split("::", 1)[-1].strip().lower()
        return cast in low
    # Plain-column index must not be a half-precision cast index.
    return "halfvec" not in low


def _has_fts_index(indexes: list[tuple[str, str]]) -> bool:
    return any(
        "using gin" in indexdef.lower() and "fts_vector" in indexdef.lower()
        for _, indexdef in indexes
    )


def _validate_physical_store(
    conn, settings: Settings, configured: StoreManifest, namespace: str
) -> str:
    """Introspect the actual table + indexes and confirm they match `configured`.
    The manifest records intended configuration, not proof the objects still exist, so
    this runs on both the manifest-present (restart) and legacy-adoption paths. Raises
    PreflightError with specific guidance on any drift; returns the vector index name.
    Structural failures here are NOT overridable by EMBEDDING_ADOPT_EXISTING."""
    if not _table_exists(conn, namespace):
        raise PreflightError(
            f"Document store '{namespace}' has a manifest but its table is missing. The store "
            "appears to have been dropped — delete the manifest row and restart to recreate it, "
            "then re-embed all sources."
        )

    columns = _table_columns(conn, namespace)
    missing = sorted(_REQUIRED_COLUMNS - set(columns))
    if missing:
        raise PreflightError(
            f"Document store '{namespace}' is missing required column(s) {missing}; it is not a "
            "valid Ragpi store. Recreate it and re-embed all sources."
        )

    col_type, col_dims = _parse_column_type(columns["embedding"])
    if (
        col_type != configured.storage_schema.column_type
        or col_dims != configured.storage_schema.dimensions
    ):
        raise PreflightError(
            f"Existing '{namespace}.embedding' column is '{columns['embedding']}', but the "
            f"configuration expects {configured.storage_schema.column_type}"
            f"({configured.storage_schema.dimensions}) "
            f"(EMBEDDING_MODEL={settings.EMBEDDING_MODEL}, "
            f"EMBEDDING_DIMENSIONS={settings.EMBEDDING_DIMENSIONS}). Changing embedding "
            "dimensions/type requires recreating the store and re-embedding all sources."
        )

    indexes = _table_indexes(conn, namespace)
    vector_index = _find_vector_index(indexes)
    if vector_index is None:
        raise PreflightError(
            f"Document store '{namespace}' has no vector index on 'embedding'. Rebuild the "
            f"{configured.index_schema.algorithm}/{configured.index_schema.opclass} index."
        )
    index_name, indexdef = vector_index
    if not _vector_index_matches(indexdef, configured.index_schema):
        raise PreflightError(
            f"Document store '{namespace}' vector index '{index_name}' does not match the "
            f"configured {configured.index_schema.algorithm}/{configured.index_schema.opclass} "
            f"definition (found: {indexdef}). Rebuild the vector index."
        )
    if not _has_fts_index(indexes):
        raise PreflightError(
            f"Document store '{namespace}' is missing the full-text GIN index on 'fts_vector'. "
            "Rebuild it."
        )
    return index_name


def _with_index_name(manifest: StoreManifest, name: str) -> StoreManifest:
    return dataclasses.replace(
        manifest, index_schema=dataclasses.replace(manifest.index_schema, name=name)
    )


def _adopt_legacy(conn, settings: Settings, configured: StoreManifest, namespace: str) -> None:
    # Full structural validation first — columns, embedding type/dims, vector index
    # definition, and FTS index. A structural mismatch is NOT overridable.
    index_name = _validate_physical_store(conn, settings, configured, namespace)
    # Only the embedding *identity* (provider/model) is unknowable from the database;
    # that is what EMBEDDING_ADOPT_EXISTING may override, not structural incompatibility.
    if _is_legacy_default(configured) or settings.EMBEDDING_ADOPT_EXISTING:
        _write_manifest(conn, namespace, _with_index_name(configured, index_name))
        logger.info(
            "Adopted existing document store '%s' (vector index '%s') and wrote its manifest.",
            namespace,
            index_name,
        )
    else:
        raise PreflightError(
            f"Existing document store '{namespace}' has no manifest, and the configured embedding "
            f"({configured.embedding_identity.model}, dims={configured.embedding_identity.dimensions}) "
            "is not the legacy default, so the embedding space of the stored vectors cannot be "
            "verified from dimensions alone. If you are certain the existing vectors were produced "
            "by this exact configuration, set EMBEDDING_ADOPT_EXISTING=true; otherwise recreate the "
            "store and re-embed all sources."
        )


def _create_fresh(conn, configured: StoreManifest, namespace: str, dims: int) -> None:
    # Import lazily so importing this module doesn't pull the ORM model at import time.
    from src.document_store.postgres.model import Base

    Base.metadata.create_all(conn)  # table only; indexes are created below (namespace-aware)
    index_name = _create_indexes(conn, namespace, dims)
    _write_manifest(conn, namespace, _with_index_name(configured, index_name))
    logger.info(
        "Created fresh document store '%s' (dims=%d, vector index '%s') and wrote its manifest.",
        namespace,
        dims,
        index_name,
    )


def _pg_preflight(engine: Engine, settings: Settings) -> None:
    configured = build_configured_manifest(settings)
    namespace = _validate_ident(settings.DOCUMENT_STORE_NAMESPACE)
    dims = settings.EMBEDDING_DIMENSIONS

    with engine.begin() as conn:
        _acquire_pg_lock(conn, advisory_lock_key("ragpi", "document_store", namespace))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        _ensure_extension_version(conn, settings, dims)
        _ensure_manifest_table(conn)

        existing = _read_manifest(conn, namespace)
        if existing is not None:
            comparison = compare_manifests(configured, existing)
            if not comparison.compatible:
                raise PreflightError(comparison.message)
            # The manifest records intended configuration, not proof the objects still
            # exist — confirm the physical table/columns/indexes match before trusting it.
            _validate_physical_store(conn, settings, configured, namespace)
            logger.info("Document store '%s' is compatible and physically valid.", namespace)
            return

        if _table_exists(conn, namespace):
            _adopt_legacy(conn, settings, configured, namespace)
        else:
            _create_fresh(conn, configured, namespace, dims)


# --------------------------------------------------------------------------- #
# Redis preflight                                                             #
# --------------------------------------------------------------------------- #
@contextmanager
def _redis_lock(
    client: Any,
    key: str,
    *,
    ttl: int = REDIS_LOCK_TTL_S,
    acquire_timeout: float = REDIS_LOCK_ACQUIRE_TIMEOUT_S,
) -> Iterator[str]:
    """Short-lived distributed lock: SET NX EX with a unique token, released via a
    compare-and-delete Lua script so we never delete a replacement process's lock. The
    acquisition timeout exceeds the TTL so a process can outwait an abandoned lock."""
    token = secrets.token_hex(16)
    deadline = time.monotonic() + acquire_timeout
    while not client.set(key, token, nx=True, ex=ttl):
        if time.monotonic() >= deadline:
            raise PreflightError(
                f"Timed out acquiring the Redis document-store startup lock ({key}); "
                "another process may be initializing the store."
            )
        time.sleep(0.5)
    try:
        yield token
    finally:
        try:
            client.eval(_REDIS_UNLOCK_LUA, 1, key, token)
        except Exception:  # pragma: no cover - best-effort release
            logger.warning("Failed to release Redis preflight lock %s", key)


def _assert_lock_owned(client: Any, key: str, token: str) -> None:
    """Fail if we no longer hold the startup lock (it expired and a replacement took
    over), so a stale initializer never keeps mutating shared state after losing it."""
    if client.get(key) != token:
        raise PreflightError(
            "Lost the Redis document-store startup lock mid-initialization (it expired and "
            "another process took over). Aborting to avoid concurrent initialization — restart "
            "this process to retry."
        )


def _canon(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _validate_redis_index(client: Any, configured: StoreManifest, name: str) -> None:
    """Confirm the actual server-side index matches `configured`: existence, HASH
    storage, key prefix, required fields, and the embedding vector schema. A manifest or
    crash marker is not proof the index still exists or matches, so this runs on the
    manifest, crash-resume, and legacy-adoption paths. Structural mismatches here are
    NOT overridable by EMBEDDING_ADOPT_EXISTING."""
    from redisvl.index import SearchIndex  # type: ignore

    from src.document_store.redis.store import build_index_schema

    dims = configured.storage_schema.dimensions
    if not SearchIndex(build_index_schema(name, dims), redis_client=client).exists():
        raise PreflightError(
            f"Redis index '{name}' is expected (manifest/marker present) but missing on the "
            "server. Recreate the index and re-embed all sources."
        )
    existing = SearchIndex.from_existing(name, redis_client=client)
    info = existing.schema.index

    storage = _canon(getattr(info, "storage_type", "hash"))
    if storage != "hash":
        raise PreflightError(
            f"Redis index '{name}' uses '{storage}' storage but Ragpi requires HASH. Recreate "
            "the index and re-embed."
        )

    expected_prefix = f"{name}:sources"
    prefixes = info.prefix if isinstance(info.prefix, list) else [info.prefix]
    if expected_prefix not in prefixes:
        raise PreflightError(
            f"Redis index '{name}' key prefix {prefixes} does not include the expected "
            f"'{expected_prefix}'. Recreate the index and re-embed."
        )

    fields = existing.schema.fields
    missing = sorted(_REDIS_REQUIRED_FIELDS - set(fields))
    if missing:
        raise PreflightError(
            f"Redis index '{name}' is missing required field(s) {missing}. Recreate the index "
            "and re-embed."
        )

    attrs = fields["embedding"].attrs
    existing_dims = int(getattr(attrs, "dims"))
    if existing_dims != dims:
        raise PreflightError(
            f"Redis index '{name}' embedding dims {existing_dims} != configured {dims}. Drop the "
            "index and its keys and re-embed."
        )
    schema = (
        _canon(getattr(attrs, "datatype", "float32")),
        _canon(getattr(attrs, "distance_metric", "cosine")),
        _canon(getattr(attrs, "algorithm", "hnsw")),
    )
    if schema != ("float32", "cosine", "hnsw"):
        raise PreflightError(
            f"Redis index '{name}' vector schema {schema} is incompatible with float32/cosine/"
            "hnsw. Drop the index and its keys and re-embed."
        )


def _adopt_legacy_redis(
    client: Any, settings: Settings, configured: StoreManifest, name: str, token: str
) -> None:
    # Full structural validation first (not overridable by EMBEDDING_ADOPT_EXISTING).
    _validate_redis_index(client, configured, name)
    # Only the embedding *identity* (provider/model) is unverifiable from the index.
    if _is_legacy_default(configured) or settings.EMBEDDING_ADOPT_EXISTING:
        _assert_lock_owned(client, f"{name}:__preflight_lock__", token)
        client.set(f"{name}:__manifest__", json.dumps(configured.to_dict()))
        logger.info("Adopted existing Redis document store '%s' and wrote its manifest.", name)
    else:
        raise PreflightError(
            f"Existing Redis index '{name}' has no manifest, and the configured embedding "
            f"({configured.embedding_identity.model}, "
            f"dims={configured.embedding_identity.dimensions}) is not the legacy default, so "
            "the embedding space cannot be verified. If you are certain the existing vectors "
            "match this configuration, set EMBEDDING_ADOPT_EXISTING=true; otherwise drop the "
            "index and its keys and re-embed."
        )


def _redis_preflight(client: Any, settings: Settings) -> None:
    from redisvl.index import SearchIndex  # type: ignore

    from src.document_store.redis.store import build_index_schema

    configured = build_configured_manifest(settings, backend="redis")
    name = _validate_ident(settings.DOCUMENT_STORE_NAMESPACE)
    dims = settings.EMBEDDING_DIMENSIONS
    manifest_key = f"{name}:__manifest__"
    marker_key = f"{name}:__init__"
    lock_key = f"{name}:__preflight_lock__"
    fingerprint = _fingerprint(configured)

    with _redis_lock(client, lock_key) as token:
        index = SearchIndex(build_index_schema(name, dims), redis_client=client)

        existing_raw = client.get(manifest_key)
        if existing_raw:
            comparison = compare_manifests(
                configured, StoreManifest.from_dict(json.loads(existing_raw))
            )
            if not comparison.compatible:
                raise PreflightError(comparison.message)
            # A manifest is not proof the index still exists or matches.
            _validate_redis_index(client, configured, name)
            _assert_lock_owned(client, lock_key, token)
            client.delete(marker_key)  # clear any stale init marker
            logger.info("Redis document store '%s' is compatible and physically valid.", name)
            return

        if index.exists():
            marker_raw = client.get(marker_key)
            if marker_raw:
                if json.loads(marker_raw).get("fingerprint") == fingerprint:
                    # Interrupted init for THIS config — validate the actual index before
                    # certifying it (a stale marker must not bless an incompatible index).
                    _validate_redis_index(client, configured, name)
                    _assert_lock_owned(client, lock_key, token)
                    client.set(manifest_key, json.dumps(configured.to_dict()))
                    client.delete(marker_key)
                    logger.info("Resumed interrupted Redis index init for '%s'.", name)
                    return
                raise PreflightError(
                    f"Redis index '{name}' carries an initialization marker for a different "
                    "configuration (an interrupted init). Drop the index and its keys, then re-embed."
                )
            _adopt_legacy_redis(client, settings, configured, name, token)
            return

        # Fresh: init-marker protocol makes a crash between create and manifest-write
        # recoverable (and an unmarked index is never silently adopted). Re-check lock
        # ownership before each mutation so a stale owner that lost the lock stops writing.
        _assert_lock_owned(client, lock_key, token)
        client.set(
            marker_key,
            json.dumps({"fingerprint": fingerprint, "token": token, "phase": "pending_manifest"}),
        )
        index.create()
        _assert_lock_owned(client, lock_key, token)
        client.set(manifest_key, json.dumps(configured.to_dict()))
        client.delete(marker_key)
        logger.info("Created fresh Redis document store '%s' and wrote its manifest.", name)


def run_preflight(settings: Settings) -> None:
    """Entry point for API lifespan and Celery worker startup.

    Uses dedicated, short-lived resources (a disposed engine / a closed Redis client)
    so a Celery parent never leaves pooled connections to be inherited across prefork.
    """
    if settings.DOCUMENT_STORE_BACKEND == "postgres":
        engine = create_engine(settings.POSTGRES_URL, pool_pre_ping=True)
        try:
            _pg_preflight(engine, settings)
        finally:
            engine.dispose()
    elif settings.DOCUMENT_STORE_BACKEND == "redis":
        from src.common.redis import create_redis_client

        client = create_redis_client(settings.REDIS_URL)
        try:
            _redis_preflight(client, settings)
        finally:
            client.close()
    else:
        raise PreflightError(
            f"Unsupported document store backend: {settings.DOCUMENT_STORE_BACKEND}"
        )
