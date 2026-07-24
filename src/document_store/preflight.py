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

# iterative_scan (needed for the filtered halfvec query path) was added in 0.8.0;
# halfvec itself is 0.7.0, but the >2000-dim query path depends on 0.8.0.
MIN_PGVECTOR_FOR_LARGE = (0, 8, 0)
MANIFEST_TABLE = "ragpi_store_manifest"
PREFLIGHT_LOCK_TIMEOUT_S = 30.0
REDIS_LOCK_TTL_S = 60
# Only delete a lock we still own (guards against deleting a replacement's lock).
_REDIS_UNLOCK_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) "
    "else return 0 end"
)

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
                "0.8.0 required for the >2000-dim index path. Upgrade the Postgres image/server."
            )
    else:
        raise PreflightError(
            f"EMBEDDING_DIMENSIONS={dims} (>2000) requires the pgvector server extension >= 0.8.0 "
            f"(half-precision index + iterative scans), but the database reports {version!r}. "
            "Updating the Docker image/package does NOT upgrade an extension in an existing "
            "database. Set PG_UPDATE_VECTOR_EXTENSION=true to run 'ALTER EXTENSION vector UPDATE' "
            "at startup — note this upgrades the extension for the WHOLE database: back it up, "
            "check other pgvector-dependent apps, and revalidate them afterward."
        )


def _create_indexes(conn, namespace: str, dims: int) -> None:
    qns = _quote(namespace)
    fts_idx = _quote(_index_name(namespace, "fts_vector_idx"))
    conn.execute(
        text(f"CREATE INDEX IF NOT EXISTS {fts_idx} ON {qns} USING gin (fts_vector)")
    )
    if dims > 2000:
        emb_idx = _quote(_index_name(namespace, "embedding_halfvec_idx"))
        conn.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS {emb_idx} ON {qns} "
                f"USING hnsw ((embedding::halfvec({dims})) halfvec_cosine_ops) "
                "WITH (m = 16, ef_construction = 64)"
            )
        )
    else:
        emb_idx = _quote(_index_name(namespace, "embedding_idx"))
        conn.execute(
            text(
                f"CREATE INDEX IF NOT EXISTS {emb_idx} ON {qns} "
                "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            )
        )


def _table_exists(conn, namespace: str) -> bool:
    return bool(
        conn.execute(
            text("SELECT to_regclass(:q) IS NOT NULL"), {"q": namespace}
        ).scalar()
    )


def _adopt_legacy(conn, settings: Settings, configured: StoreManifest, namespace: str) -> None:
    fmt = conn.execute(
        text(
            "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
            "WHERE a.attrelid = to_regclass(:q) AND a.attname = 'embedding' "
            "AND a.attnum > 0 AND NOT a.attisdropped"
        ),
        {"q": namespace},
    ).scalar()
    if fmt is None:
        raise PreflightError(
            f"Existing table '{namespace}' has no 'embedding' column; cannot adopt it as a "
            "Ragpi document store."
        )
    col_type, col_dims = _parse_column_type(fmt)
    if (
        col_type != configured.storage_schema.column_type
        or col_dims != configured.storage_schema.dimensions
    ):
        raise PreflightError(
            f"Existing '{namespace}.embedding' column is '{fmt}', but the configuration expects "
            f"{configured.storage_schema.column_type}({configured.storage_schema.dimensions}) "
            f"(EMBEDDING_MODEL={settings.EMBEDDING_MODEL}, "
            f"EMBEDDING_DIMENSIONS={settings.EMBEDDING_DIMENSIONS}). Changing embedding "
            "dimensions/type requires recreating the store and re-embedding all sources."
        )
    if _is_legacy_default(configured) or settings.EMBEDDING_ADOPT_EXISTING:
        _write_manifest(conn, namespace, configured)
        logger.info("Adopted existing document store '%s' and wrote its manifest.", namespace)
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
    _create_indexes(conn, namespace, dims)
    _write_manifest(conn, namespace, configured)
    logger.info(
        "Created fresh document store '%s' (dims=%d) and wrote its manifest.", namespace, dims
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
            logger.info("Document store '%s' is compatible with its manifest.", namespace)
            return

        if _table_exists(conn, namespace):
            _adopt_legacy(conn, settings, configured, namespace)
        else:
            _create_fresh(conn, configured, namespace, dims)


# --------------------------------------------------------------------------- #
# Redis preflight                                                             #
# --------------------------------------------------------------------------- #
@contextmanager
def _redis_lock(client: Any, key: str) -> Iterator[str]:
    """Short-lived distributed lock: SET NX EX with a unique token, released via a
    compare-and-delete Lua script so we never delete a replacement process's lock."""
    token = secrets.token_hex(16)
    deadline = time.monotonic() + PREFLIGHT_LOCK_TIMEOUT_S
    while not client.set(key, token, nx=True, ex=REDIS_LOCK_TTL_S):
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


def _canon(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _adopt_legacy_redis(
    client: Any, settings: Settings, configured: StoreManifest, name: str
) -> None:
    from redisvl.index import SearchIndex  # type: ignore

    existing = SearchIndex.from_existing(name, redis_client=client)
    attrs = existing.schema.fields["embedding"].attrs
    existing_dims = int(getattr(attrs, "dims"))
    if existing_dims != configured.storage_schema.dimensions:
        raise PreflightError(
            f"Existing Redis index '{name}' has embedding dims {existing_dims}, but the "
            f"configuration expects {configured.storage_schema.dimensions} "
            f"(EMBEDDING_MODEL={settings.EMBEDDING_MODEL}). Changing embedding dimensions "
            "requires dropping the index and its keys and re-embedding all sources."
        )
    existing_schema = (
        _canon(getattr(attrs, "datatype", "float32")),
        _canon(getattr(attrs, "distance_metric", "cosine")),
        _canon(getattr(attrs, "algorithm", "hnsw")),
    )
    if existing_schema != ("float32", "cosine", "hnsw"):
        raise PreflightError(
            f"Existing Redis index '{name}' vector schema {existing_schema} is incompatible "
            "with the configured float32/cosine/hnsw; drop the index and its keys and re-embed."
        )
    if _is_legacy_default(configured) or settings.EMBEDDING_ADOPT_EXISTING:
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

    with _redis_lock(client, lock_key):
        index = SearchIndex(build_index_schema(name, dims), redis_client=client)

        existing_raw = client.get(manifest_key)
        if existing_raw:
            comparison = compare_manifests(
                configured, StoreManifest.from_dict(json.loads(existing_raw))
            )
            if not comparison.compatible:
                raise PreflightError(comparison.message)
            client.delete(marker_key)  # clear any stale init marker
            logger.info("Redis document store '%s' is compatible with its manifest.", name)
            return

        if index.exists():
            marker_raw = client.get(marker_key)
            if marker_raw:
                if json.loads(marker_raw).get("fingerprint") == fingerprint:
                    # Interrupted init for THIS config (index created, manifest not
                    # written): safe to resume.
                    client.set(manifest_key, json.dumps(configured.to_dict()))
                    client.delete(marker_key)
                    logger.info("Resumed interrupted Redis index init for '%s'.", name)
                    return
                raise PreflightError(
                    f"Redis index '{name}' carries an initialization marker for a different "
                    "configuration (an interrupted init). Drop the index and its keys, then re-embed."
                )
            _adopt_legacy_redis(client, settings, configured, name)
            return

        # Fresh: init-marker protocol makes a crash between create and manifest-write
        # recoverable (and an unmarked index is never silently adopted).
        client.set(marker_key, json.dumps({"fingerprint": fingerprint}))
        index.create()
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
