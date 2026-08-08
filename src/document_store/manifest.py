"""Embedding-store manifest: identity + schema fingerprint persisted alongside a
document store so startup preflight can detect incompatible configuration changes
and emit the *correct* remediation (re-embed vs rebuild-index vs nothing).

The manifest is split into three sections because different mismatches need
different fixes:

- ``embedding_identity`` (provider/model/dims/metric/endpoint-space) — a change
  means the stored vectors occupy a different space, so documents must be
  **re-embedded**.
- ``storage_schema`` (physical column type + dimensions) — an incompatible
  change (e.g. different dims) also requires **re-embedding**.
- ``index_schema`` (algorithm/opclass/expression/build params) — a change can be
  satisfied by **rebuilding the index only**, keeping the stored float32 vectors.

Tuning knobs that don't affect stored data or the index definition (candidate
multiplier, ``ef_search``, iterative-scan mode) are deliberately **not** part of
the manifest, so changing them needs no migration.

This module is backend-agnostic and pure (no DB / Redis access) so it can be
unit-tested in isolation; persistence lives in the individual store backends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

# Bump when the *format* of the manifest itself changes (not when a deployment's
# embedding/index config changes). Older known versions may be migrated in place;
# an unknown *newer* version must never be overwritten.
MANIFEST_VERSION = 1


class Remediation(str, Enum):
    """What an operator must do to reconcile a configured store with an existing one."""

    COMPATIBLE = "compatible"  # no migration needed
    REBUILD_INDEX = "rebuild_index"  # index definition changed; stored vectors are fine
    REEMBED = "reembed"  # embedding identity or storage dims changed
    UNSUPPORTED_VERSION = "unsupported_version"  # existing manifest is newer than we understand


def normalize_endpoint_origin(base_url: str | None) -> str | None:
    """Reduce a provider base URL to a stable, **non-secret** origin identity
    (``scheme://host[:port]``), stripping credentials, path, query and fragment.

    Used so two openai-compatible/ollama deployments that share provider+model+dims
    but point at different servers are treated as different embedding spaces.
    Returns ``None`` for a falsy input.
    """
    if not base_url:
        return None
    if "://" not in base_url:
        # Not a URL — a bare host or host:port ("myhost:11434" would otherwise be
        # misparsed by urlsplit as scheme "myhost", dropping the port and collapsing
        # distinct endpoints). Strip any credentials, lowercase, keep the port.
        token = base_url.strip().rsplit("@", 1)[-1].lower()
        return token or None
    parts = urlsplit(base_url)
    host = (parts.hostname or "").lower()
    origin = f"{parts.scheme.lower()}://{host}" if parts.scheme else host
    if parts.port is not None:
        origin = f"{origin}:{parts.port}"
    return origin


@dataclass(frozen=True)
class EmbeddingIdentity:
    """The vector space the stored embeddings live in. A change requires re-embedding."""

    provider: str
    model: str
    dimensions: int
    distance_metric: str = "cosine"
    # Endpoint-space identity for providers where provider+model+dims is ambiguous
    # (openai-compatible / ollama). None for standard OpenAI (provider identity suffices).
    space_id: str | None = None


@dataclass(frozen=True)
class StorageSchema:
    """Physical storage of the embedding column. Incompatible change ⇒ re-embed."""

    column_type: str  # e.g. "vector" (float32) — the column we always keep
    dimensions: int


@dataclass(frozen=True)
class IndexSchema:
    """The vector index definition. Change ⇒ rebuild index only (vectors kept)."""

    algorithm: str  # "ivfflat" | "hnsw"
    opclass: str  # "vector_cosine_ops" | "halfvec_cosine_ops"
    expression: str | None  # e.g. "embedding::halfvec(3072)"; None = plain column index
    build_params: dict[str, Any] = field(default_factory=dict)  # {"lists":100} | {"m":16,"ef_construction":64}
    # Physical index name (namespace-derived for fresh stores, legacy name for adopted
    # ones). Administrative metadata only — excluded from compatibility equality.
    name: str | None = field(default=None, compare=False)


@dataclass(frozen=True)
class StoreManifest:
    manifest_version: int
    embedding_identity: EmbeddingIdentity
    storage_schema: StorageSchema
    index_schema: IndexSchema

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "embedding_identity": {
                "provider": self.embedding_identity.provider,
                "model": self.embedding_identity.model,
                "dimensions": self.embedding_identity.dimensions,
                "distance_metric": self.embedding_identity.distance_metric,
                "space_id": self.embedding_identity.space_id,
            },
            "storage_schema": {
                "column_type": self.storage_schema.column_type,
                "dimensions": self.storage_schema.dimensions,
            },
            "index_schema": {
                "algorithm": self.index_schema.algorithm,
                "opclass": self.index_schema.opclass,
                "expression": self.index_schema.expression,
                "build_params": dict(self.index_schema.build_params),
                "name": self.index_schema.name,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoreManifest":
        ei = data["embedding_identity"]
        ss = data["storage_schema"]
        ix = data["index_schema"]
        return cls(
            manifest_version=int(data["manifest_version"]),
            embedding_identity=EmbeddingIdentity(
                provider=ei["provider"],
                model=ei["model"],
                dimensions=int(ei["dimensions"]),
                distance_metric=ei.get("distance_metric", "cosine"),
                space_id=ei.get("space_id"),
            ),
            storage_schema=StorageSchema(
                column_type=ss["column_type"],
                dimensions=int(ss["dimensions"]),
            ),
            index_schema=IndexSchema(
                algorithm=ix["algorithm"],
                opclass=ix["opclass"],
                expression=ix.get("expression"),
                build_params=dict(ix.get("build_params") or {}),
                name=ix.get("name"),
            ),
        )


@dataclass(frozen=True)
class ManifestComparison:
    remediation: Remediation
    message: str

    @property
    def compatible(self) -> bool:
        return self.remediation is Remediation.COMPATIBLE


def compare_manifests(
    configured: StoreManifest, existing: StoreManifest
) -> ManifestComparison:
    """Compare the configured store against the persisted one and return the
    least-destructive remediation required. Order matters: check the format
    version first, then most-destructive (re-embed) before least (rebuild index).
    """
    # 1. Format version. Fail closed on ANY version mismatch — a newer manifest must
    # never be overwritten, and an older one must not be reinterpreted through today's
    # field semantics without an explicit migration handler (none exists yet).
    if existing.manifest_version != configured.manifest_version:
        return ManifestComparison(
            Remediation.UNSUPPORTED_VERSION,
            f"Existing store manifest version {existing.manifest_version} is not supported by "
            f"this build (expected {configured.manifest_version}) and no migration handler is "
            "registered. Refusing to start to avoid misreading a store written by a different "
            "Ragpi version.",
        )

    # 2. Embedding identity — the stored vectors live in a different space.
    if configured.embedding_identity != existing.embedding_identity:
        return ManifestComparison(
            Remediation.REEMBED,
            "Embedding identity changed "
            f"(existing: {existing.embedding_identity}; configured: {configured.embedding_identity}). "
            "Stored vectors are incompatible — you must recreate the store and re-embed all "
            "sources. See 'Changing the embedding model' in the README.",
        )

    # 3. Physical storage — incompatible column type / dimensions ⇒ re-embed.
    if configured.storage_schema != existing.storage_schema:
        return ManifestComparison(
            Remediation.REEMBED,
            "Storage schema changed "
            f"(existing: {existing.storage_schema}; configured: {configured.storage_schema}). "
            "Recreate the store and re-embed all sources.",
        )

    # 4. Index definition — vectors are fine, but the index must be rebuilt.
    if configured.index_schema != existing.index_schema:
        return ManifestComparison(
            Remediation.REBUILD_INDEX,
            "Vector index definition changed "
            f"(existing: {existing.index_schema}; configured: {configured.index_schema}). "
            "Stored vectors are compatible — rebuild the vector index (no re-embedding needed).",
        )

    return ManifestComparison(
        Remediation.COMPATIBLE, "Store configuration is compatible; no migration needed."
    )
