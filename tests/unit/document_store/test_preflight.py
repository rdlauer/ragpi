import pytest

from src.config import Settings
from src.document_store.preflight import (
    PreflightError,
    advisory_lock_key,
    build_configured_manifest,
    parse_version,
    _ensure_extension_version,
    _index_name,
    _is_legacy_default,
)


def _settings(**overrides) -> Settings:
    base = dict(OPENAI_API_KEY="test-key")
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


class TestAdvisoryLockKey:
    def test_deterministic_and_signed_64bit(self):
        k1 = advisory_lock_key("ragpi", "document_store", "ns")
        k2 = advisory_lock_key("ragpi", "document_store", "ns")
        assert k1 == k2
        assert -(2**63) <= k1 < 2**63

    def test_scope_changes_key(self):
        assert advisory_lock_key("ragpi", "document_store", "a") != advisory_lock_key(
            "ragpi", "document_store", "b"
        )


class TestParseVersion:
    def test_semantic_not_lexicographic(self):
        assert parse_version("0.10.0") > parse_version("0.9.0")
        assert parse_version("0.8.2") >= (0, 8, 0)
        assert parse_version("0.7.4") < (0, 8, 0)

    def test_handles_none_and_suffix(self):
        assert parse_version(None) == (0,)
        assert parse_version("0.8.0-dev1")[:3] == (0, 8, 0)


class TestIndexName:
    def test_short_name_unchanged(self):
        assert _index_name("document_store", "embedding_idx") == "document_store_embedding_idx"

    def test_long_name_truncated_within_63_bytes_and_unique(self):
        ns_a = "a" * 80
        ns_b = "a" * 79 + "b"
        n_a = _index_name(ns_a, "embedding_idx")
        n_b = _index_name(ns_b, "embedding_idx")
        assert len(n_a.encode()) <= 63
        assert n_a != n_b  # digest keeps distinct long namespaces from colliding


class TestBuildConfiguredManifest:
    def test_small_uses_vector_ivfflat(self):
        m = build_configured_manifest(_settings())
        assert m.storage_schema.column_type == "vector"
        assert m.index_schema.algorithm == "ivfflat"
        assert m.index_schema.opclass == "vector_cosine_ops"
        assert m.index_schema.expression is None

    def test_large_uses_halfvec_hnsw_expression(self):
        m = build_configured_manifest(
            _settings(EMBEDDING_MODEL="text-embedding-3-large", EMBEDDING_DIMENSIONS=3072)
        )
        assert m.storage_schema.column_type == "vector"  # float32 column preserved
        assert m.storage_schema.dimensions == 3072
        assert m.index_schema.algorithm == "hnsw"
        assert m.index_schema.opclass == "halfvec_cosine_ops"
        assert m.index_schema.expression == "embedding::halfvec(3072)"
        assert m.index_schema.build_params == {"m": 16, "ef_construction": 64}

    def test_openai_has_no_endpoint_space_id(self):
        assert build_configured_manifest(_settings()).embedding_identity.space_id is None

    def test_ollama_derives_endpoint_space_id(self):
        m = build_configured_manifest(
            _settings(
                EMBEDDING_PROVIDER="ollama",
                OLLAMA_BASE_URL="http://ollama-host:11434/v1",
                EMBEDDING_MODEL="nomic-embed-text",
                EMBEDDING_DIMENSIONS=768,
            )
        )
        assert m.embedding_identity.space_id == "http://ollama-host:11434"

    def test_explicit_space_id_override(self):
        m = build_configured_manifest(
            _settings(
                EMBEDDING_PROVIDER="ollama",
                OLLAMA_BASE_URL="http://ollama-host:11434/v1",
                EMBEDDING_SPACE_ID="my-fixed-space",
                EMBEDDING_MODEL="nomic-embed-text",
                EMBEDDING_DIMENSIONS=768,
            )
        )
        assert m.embedding_identity.space_id == "my-fixed-space"


class TestIsLegacyDefault:
    def test_default_config_is_legacy_default(self):
        assert _is_legacy_default(build_configured_manifest(_settings())) is True

    def test_large_config_is_not_legacy_default(self):
        m = build_configured_manifest(
            _settings(EMBEDDING_MODEL="text-embedding-3-large", EMBEDDING_DIMENSIONS=3072)
        )
        assert _is_legacy_default(m) is False


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class _FakeConn:
    """Minimal stand-in for a SQLAlchemy connection for _ensure_extension_version."""

    def __init__(self, versions: list[str]):
        self._versions = list(versions)
        self.executed: list[str] = []

    def execute(self, statement, params=None):
        sql = str(statement)
        self.executed.append(sql)
        if "extversion" in sql:
            return _FakeResult(self._versions.pop(0))
        return _FakeResult(None)


class TestEnsureExtensionVersion:
    def test_small_dims_never_fail(self):
        conn = _FakeConn(["0.5.1"])  # old extension is fine for <=2000
        _ensure_extension_version(conn, _settings(), dims=1536)

    def test_large_dims_old_version_without_flag_raises(self):
        conn = _FakeConn(["0.7.4"])
        with pytest.raises(PreflightError, match="0.8.0"):
            _ensure_extension_version(
                conn,
                _settings(EMBEDDING_MODEL="text-embedding-3-large", EMBEDDING_DIMENSIONS=3072),
                dims=3072,
            )

    def test_large_dims_new_version_ok(self):
        conn = _FakeConn(["0.8.2"])
        _ensure_extension_version(
            conn,
            _settings(EMBEDDING_MODEL="text-embedding-3-large", EMBEDDING_DIMENSIONS=3072),
            dims=3072,
        )

    def test_large_dims_updates_when_authorized(self):
        conn = _FakeConn(["0.7.4", "0.8.2"])  # before + after ALTER
        _ensure_extension_version(
            conn,
            _settings(
                EMBEDDING_MODEL="text-embedding-3-large",
                EMBEDDING_DIMENSIONS=3072,
                PG_UPDATE_VECTOR_EXTENSION=True,
            ),
            dims=3072,
        )
        assert any("ALTER EXTENSION vector UPDATE" in sql for sql in conn.executed)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
