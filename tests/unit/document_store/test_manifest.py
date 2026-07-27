import pytest

from src.document_store.manifest import (
    MANIFEST_VERSION,
    EmbeddingIdentity,
    IndexSchema,
    Remediation,
    StorageSchema,
    StoreManifest,
    compare_manifests,
    normalize_endpoint_origin,
)


def _manifest(
    *,
    provider="openai",
    model="text-embedding-3-small",
    dimensions=1536,
    metric="cosine",
    space_id=None,
    column_type="vector",
    algorithm="ivfflat",
    opclass="vector_cosine_ops",
    expression=None,
    build_params=None,
    version=MANIFEST_VERSION,
) -> StoreManifest:
    return StoreManifest(
        manifest_version=version,
        embedding_identity=EmbeddingIdentity(
            provider=provider,
            model=model,
            dimensions=dimensions,
            distance_metric=metric,
            space_id=space_id,
        ),
        storage_schema=StorageSchema(column_type=column_type, dimensions=dimensions),
        index_schema=IndexSchema(
            algorithm=algorithm,
            opclass=opclass,
            expression=expression,
            build_params=build_params or {"lists": 100},
        ),
    )


class TestNormalizeEndpointOrigin:
    def test_none_and_empty(self):
        assert normalize_endpoint_origin(None) is None
        assert normalize_endpoint_origin("") is None

    def test_strips_credentials_path_and_query(self):
        origin = normalize_endpoint_origin(
            "https://user:secret@Example.com:8443/v1/embeddings?key=abc#frag"
        )
        assert origin == "https://example.com:8443"

    def test_default_port_preserved_only_when_explicit(self):
        assert normalize_endpoint_origin("http://host/v1") == "http://host"
        assert normalize_endpoint_origin("http://host:11434/v1") == "http://host:11434"

    def test_bare_host_without_scheme(self):
        assert normalize_endpoint_origin("Ollama-Host") == "ollama-host"

    def test_schemeless_host_port_keeps_the_port(self):
        # urlsplit would misparse "myhost:11434" as scheme "myhost" and drop the port,
        # collapsing distinct endpoints into one space identity.
        assert normalize_endpoint_origin("myhost:11434") == "myhost:11434"
        assert normalize_endpoint_origin("MyHost:11434") == "myhost:11434"
        assert normalize_endpoint_origin("myhost:11434") != normalize_endpoint_origin(
            "myhost:11435"
        )

    def test_schemeless_credentials_are_stripped(self):
        assert normalize_endpoint_origin("user:secret@MyHost:11434") == "myhost:11434"


class TestManifestRoundTrip:
    def test_to_from_dict_roundtrip(self):
        m = _manifest(
            provider="openai-compatible",
            space_id="https://api.internal:8443",
            column_type="vector",
            algorithm="hnsw",
            opclass="halfvec_cosine_ops",
            expression="embedding::halfvec(3072)",
            build_params={"m": 16, "ef_construction": 64},
            dimensions=3072,
        )
        assert StoreManifest.from_dict(m.to_dict()) == m


class TestCompareManifests:
    def test_identical_is_compatible(self):
        assert compare_manifests(_manifest(), _manifest()).remediation is Remediation.COMPATIBLE
        assert compare_manifests(_manifest(), _manifest()).compatible is True

    def test_newer_existing_version_is_unsupported(self):
        configured = _manifest(version=MANIFEST_VERSION)
        existing = _manifest(version=MANIFEST_VERSION + 1)
        result = compare_manifests(configured, existing)
        assert result.remediation is Remediation.UNSUPPORTED_VERSION
        assert result.compatible is False

    def test_older_existing_version_is_unsupported(self):
        # Fail closed on ANY version mismatch — an older manifest must not be
        # reinterpreted through today's field semantics without a migration handler.
        configured = _manifest(version=MANIFEST_VERSION)
        existing = _manifest(version=MANIFEST_VERSION - 1)
        result = compare_manifests(configured, existing)
        assert result.remediation is Remediation.UNSUPPORTED_VERSION
        assert result.compatible is False

    def test_physical_index_name_excluded_from_compatibility(self):
        from src.document_store.manifest import IndexSchema

        a = IndexSchema("ivfflat", "vector_cosine_ops", None, {"lists": 100}, name="a_idx")
        b = IndexSchema("ivfflat", "vector_cosine_ops", None, {"lists": 100}, name="b_idx")
        assert a == b  # name is administrative metadata, not a compatibility criterion

    def test_model_change_requires_reembed(self):
        result = compare_manifests(
            _manifest(model="text-embedding-3-large", dimensions=1536),
            _manifest(model="text-embedding-3-small", dimensions=1536),
        )
        assert result.remediation is Remediation.REEMBED

    def test_same_dims_different_model_requires_reembed(self):
        # The core reason the manifest exists: equal dimensions must NOT imply compatibility.
        result = compare_manifests(
            _manifest(model="text-embedding-3-large", dimensions=1536),
            _manifest(model="text-embedding-3-small", dimensions=1536),
        )
        assert result.remediation is Remediation.REEMBED

    def test_endpoint_space_change_requires_reembed(self):
        result = compare_manifests(
            _manifest(provider="openai-compatible", space_id="https://a.internal"),
            _manifest(provider="openai-compatible", space_id="https://b.internal"),
        )
        assert result.remediation is Remediation.REEMBED

    def test_dimension_change_requires_reembed(self):
        result = compare_manifests(
            _manifest(dimensions=3072), _manifest(dimensions=1536)
        )
        assert result.remediation is Remediation.REEMBED

    def test_index_only_change_requires_rebuild_not_reembed(self):
        # Same embedding identity + storage, different index → rebuild index, keep vectors.
        configured = _manifest(
            algorithm="hnsw",
            opclass="halfvec_cosine_ops",
            expression="embedding::halfvec(1536)",
            build_params={"m": 16, "ef_construction": 64},
        )
        existing = _manifest(
            algorithm="ivfflat", opclass="vector_cosine_ops", expression=None
        )
        result = compare_manifests(configured, existing)
        assert result.remediation is Remediation.REBUILD_INDEX

    def test_build_param_change_requires_rebuild(self):
        configured = _manifest(
            algorithm="hnsw", opclass="halfvec_cosine_ops", build_params={"m": 32}
        )
        existing = _manifest(
            algorithm="hnsw", opclass="halfvec_cosine_ops", build_params={"m": 16}
        )
        assert compare_manifests(configured, existing).remediation is Remediation.REBUILD_INDEX


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
