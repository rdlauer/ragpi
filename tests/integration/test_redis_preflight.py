import json
from typing import Generator, cast

import pytest
from testcontainers.redis import RedisContainer  # type: ignore

from src.common.redis import create_redis_client
from src.config import Settings
from src.document_store.manifest import StoreManifest
from src.document_store.preflight import (
    PreflightError,
    _fingerprint,
    build_configured_manifest,
    run_preflight,
)

NS = "document_store"
MANIFEST_KEY = f"{NS}:__manifest__"
MARKER_KEY = f"{NS}:__init__"


def _settings(url: str, **overrides) -> Settings:
    base: dict = dict(
        OPENAI_API_KEY="test-key",
        REDIS_URL=url,
        DOCUMENT_STORE_BACKEND="redis",
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


@pytest.fixture
def redis_url(redis_container: RedisContainer) -> Generator[str, None, None]:
    url = (
        f"redis://{redis_container.get_container_host_ip()}:"
        f"{redis_container.get_exposed_port(6379)}"
    )
    client = create_redis_client(url)
    client.flushdb()
    yield url
    client.flushdb()
    client.close()


def _read_manifest(url: str) -> StoreManifest | None:
    client = create_redis_client(url)
    try:
        raw = cast("str | None", client.get(MANIFEST_KEY))
    finally:
        client.close()
    return StoreManifest.from_dict(json.loads(raw)) if raw else None


def test_fresh_create_writes_index_and_manifest(redis_url: str) -> None:
    settings = _settings(redis_url)
    run_preflight(settings)

    client = create_redis_client(redis_url)
    try:
        from redisvl.index import SearchIndex  # type: ignore

        from src.document_store.redis.store import build_index_schema

        index = SearchIndex(build_index_schema(NS, 1536), redis_client=client)
        assert index.exists()
        assert client.get(MARKER_KEY) is None  # init marker cleaned up
    finally:
        client.close()

    assert _read_manifest(redis_url) == build_configured_manifest(settings, "redis")


def test_preflight_is_idempotent(redis_url: str) -> None:
    settings = _settings(redis_url)
    run_preflight(settings)
    run_preflight(settings)
    assert _read_manifest(redis_url) == build_configured_manifest(settings, "redis")


def test_adopts_legacy_default_index_without_manifest(redis_url: str) -> None:
    settings = _settings(redis_url)
    run_preflight(settings)  # create index + manifest

    client = create_redis_client(redis_url)
    try:
        client.delete(MANIFEST_KEY)  # simulate a pre-manifest deployment
    finally:
        client.close()

    run_preflight(settings)  # legacy default → adopt + rewrite manifest
    assert _read_manifest(redis_url) == build_configured_manifest(settings, "redis")


def test_manifest_model_mismatch_fails(redis_url: str) -> None:
    settings = _settings(redis_url)
    run_preflight(settings)

    other = build_configured_manifest(
        _settings(redis_url, EMBEDDING_MODEL="text-embedding-3-large", EMBEDDING_DIMENSIONS=1536),
        "redis",
    )
    client = create_redis_client(redis_url)
    try:
        client.set(MANIFEST_KEY, json.dumps(other.to_dict()))
    finally:
        client.close()

    with pytest.raises(PreflightError, match="Embedding identity changed"):
        run_preflight(settings)


def test_crash_between_index_and_manifest_resumes(redis_url: str) -> None:
    settings = _settings(redis_url)
    run_preflight(settings)  # creates index + manifest

    # Simulate a crash after index creation but before the manifest was written:
    # manifest gone, an init marker for THIS config present.
    fingerprint = _fingerprint(build_configured_manifest(settings, "redis"))
    client = create_redis_client(redis_url)
    try:
        client.delete(MANIFEST_KEY)
        client.set(MARKER_KEY, json.dumps({"fingerprint": fingerprint}))
    finally:
        client.close()

    run_preflight(settings)  # resumes: writes manifest, clears marker
    assert _read_manifest(redis_url) == build_configured_manifest(settings, "redis")

    client = create_redis_client(redis_url)
    try:
        assert client.get(MARKER_KEY) is None
    finally:
        client.close()


def test_crash_marker_for_different_config_fails(redis_url: str) -> None:
    settings = _settings(redis_url)
    run_preflight(settings)

    client = create_redis_client(redis_url)
    try:
        client.delete(MANIFEST_KEY)
        client.set(MARKER_KEY, json.dumps({"fingerprint": "some-other-config"}))
    finally:
        client.close()

    with pytest.raises(PreflightError, match="different configuration"):
        run_preflight(settings)
