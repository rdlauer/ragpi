import pytest
from pydantic import ValidationError

from src.config import Settings


def _settings(**overrides) -> Settings:
    base: dict = dict(OPENAI_API_KEY="test-key")
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def test_retrieval_top_k_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="RETRIEVAL_TOP_K must be >= 1"):
        _settings(RETRIEVAL_TOP_K=0)


def test_hnsw_ef_search_upper_bound() -> None:
    with pytest.raises(ValidationError, match="HNSW_EF_SEARCH must be between 1 and 1000"):
        _settings(HNSW_EF_SEARCH=1001)


def test_hnsw_ef_search_lower_bound() -> None:
    with pytest.raises(ValidationError, match="HNSW_EF_SEARCH must be between 1 and 1000"):
        _settings(HNSW_EF_SEARCH=0)


def test_hnsw_ef_search_at_limit_is_allowed() -> None:
    assert _settings(HNSW_EF_SEARCH=1000).HNSW_EF_SEARCH == 1000


def test_candidate_multiplier_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="EMBEDDING_CANDIDATE_MULTIPLIER must be >= 1"):
        _settings(EMBEDDING_CANDIDATE_MULTIPLIER=0)
