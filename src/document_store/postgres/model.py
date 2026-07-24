from datetime import datetime
from typing import Any
from sqlalchemy import Computed, DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector  # type: ignore
from sqlalchemy_utils import TSVectorType  # type: ignore

from src.config import get_settings  # type: ignore


settings = get_settings()


class Base(DeclarativeBase):
    pass


class DocumentStoreModel(Base):
    __tablename__ = settings.DOCUMENT_STORE_NAMESPACE

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source: Mapped[str] = mapped_column(String, index=True, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    embedding: Mapped[Any] = mapped_column(
        Vector(settings.EMBEDDING_DIMENSIONS), nullable=False
    )
    fts_vector: Mapped[Any] = mapped_column(
        TSVectorType("content", "title", regconfig="english"),
        Computed(
            "to_tsvector('english', title) || to_tsvector('english', content)",
            persisted=True,
        ),
    )

    # Indexes (the dimension-dependent vector index and the FTS GIN index) are
    # created by the document-store preflight with namespace-aware names, not here,
    # so create_all() manages only the table and adopted legacy stores keep their
    # existing physical index names. See src/document_store/preflight.py.
    __table_args__ = {"extend_existing": True}

    def __init__(
        self,
        id: str,
        source: str,
        title: str,
        content: str,
        url: str,
        created_at: datetime,
        embedding: Any,
    ):
        self.id = id
        self.source = source
        self.title = title
        self.content = content
        self.url = url
        self.created_at = created_at
        self.embedding = embedding
