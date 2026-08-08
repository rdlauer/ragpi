from sqlalchemy import Engine, func, text
from sqlalchemy.orm import aliased, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from pgvector.sqlalchemy import HALFVEC  # type: ignore
import numpy as np
from openai import OpenAI

from src.document_store.schemas import Document
from src.document_store.base import DocumentStoreBackend
from src.document_store.postgres.model import DocumentStoreModel
from src.document_store.ranking import reciprocal_rank_fusion

# pgvector caps the approximate index on the `vector` type at 2000 dimensions; above
# that we keep the float32 `vector` column but query through a half-precision (halfvec)
# expression index and rerank candidates by exact float32 cosine. hnsw.ef_search maxes
# at 1000, so we cap there and rely on iterative scans for larger candidate sets.
HALFVEC_INDEX_THRESHOLD = 2000
MAX_HNSW_EF_SEARCH = 1000


class PostgresDocumentStore(DocumentStoreBackend):
    def __init__(
        self,
        *,
        engine: Engine,
        openai_client: OpenAI,
        embedding_model: str,
        embedding_dimensions: int,
        candidate_multiplier: int = 10,
        hnsw_ef_search: int | None = None,
    ):
        self.engine = engine
        self.Session = sessionmaker(bind=self.engine)
        self.embedding_client = openai_client.embeddings
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.candidate_multiplier = candidate_multiplier
        self.hnsw_ef_search = hnsw_ef_search
        self.DocumentModel = DocumentStoreModel
        # Schema/extension creation is handled once at startup by the document-store
        # preflight (src/document_store/preflight.py), not per-request here.

    def _map_document(self, doc: DocumentStoreModel) -> Document:
        return Document(
            id=doc.id,
            content=doc.content,
            title=doc.title,
            url=doc.url,
            created_at=doc.created_at,
        )

    def add_documents(self, source_name: str, documents: list[Document]) -> None:
        embeddings_result = self.embedding_client.create(
            input=[doc.content for doc in documents],
            model=self.embedding_model,
            dimensions=self.embedding_dimensions,
        )

        docs_to_add = [
            self.DocumentModel(
                id=doc.id,
                source=source_name,
                content=doc.content,
                title=doc.title,
                url=doc.url,
                created_at=doc.created_at,
                embedding=np.array(embedding_data.embedding, dtype=np.float32).tolist(),
            )
            for doc, embedding_data in zip(documents, embeddings_result.data)
        ]

        with self.Session() as session:
            try:
                session.bulk_save_objects(docs_to_add)
                session.commit()
            except SQLAlchemyError:
                session.rollback()
                raise

    def get_documents(
        self, source_name: str, limit: int, offset: int
    ) -> list[Document]:
        with self.Session() as session:
            results = (
                session.query(self.DocumentModel)
                .filter_by(source=source_name)
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [self._map_document(doc) for doc in results]

    def get_document_ids(self, source_name: str) -> list[str]:
        with self.Session() as session:
            results = (
                session.query(self.DocumentModel.id).filter_by(source=source_name).all()
            )
            return [row[0] for row in results]

    def delete_all_documents(self, source_name: str) -> None:
        with self.Session() as session:
            session.query(self.DocumentModel).filter_by(source=source_name).delete()
            session.commit()

    def delete_documents(self, source_name: str, doc_ids: list[str]) -> None:
        with self.Session() as session:
            session.query(self.DocumentModel).filter(
                self.DocumentModel.id.in_(doc_ids)
            ).delete(synchronize_session=False)
            session.commit()

    def semantic_search(
        self, source_name: str, query: str, top_k: int
    ) -> list[Document]:
        query_embedding = (
            self.embedding_client.create(
                input=query,
                model=self.embedding_model,
                dimensions=self.embedding_dimensions,
            )
            .data[0]
            .embedding
        )

        if self.embedding_dimensions > HALFVEC_INDEX_THRESHOLD:
            return self._semantic_search_halfvec(source_name, query_embedding, top_k)

        # <=2000 dims: direct exact float32 cosine search over the ivfflat index.
        with self.Session() as session:
            results = (
                session.query(self.DocumentModel)
                .filter_by(source=source_name)
                .order_by(self.DocumentModel.embedding.cosine_distance(query_embedding))  # type: ignore
                .limit(top_k)
                .all()
            )
            return [self._map_document(doc) for doc in results]

    def _semantic_search_halfvec(
        self, source_name: str, query_embedding: list[float], top_k: int
    ) -> list[Document]:
        """Two-stage retrieval for >2000-dim embeddings: fetch candidates ordered by
        half-precision (halfvec) cosine distance, then rerank by exact float32 cosine
        over the retained `vector` column — no precision loss in the final ranking.

        The candidate ordering matches the HNSW halfvec expression index, but Postgres
        chooses the access path per query: for small/medium sources it prefers an exact
        top-N scan of the filtered source (cheap and perfectly accurate at that scale)
        and switches to the index as sources grow. The ef_search / iterative_scan
        settings below only take effect on queries the planner serves via the index.
        """
        Model = self.DocumentModel
        dims = self.embedding_dimensions
        candidate_count = max(top_k, top_k * self.candidate_multiplier)
        # ef_search must be >= the number of rows fetched from the index, else the
        # over-fetch is undermined; capped at pgvector's max, with iterative scans
        # covering larger candidate sets and the source filter.
        ef_search = min(MAX_HNSW_EF_SEARCH, max(self.hnsw_ef_search or 0, candidate_count))

        # The candidate ordering expression must match the index expression exactly.
        half_distance = func.cast(Model.embedding, HALFVEC(dims)).cosine_distance(
            query_embedding
        )

        with self.Session() as session:
            # pgvector's hnsw.* GUCs are only registered once its library is loaded in
            # the session; touch a vector value first so set_config recognizes them.
            session.execute(text("SELECT '[1]'::vector"))
            session.execute(
                text("SELECT set_config('hnsw.ef_search', :ef, true)"),
                {"ef": str(ef_search)},
            )
            session.execute(
                text("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)")
            )

            candidate_subq = (
                session.query(Model)
                .filter(Model.source == source_name)
                .order_by(half_distance)  # halfvec ANN — uses the HNSW expression index
                .limit(candidate_count)
                .subquery()
            )
            candidate = aliased(Model, candidate_subq)
            results = (
                session.query(candidate)
                .order_by(candidate.embedding.cosine_distance(query_embedding))  # exact float32 rerank
                .limit(top_k)
                .all()
            )
            return [self._map_document(doc) for doc in results]

    def full_text_search(
        self, source_name: str, query: str, top_k: int
    ) -> list[Document]:
        with self.Session() as session:
            ts_query = func.websearch_to_tsquery("english", query)
            results = (
                session.query(self.DocumentModel)
                .filter(
                    self.DocumentModel.source == source_name,
                    self.DocumentModel.fts_vector.op("@@")(ts_query),  # type: ignore
                )
                .order_by(func.ts_rank(self.DocumentModel.fts_vector, ts_query).desc())  # type: ignore
                .limit(top_k)
                .all()
            )
            return [self._map_document(doc) for doc in results]

    def hybrid_search(
        self, *, source_name: str, semantic_query: str, full_text_query: str, top_k: int
    ) -> list[Document]:
        semantic_results = self.semantic_search(source_name, semantic_query, top_k)
        text_results = self.full_text_search(source_name, full_text_query, top_k)
        return reciprocal_rank_fusion([semantic_results, text_results], top_k)
