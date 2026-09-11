"""Thin LangChain adapter around an HNSW-backed pgvector table."""
from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_postgres.v2.indexes import DistanceStrategy, HNSWIndex

from .renderers import REPRESENTATION_VERSION, render_candidate
from .schemas import CandidateProfile


@dataclass(frozen=True)
class SearchResult:
    candidate_id: str
    rank: int
    distance: float


def database_id(candidate_id: str) -> str:
    """Map a candidate ID to the stable UUID expected by the store schema."""
    return str(uuid5(NAMESPACE_URL, f"marketplace-ranking-system/{candidate_id}"))


def candidate_document(candidate: CandidateProfile) -> Document:
    """Create the only document shape that may enter the candidate index."""
    return Document(
        id=candidate.candidate_id,
        page_content=render_candidate(candidate),
        metadata={
            "candidate_id": candidate.candidate_id,
            "representation_version": REPRESENTATION_VERSION,
        },
    )


class CandidateVectorStore:
    """Own table creation, candidate upserts, HNSW creation, and search."""

    def __init__(
        self,
        connection_string: str,
        embeddings: Embeddings,
        vector_size: int,
        table_name: str = "candidate_embeddings",
    ) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be positive")
        self._engine = PGEngine.from_connection_string(connection_string)
        self._embeddings = embeddings
        self._vector_size = vector_size
        self._table_name = table_name
        self._store: PGVectorStore | None = None

    def initialize(self, *, overwrite: bool = False) -> None:
        """Create the vector table. Overwriting must be explicitly requested."""
        self._engine.init_vectorstore_table(
            table_name=self._table_name,
            vector_size=self._vector_size,
            overwrite_existing=overwrite,
        )
        self.connect()

    def connect(self) -> None:
        """Connect to a table that was previously created by initialize()."""
        self._store = PGVectorStore.create_sync(
            engine=self._engine,
            table_name=self._table_name,
            embedding_service=self._embeddings,
            distance_strategy=DistanceStrategy.COSINE_DISTANCE,
        )

    def build_hnsw(self, *, m: int = 16, ef_construction: int = 64) -> None:
        """Build a cosine HNSW index using named, experiment-visible settings."""
        store = self._require_store()
        store.apply_vector_index(
            HNSWIndex(
                name=f"{self._table_name}_hnsw_cosine",
                distance_strategy=DistanceStrategy.COSINE_DISTANCE,
                m=m,
                ef_construction=ef_construction,
            )
        )

    def upsert_candidates(self, candidates: Iterable[CandidateProfile]) -> list[str]:
        """Insert candidates using their stable IDs.

        Existing IDs should be deleted by a later refresh workflow after comparing
        content hashes; this method intentionally does not hide that policy.
        """
        documents = [candidate_document(candidate) for candidate in candidates]
        if not documents:
            return []
        return self._require_store().add_documents(
            documents, ids=[database_id(str(document.id)) for document in documents]
        )

    def add_cached_embeddings(
        self,
        candidate_ids: list[str],
        embeddings: list[list[float]],
        content_hashes: list[str],
    ) -> list[str]:
        """Load cached vectors without asking the embedding provider to recompute."""
        if not (len(candidate_ids) == len(embeddings) == len(content_hashes)):
            raise ValueError("candidate IDs, embeddings, and hashes must have equal lengths")
        if not candidate_ids:
            return []
        metadata = [
            {"candidate_id": candidate_id, "content_sha256": digest,
             "representation_version": REPRESENTATION_VERSION}
            for candidate_id, digest in zip(candidate_ids, content_hashes, strict=True)
        ]
        return self._require_store().add_embeddings(
            texts=candidate_ids,
            embeddings=embeddings,
            metadatas=metadata,
            ids=[database_id(candidate_id) for candidate_id in candidate_ids],
        )

    def search(self, query: str, k: int) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        matches = self._require_store().similarity_search_with_score(query, k=k)
        return [
            SearchResult(
                candidate_id=str(document.metadata["candidate_id"]),
                rank=rank,
                distance=float(distance),
            )
            for rank, (document, distance) in enumerate(matches, start=1)
        ]

    def search_by_vector(self, embedding: list[float], k: int) -> list[SearchResult]:
        """Search a cached job vector without making an embedding API call."""
        if k <= 0:
            raise ValueError("k must be positive")
        if len(embedding) != self._vector_size:
            raise ValueError("query embedding has the wrong dimension")
        matches = self._require_store().similarity_search_with_score_by_vector(
            embedding=embedding, k=k
        )
        return [
            SearchResult(
                candidate_id=str(document.metadata["candidate_id"]),
                rank=rank,
                distance=float(distance),
            )
            for rank, (document, distance) in enumerate(matches, start=1)
        ]

    def close(self) -> None:
        asyncio.run(self._engine.close())

    async def aclose(self) -> None:
        """Close from async environments such as a Jupyter kernel."""
        await self._engine.close()

    def _require_store(self) -> PGVectorStore:
        if self._store is None:
            raise RuntimeError("initialize() must be called first")
        return self._store
