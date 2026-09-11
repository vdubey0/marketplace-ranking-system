import unittest

from retrieval.schemas import CandidateProfile
from unittest.mock import Mock

from langchain_core.documents import Document

from retrieval.vector_store import CandidateVectorStore, candidate_document, database_id


class CandidateDocumentTests(unittest.TestCase):
    def test_document_contains_only_safe_metadata(self):
        document = candidate_document(
            CandidateProfile(candidate_id="c-1", current_title="Engineer")
        )
        self.assertEqual(document.id, "c-1")
        self.assertEqual(
            set(document.metadata), {"candidate_id", "representation_version"}
        )
        self.assertNotIn("name", document.page_content.casefold())

    def test_database_id_is_stable(self):
        self.assertEqual(database_id("c-1"), database_id("c-1"))
        self.assertNotEqual(database_id("c-1"), database_id("c-2"))

    def test_cached_vector_search_does_not_call_embedding_provider(self):
        store = object.__new__(CandidateVectorStore)
        store._vector_size = 2
        backend = Mock()
        backend.similarity_search_with_score_by_vector.return_value = [
            (Document(page_content="c-1", metadata={"candidate_id": "c-1"}), 0.1)
        ]
        store._store = backend
        result = store.search_by_vector([1.0, 0.0], 1)
        self.assertEqual(result[0].candidate_id, "c-1")
        backend.similarity_search_with_score_by_vector.assert_called_once_with(
            embedding=[1.0, 0.0], k=1
        )


if __name__ == "__main__":
    unittest.main()
