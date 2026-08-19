from backend.retrieval.vector_store import FaissVectorStore


class _FakeModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 3


class _FakeEmbeddingPipeline:
    model = _FakeModel()

    def chunk_documents(self, documents):
        return []


def test_build_from_empty_documents_creates_queryable_empty_store(tmp_path):
    store = FaissVectorStore(
        persist_dir=str(tmp_path / "faiss_store"),
        embedding_pipeline=_FakeEmbeddingPipeline(),
    )

    store.build_from_documents([])

    assert store.index is not None
    assert store.index.ntotal == 0
    assert store.exists()
    assert store.query("anything") == []
