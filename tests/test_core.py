import torch
from src import embedding_store
from src.retriever import retrieve

class FakeEngine:
    def embed_query(self, question):
        assert question == "find the chart"
        return torch.tensor([[[1.0, 0.0]]])
    def score(self, query_embedding, page_embeddings):
        return torch.tensor([[0.15, 0.95, 0.42]])

def test_persistent_index_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(embedding_store, "INDEX_FILE", tmp_path / "index.pt")
    monkeypatch.setattr(embedding_store, "METADATA_FILE", tmp_path / "metadata.json")
    embeddings = torch.zeros((2, 3, 4))
    metadata = [
        {"filename": "one.pdf", "page_number": 1, "image_path": "one.jpg"},
        {"filename": "two.pdf", "page_number": 2, "image_path": "two.jpg"},
    ]
    embedding_store.save_index(embeddings, metadata)
    loaded = embedding_store.load_index()
    assert loaded["embeddings"].shape == (2, 3, 4)
    assert loaded["metadata"] == metadata

def test_retrieval_returns_highest_scoring_pages():
    index = {"embeddings": torch.zeros((3, 2, 2)), "metadata": [
        {"filename": "a.pdf", "page_number": 1, "image_path": "a.jpg"},
        {"filename": "b.pdf", "page_number": 2, "image_path": "b.jpg"},
        {"filename": "c.pdf", "page_number": 3, "image_path": "c.jpg"},
    ]}
    results, elapsed = retrieve("find the chart", index, FakeEngine(), top_k=2)
    assert [item["filename"] for item in results] == ["b.pdf", "c.pdf"]
    assert results[0]["score"] > results[1]["score"]
    assert elapsed >= 0
