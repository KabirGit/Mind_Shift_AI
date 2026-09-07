from __future__ import annotations

from backend.retrieval.retriever import Retriever


class RankedStore:
    def __init__(self, items):
        self.items = items

    def query(self, query_text, top_k=5):
        return self.items[:top_k]


def _item(memory_id: str, text: str, distance: float, entry_hash: str):
    return {
        "distance": distance,
        "metadata": {
            "id": memory_id,
            "entry_hash": entry_hash,
            "text": text,
            "emotion": "neutral",
            "timestamp": "",
        },
    }


def test_guidance_diversity_keeps_relevant_ids_and_suppresses_duplicate_and_current() -> None:
    store = RankedStore(
        [
            _item("current", "Should I accept this job offer?", 0.05, "current-hash"),
            _item(
                "relevant",
                "I compared a job offer with staying in my role and asked about team culture.",
                0.10,
                "relevant-hash",
            ),
            _item(
                "duplicate",
                "I compared a job offer with staying in my role and asked about team culture!",
                0.11,
                "duplicate-hash",
            ),
            _item(
                "trial",
                "A short trial project helped me evaluate a career change.",
                0.20,
                "trial-hash",
            ),
            _item("irrelevant", "I cooked pasta for dinner.", 0.90, "irrelevant-hash"),
        ]
    )

    results = Retriever(store, candidate_pool=10).retrieve(
        "job offer career",
        top_k=3,
        exclude_entry_hashes={"current-hash"},
        diversify=True,
    )
    ids = [item["metadata"]["id"] for item in results]

    assert "current" not in ids
    assert "duplicate" not in ids
    assert "relevant" in ids
    assert "trial" in ids


def test_default_retrieval_order_remains_unchanged() -> None:
    store = RankedStore(
        [
            _item("first", "same", 0.1, "one"),
            _item("second", "same", 0.2, "two"),
        ]
    )

    results = Retriever(store).retrieve("query", top_k=2)

    assert [item["metadata"]["id"] for item in results] == ["first", "second"]
