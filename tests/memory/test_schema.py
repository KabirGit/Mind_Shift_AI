from __future__ import annotations

from backend.memory.schema import MemoryEntry


def test_metadata_includes_topics():
    meta = MemoryEntry(
        text="career stuff", timestamp="2026-06-20T12:00:00Z", topics=["career"]
    ).to_metadata()
    assert meta["topics"] == ["career"]


def test_default_topics_present_as_empty_list():
    meta = MemoryEntry(text="hi", timestamp="2026-06-20T12:00:00Z").to_metadata()
    assert "topics" in meta
    assert meta["topics"] == []
