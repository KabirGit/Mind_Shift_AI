from __future__ import annotations

from pydantic import BaseModel, Field


class JournalRecord(BaseModel):
    """Structured record for a journal entry, stored in SQLite.

    Additive companion to the existing FAISS metadata; does not replace it.
    """

    id: str  # = entry_hash from existing FaissVectorStore
    text: str
    timestamp: str  # ISO8601, same format as MemoryEntry.timestamp
    emotion: str
    emotion_confidence: float

    entities_people: list[str] = Field(default_factory=list)
    entities_places: list[str] = Field(default_factory=list)
    entities_orgs: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    habits: list[str] = Field(default_factory=list)
    person_relationship_types: dict[str, str] = Field(default_factory=dict)

    sentiment_compound: float = 0.0  # VADER compound score, -1 to 1
    sentiment_valence: float = 0.0  # alias/derived, -1 to 1
