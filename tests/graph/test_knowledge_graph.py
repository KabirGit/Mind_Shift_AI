from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.graph.knowledge_graph import KnowledgeGraph
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


def _ts(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat().replace(
        "+00:00", "Z"
    )


def _seed(db: JournalDB) -> None:
    db.insert(JournalRecord(id="1", text="work with Sarah", timestamp=_ts(5),
                            emotion="joy", emotion_confidence=0.9,
                            topics=["career"], entities_people=["Sarah"],
                            person_relationship_types={"Sarah": "colleague"},
                            habits=["exercise"], sentiment_compound=0.3))
    db.insert(JournalRecord(id="2", text="career and Sarah again", timestamp=_ts(3),
                            emotion="fear", emotion_confidence=0.9,
                            topics=["career"], entities_people=["Sarah"],
                            person_relationship_types={"Sarah": "colleague"},
                            sentiment_compound=-0.1))


def test_build_and_query(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    _seed(db)
    kg = KnowledgeGraph(db)
    g = kg.build(lookback_days=90)

    assert "User" in g
    assert g.has_edge("User", "career")
    assert g.has_edge("User", "Sarah")
    # career and Sarah co-occur
    assert g.has_edge("career", "Sarah")

    res = kg.query(g, "career")
    assert "Sarah" in res["neighbors"]
    assert "User" in res["neighbors"]

    summary = kg.summarize_node(g, "career")
    assert "career" in summary


def test_people_graph_shape(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    _seed(db)
    out = KnowledgeGraph(db).people_graph(lookback_days=90)
    assert out["nodes"][0] == {
        "id": "User",
        "label": "You",
        "type": "user",
        "relationship_type": "self",
        "mention_count": 0,
    }
    sarah = [node for node in out["nodes"] if node["id"] == "Sarah"][0]
    assert sarah["relationship_type"] == "colleague"
    edge = out["edges"][0]
    assert edge["source"] == "User"
    assert edge["target"] == "Sarah"
    assert edge["weight"] == 2
    assert edge["closeness_score"] > 0


def test_missing_node_safe(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    _seed(db)
    kg = KnowledgeGraph(db)
    g = kg.build()
    assert kg.query(g, "nonexistent") == {"neighbors": [], "edge_data": []}
    assert "No connections" in kg.summarize_node(g, "nonexistent")
