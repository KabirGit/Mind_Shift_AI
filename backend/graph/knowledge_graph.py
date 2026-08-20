from __future__ import annotations

import logging
from collections import defaultdict

import networkx as nx

from backend.analytics._stats_utils import filter_window
from backend.analytics.relationship_engine import RelationshipEngine
from backend.storage.db import JournalDB

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Builds an in-memory networkx graph of the user's entities/topics/habits."""

    def __init__(self, db: JournalDB) -> None:
        self.db = db

    def build(self, lookback_days: int = 90) -> nx.Graph:
        graph = nx.Graph()
        graph.add_node("User")
        try:
            records = filter_window(self.db.get_all(), lookback_days)
            if not records:
                return graph

            topic_sent: dict[str, list[float]] = defaultdict(list)
            person_sent: dict[str, list[float]] = defaultdict(list)
            topic_count: dict[str, int] = defaultdict(int)
            person_count: dict[str, int] = defaultdict(int)

            for r in records:
                s = r.sentiment_compound
                topics = list(r.topics or [])
                people = list(r.entities_people or [])
                habits = list(r.habits or [])
                places = list(r.entities_places or [])
                orgs = list(r.entities_orgs or [])

                for t in topics:
                    topic_sent[t].append(s)
                    topic_count[t] += 1
                for p in people:
                    person_sent[p].append(s)
                    person_count[p] += 1

                # co-occurrence edges within the same record
                for t in topics:
                    for p in people:
                        self._touch_edge(graph, t, p, "co_occurs")
                    for h in habits:
                        self._touch_edge(graph, t, h, "co_occurs")

                for node in places + orgs + habits:
                    graph.add_node(node)

            for t, count in topic_count.items():
                avg = sum(topic_sent[t]) / len(topic_sent[t])
                graph.add_edge("User", t, weight=count, sentiment=round(avg, 4),
                               type="topic")
            for p, count in person_count.items():
                avg = sum(person_sent[p]) / len(person_sent[p])
                graph.add_edge(
                    "User",
                    p,
                    weight=count,
                    sentiment=round(avg, 4),
                    type="person",
                    relationship_type="unknown",
                    closeness_score=0.0,
                )

            for profile in RelationshipEngine(self.db).analyze(lookback_days=lookback_days):
                graph.add_node(
                    profile.person,
                    type="person",
                    relationship_type=profile.relationship_type,
                    mention_count=profile.mention_count,
                )
                graph.add_edge(
                    "User",
                    profile.person,
                    weight=profile.mention_count,
                    sentiment=profile.avg_sentiment,
                    type="person",
                    relationship_type=profile.relationship_type,
                    closeness_score=profile.closeness_score,
                )

            return graph
        except Exception as exc:
            logger.exception("KnowledgeGraph.build failed: %s", exc)
            return graph

    @staticmethod
    def _touch_edge(graph: nx.Graph, a: str, b: str, edge_type: str) -> None:
        if a == b:
            return
        if graph.has_edge(a, b):
            graph[a][b]["weight"] = graph[a][b].get("weight", 1) + 1
        else:
            graph.add_edge(a, b, weight=1, type=edge_type)

    def query(self, graph: nx.Graph, node: str) -> dict:
        if node not in graph:
            return {"neighbors": [], "edge_data": []}
        neighbors = list(graph.neighbors(node))
        edge_data = [dict(graph[node][n], neighbor=n) for n in neighbors]
        return {"neighbors": neighbors, "edge_data": edge_data}

    def summarize_node(self, graph: nx.Graph, node: str) -> str:
        if node not in graph:
            return f"No connections found for '{node}'."
        ranked = sorted(
            graph.neighbors(node),
            key=lambda n: graph[node][n].get("weight", 1),
            reverse=True,
        )
        top = ranked[:5]
        if not top:
            return f"'{node}' has no connections yet."
        return f"'{node}' connects to: {', '.join(top)}."

    def people_graph(self, lookback_days: int = 90) -> dict:
        profiles = RelationshipEngine(self.db).analyze(lookback_days=lookback_days)
        nodes = [
            {
                "id": "User",
                "label": "You",
                "type": "user",
                "relationship_type": "self",
                "mention_count": 0,
            }
        ]
        edges = []
        for profile in profiles:
            nodes.append(
                {
                    "id": profile.person,
                    "label": profile.person,
                    "type": "person",
                    "relationship_type": profile.relationship_type,
                    "mention_count": profile.mention_count,
                }
            )
            edges.append(
                {
                    "source": "User",
                    "target": profile.person,
                    "sentiment": profile.avg_sentiment,
                    "weight": profile.mention_count,
                    "closeness_score": profile.closeness_score,
                }
            )
        return {"nodes": nodes, "edges": edges}
