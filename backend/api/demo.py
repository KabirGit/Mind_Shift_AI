from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from backend.api.schemas import (
    DashboardStoryResponse,
    DashboardSummaryResponse,
    DemoChatHistoryResponse,
    DiagnosticsResponse,
    GoalsResponse,
    GraphPeopleResponse,
    GraphQueryResponse,
    GrowthResponse,
    PredictionsResponse,
    TimelineResponse,
)

router = APIRouter(prefix="/api/demo", tags=["demo"])
_DEMO_DATA_DIR = Path(__file__).resolve().parents[1] / "demo_data"


def _read_json(filename: str) -> Any:
    return json.loads((_DEMO_DATA_DIR / filename).read_text(encoding="utf-8"))


@router.get("/dashboard/summary", response_model=DashboardSummaryResponse)
def demo_dashboard_summary() -> Any:
    return _read_json("dashboard_summary.json")


@router.get(
    "/dashboard/story",
    response_model=DashboardStoryResponse,
    response_model_exclude_none=True,
)
def demo_dashboard_story() -> Any:
    return _read_json("dashboard_story.json")


@router.get("/dashboard/goals", response_model=GoalsResponse)
def demo_dashboard_goals() -> Any:
    return _read_json("goals.json")


@router.get("/dashboard/predictions", response_model=PredictionsResponse)
def demo_dashboard_predictions() -> Any:
    return _read_json("predictions.json")


@router.get("/dashboard/timeline", response_model=TimelineResponse)
def demo_dashboard_timeline() -> Any:
    return _read_json("timeline.json")


@router.get("/dashboard/growth", response_model=GrowthResponse)
def demo_dashboard_growth() -> Any:
    return _read_json("growth.json")


@router.get("/diagnostics", response_model=DiagnosticsResponse)
def demo_diagnostics() -> Any:
    return _read_json("diagnostics.json")


@router.get("/graph/people", response_model=GraphPeopleResponse)
def demo_graph_people() -> Any:
    return _read_json("graph_people.json")


@router.get("/graph/query", response_model=GraphQueryResponse)
def demo_graph_query(node: str = Query(default="User")) -> Any:
    queries = _read_json("graph_queries.json")
    return queries.get(
        node.lower(),
        {
            "node": node,
            "summary": f"No demo connections found for '{node}'.",
            "neighbors": [],
            "edge_data": [],
        },
    )


@router.get("/chat-history", response_model=DemoChatHistoryResponse)
def demo_chat_history() -> Any:
    return _read_json("chat_transcript.json")
