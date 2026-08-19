from __future__ import annotations

import os
from collections import Counter
from io import BytesIO
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.analytics._stats_utils import filter_window
from backend.api.rag_service import RAGService
from backend.api.schemas import (
    ChatRequest,
    ChatResponse,
    DashboardSummaryResponse,
    DiagnosticsResponse,
    EmotionPoint,
    GoalsResponse,
    GraphQueryResponse,
    GrowthResponse,
    HealthResponse,
    PredictionsResponse,
    TimelineResponse,
)
from backend.config.logger import setup_logging

setup_logging()

app = FastAPI(
    title="Mind Shift AI API",
    version="1.0.0",
    description="REST transport for the existing Mind Shift AI RAG service.",
)


def _allowed_origins() -> list[str]:
    configured = os.getenv("ALLOWED_ORIGIN", "")
    origins = [o.strip() for o in configured.split(",") if o.strip()]
    if not origins:
        return ["http://localhost:3000", "http://127.0.0.1:3000"]
    return origins


def _allowed_origin_regex() -> str | None:
    return os.getenv("ALLOWED_ORIGIN_REGEX", r"https://.*\.pages\.dev")


app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_origin_regex=_allowed_origin_regex(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_SERVICE: RAGService | None = None


def get_service() -> RAGService:
    """Single shared demo service, matching the current one-user Streamlit model."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = RAGService()
    return _SERVICE


def _lookback_days(range_label: str) -> int:
    normalized = range_label.strip().lower().replace("_", " ")
    aliases = {
        "7": 7,
        "7 days": 7,
        "last 7 days": 7,
        "30": 30,
        "30 days": 30,
        "last 30 days": 30,
        "all": 36500,
        "all time": 36500,
    }
    return aliases.get(normalized, 30)


def _emotion_over_time(records) -> list[EmotionPoint]:
    counts: Counter[tuple[str, str]] = Counter()
    for record in records:
        date = (record.timestamp or "")[:10]
        if date:
            counts[(date, record.emotion or "neutral")] += 1
    return [
        EmotionPoint(date=date, emotion=emotion, count=count)
        for (date, emotion), count in sorted(counts.items())
    ]


def _crisis_payload(crisis: dict[str, Any]) -> dict[str, Any]:
    if not crisis.get("flagged"):
        return crisis
    from backend.safety.crisis_detector import CRISIS_MESSAGE

    return {**crisis, "resources": CRISIS_MESSAGE}


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    service: Annotated[RAGService, Depends(get_service)],
) -> ChatResponse:
    output = service.run_pipeline(
        text=request.text,
        chat_history=[m.model_dump() for m in request.chat_history],
        top_k=request.top_k,
        tags=request.tags,
    )
    packet = output.get("packet")
    return ChatResponse(
        emotion=output.get("emotion", {}),
        response=output.get("response", ""),
        memory_replay=getattr(packet, "memory_replay", None)
        if packet is not None
        else None,
        crisis=_crisis_payload(output.get("crisis", {})),
        retrieved_memories=output.get("retrieved_memories", []),
        stored_entry=output.get("stored_entry"),
        packet=packet,
        prompt=output.get("prompt"),
    )


@app.get("/api/dashboard/summary", response_model=DashboardSummaryResponse)
def dashboard_summary(
    service: Annotated[RAGService, Depends(get_service)],
    range_label: Annotated[str, Query(alias="range")] = "Last 30 days",
) -> DashboardSummaryResponse:
    lookback = _lookback_days(range_label)
    records = filter_window(service.journal_db.get_all(), lookback)
    summary = service.pattern_engine.analyze(lookback_days=lookback)
    return DashboardSummaryResponse(
        range=range_label,
        lookback_days=lookback,
        emotion_over_time=_emotion_over_time(records),
        pattern_summary=summary,
        recurring_topics=summary.recurring_topics,
        triggers=summary.triggers,
        habits=service.habit_engine.analyze(lookback_days=lookback),
        relationships=service.relationship_engine.analyze(lookback_days=lookback),
        insights=service.insight_engine.generate(lookback_days=lookback),
    )


@app.get("/api/dashboard/goals", response_model=GoalsResponse)
def dashboard_goals(
    service: Annotated[RAGService, Depends(get_service)],
) -> GoalsResponse:
    return GoalsResponse(goals=service.goal_engine.analyze(lookback_days=90))


@app.get("/api/dashboard/predictions", response_model=PredictionsResponse)
def dashboard_predictions(
    service: Annotated[RAGService, Depends(get_service)],
) -> PredictionsResponse:
    return PredictionsResponse(
        sentiment_forecast=service.prediction_engine.forecast_sentiment(),
        burnout_risk=service.prediction_engine.assess_burnout_risk(),
    )


@app.get("/api/dashboard/timeline", response_model=TimelineResponse)
def dashboard_timeline(
    service: Annotated[RAGService, Depends(get_service)],
) -> TimelineResponse:
    return TimelineResponse(events=service.timeline_engine.build(lookback_days=90))


@app.get("/api/dashboard/growth", response_model=GrowthResponse)
def dashboard_growth(
    service: Annotated[RAGService, Depends(get_service)],
) -> GrowthResponse:
    return GrowthResponse(
        snapshots=service.growth_tracker.compute_snapshots(),
        narrative=service.growth_tracker.narrative(),
    )


@app.get("/api/graph/query", response_model=GraphQueryResponse)
def graph_query(
    node: str,
    service: Annotated[RAGService, Depends(get_service)],
) -> GraphQueryResponse:
    graph = service.knowledge_graph.build(lookback_days=90)
    result = service.knowledge_graph.query(graph, node)
    return GraphQueryResponse(
        node=node,
        summary=service.knowledge_graph.summarize_node(graph, node),
        neighbors=result.get("neighbors", []),
        edge_data=result.get("edge_data", []),
    )


@app.get("/api/report/weekly")
def weekly_report(
    service: Annotated[RAGService, Depends(get_service)],
) -> StreamingResponse:
    report_bytes = service.report_generator.generate(lookback_days=30)
    headers = {"Content-Disposition": 'attachment; filename="weekly_report.pdf"'}
    return StreamingResponse(
        BytesIO(report_bytes),
        media_type="application/pdf",
        headers=headers,
    )


@app.get("/api/diagnostics", response_model=DiagnosticsResponse)
def diagnostics(
    service: Annotated[RAGService, Depends(get_service)],
) -> DiagnosticsResponse:
    return DiagnosticsResponse(
        retrieval_precision=service.eval_engine.retrieval_precision_at_k(k=3),
        emotion_confidence=service.eval_engine.emotion_confidence_stats(),
        latency=service.eval_engine.latency_summary(),
    )


@app.options("/{path:path}")
def options_preflight() -> Response:
    return Response(status_code=204)
