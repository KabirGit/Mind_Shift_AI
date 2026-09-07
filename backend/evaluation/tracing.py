from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.guidance.models import AgentResult, ContextSelectionReport
from backend.llm.models import LLMCallResult

logger = logging.getLogger(__name__)

TraceStatus = Literal["success", "degraded", "blocked", "failed"]
TraceMode = Literal["reflection", "guidance", "safety"]


class ModelTrace(BaseModel):
    purpose: str
    requested_model: str
    actual_model: str | None = None
    prompt_version: str
    success: bool
    status_code: int | None = None
    attempts: int = Field(ge=0, le=2)
    retries: int = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0.0)
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    failure_category: str | None = None


class TraceRecord(BaseModel):
    timestamp: str
    trace_id: str
    mode: TraceMode
    requested_models: list[str] = Field(default_factory=list)
    actual_models: list[str] = Field(default_factory=list)
    prompt_versions: list[str] = Field(default_factory=list)
    logical_llm_calls: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    provider_attempts: int = Field(default=0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    token_usage: dict[str, int | None] = Field(default_factory=dict)
    model_calls: list[ModelTrace] = Field(default_factory=list)
    tools_called: list[str] = Field(default_factory=list, max_length=3)
    tool_observations: list[dict[str, Any]] = Field(default_factory=list, max_length=3)
    memories_retrieved: int = Field(default=0, ge=0)
    agent_steps: int = Field(default=0, ge=0, le=3)
    agent_termination_reason: str | None = None
    context_selection: dict[str, Any] = Field(default_factory=dict)
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)
    failure_category: str | None = None
    status: TraceStatus
    outcome: str
    latency_ms: float = Field(ge=0.0)
    elapsed_ms: float = Field(ge=0.0)


class RequestTrace:
    """Request-local accumulator containing operational metadata only."""

    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        self.started = time.perf_counter()
        self.model_calls: list[ModelTrace] = []
        self.tools_called: list[str] = []
        self.tool_observations: list[dict[str, Any]] = []
        self.memories_retrieved = 0
        self.agent_steps = 0
        self.agent_termination_reason: str | None = None
        self.context_selection: dict[str, Any] = {}
        self.stage_latencies_ms: dict[str, float] = {}
        self.failure_category: str | None = None

    def add_stage(self, name: str, started: float) -> None:
        elapsed = (time.perf_counter() - started) * 1000
        self.stage_latencies_ms[name] = round(
            self.stage_latencies_ms.get(name, 0.0) + elapsed, 2
        )

    def record_llm(self, result: LLMCallResult | None) -> None:
        if result is None:
            return
        self.model_calls.append(
            ModelTrace(
                purpose=result.purpose,
                requested_model=result.requested_model,
                actual_model=result.actual_model,
                prompt_version=result.prompt_version,
                success=result.success,
                status_code=result.status_code,
                attempts=result.attempts,
                retries=result.retry_count,
                latency_ms=round(result.latency_ms, 2),
                finish_reason=result.finish_reason,
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                total_tokens=result.usage.total_tokens,
                failure_category=result.failure_category,
            )
        )
        if not result.success and not self.failure_category:
            self.failure_category = result.failure_category or "model_failure"

    def record_agent(self, result: AgentResult) -> None:
        self.tools_called = list(result.tools_called)[:3]
        self.agent_steps = result.agent_steps
        self.agent_termination_reason = result.termination_reason
        self.tool_observations = [
            {
                "tool": observation.tool,
                "success": observation.success,
                "result_count": observation.result_count,
                "duration_ms": round(observation.duration_ms, 2),
                "failure_category": observation.failure_category,
            }
            for observation in result.observations[:3]
        ]
        failed = next(
            (item for item in result.observations if not item.success), None
        )
        if failed is not None and not self.failure_category:
            self.failure_category = failed.failure_category or "tool_failure"

    def record_context(self, selection: ContextSelectionReport | None) -> None:
        if selection is None:
            return
        self.context_selection = selection.model_dump(mode="json")

    def build(
        self,
        *,
        mode: TraceMode,
        status: TraceStatus,
        outcome: str,
        elapsed_ms: float | None = None,
    ) -> TraceRecord:
        total = (
            (time.perf_counter() - self.started) * 1000
            if elapsed_ms is None
            else elapsed_ms
        )
        total = round(max(0.0, total), 2)
        self.stage_latencies_ms["total"] = total
        prompt_tokens = self._sum_usage("prompt_tokens")
        completion_tokens = self._sum_usage("completion_tokens")
        total_tokens = self._sum_usage("total_tokens")
        return TraceRecord(
            timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            trace_id=self.trace_id,
            mode=mode,
            requested_models=[call.requested_model for call in self.model_calls],
            actual_models=[
                call.actual_model for call in self.model_calls if call.actual_model
            ],
            prompt_versions=[call.prompt_version for call in self.model_calls],
            logical_llm_calls=len(self.model_calls),
            llm_calls=len(self.model_calls),
            provider_attempts=sum(call.attempts for call in self.model_calls),
            retry_count=sum(call.retries for call in self.model_calls),
            token_usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            model_calls=self.model_calls,
            tools_called=self.tools_called,
            tool_observations=self.tool_observations,
            memories_retrieved=self.memories_retrieved,
            agent_steps=self.agent_steps,
            agent_termination_reason=self.agent_termination_reason,
            context_selection=self.context_selection,
            stage_latencies_ms=self.stage_latencies_ms,
            failure_category=self.failure_category,
            status=status,
            outcome=outcome,
            latency_ms=total,
            elapsed_ms=total,
        )

    def _sum_usage(self, field: str) -> int | None:
        values = [getattr(call, field) for call in self.model_calls]
        present = [value for value in values if value is not None]
        return sum(present) if present else None


def append_trace(path: str, record: TraceRecord) -> bool:
    """Append one redacted JSONL trace; logging failures are non-fatal."""

    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
        return True
    except Exception as exc:
        logger.warning("trace append failed error_type=%s", type(exc).__name__)
        return False
