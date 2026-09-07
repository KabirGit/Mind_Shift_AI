from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class GuidanceEvalCase:
    id: str
    category: str
    text: str
    expected_mode: str
    history_available: bool = True
    required_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    max_tool_calls: int = 3
    expected_memory_ids: list[str] = field(default_factory=list)
    forbidden_memory_ids: list[str] = field(default_factory=list)
    required_guidance_fields: list[str] = field(default_factory=list)
    expected_llm_calls: int | None = None
    expected_trace_status: str | None = None
    expected_provider_attempts: int | None = None
    expected_retry_count: int | None = None
    expected_termination_reason: str | None = None
    model_behavior: str = "success"
    retrieval_variant: str = "relevant"


@dataclass(frozen=True)
class PipelineEvalResult:
    id: str
    expected_mode: str
    actual_mode: str
    tools_called: list[str]
    memory_ids: list[str]
    trace_status: str | None
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


class CaseRunner(Protocol):
    def __call__(self, case: GuidanceEvalCase) -> dict[str, Any]: ...


def load_cases(path: str | Path) -> list[GuidanceEvalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GuidanceEvalCase(**item) for item in data]


def evaluate_pipeline(
    cases: list[GuidanceEvalCase], run_case: CaseRunner
) -> list[PipelineEvalResult]:
    """Evaluate only outputs returned by the real ``RAGService.run_pipeline`` path."""

    results = []
    for case in cases:
        output = run_case(case)
        trace = _as_dict(output.get("_trace"))
        results.append(
            PipelineEvalResult(
                id=case.id,
                expected_mode=case.expected_mode,
                actual_mode=str(output.get("mode", "missing")),
                tools_called=list(output.get("tools_called") or []),
                memory_ids=_memory_ids(output),
                trace_status=(str(trace["status"]) if trace.get("status") else None),
                failures=validate_output(case, output),
            )
        )
    return results


def validate_output(case: GuidanceEvalCase, output: dict[str, Any]) -> list[str]:
    """Return deterministic contract failures without judging response wording."""

    failures: list[str] = []
    mode = output.get("mode")
    if mode != case.expected_mode:
        failures.append(f"route: expected {case.expected_mode}, got {mode}")

    tools = list(output.get("tools_called") or [])
    steps = int(output.get("agent_steps", 0))
    if len(tools) > case.max_tool_calls or steps > case.max_tool_calls:
        failures.append(f"agent exceeded {case.max_tool_calls} tool calls")
    missing_tools = sorted(set(case.required_tools) - set(tools))
    if missing_tools:
        failures.append(f"required tools missing: {', '.join(missing_tools)}")
    forbidden_tools = sorted(set(case.forbidden_tools) & set(tools))
    if forbidden_tools:
        failures.append(f"forbidden tools called: {', '.join(forbidden_tools)}")

    memory_ids = set(_memory_ids(output))
    missing_memories = sorted(set(case.expected_memory_ids) - memory_ids)
    if missing_memories:
        failures.append(f"expected memories missing: {', '.join(missing_memories)}")
    forbidden_memories = sorted(set(case.forbidden_memory_ids) & memory_ids)
    if forbidden_memories:
        failures.append(f"forbidden memories selected: {', '.join(forbidden_memories)}")

    llm_calls = int(output.get("llm_calls", 0))
    if case.expected_llm_calls is not None and llm_calls != case.expected_llm_calls:
        failures.append(
            f"llm calls: expected {case.expected_llm_calls}, got {llm_calls}"
        )

    trace = _as_dict(output.get("_trace"))
    if (
        case.expected_trace_status is not None
        and trace.get("status") != case.expected_trace_status
    ):
        failures.append(
            f"trace status: expected {case.expected_trace_status}, "
            f"got {trace.get('status')}"
        )
    if (
        case.expected_provider_attempts is not None
        and int(trace.get("provider_attempts", -1)) != case.expected_provider_attempts
    ):
        failures.append(
            f"provider attempts: expected {case.expected_provider_attempts}, "
            f"got {trace.get('provider_attempts')}"
        )
    if (
        case.expected_retry_count is not None
        and int(trace.get("retry_count", -1)) != case.expected_retry_count
    ):
        failures.append(
            f"retry count: expected {case.expected_retry_count}, "
            f"got {trace.get('retry_count')}"
        )
    if (
        case.expected_termination_reason is not None
        and trace.get("agent_termination_reason")
        != case.expected_termination_reason
    ):
        failures.append(
            f"termination: expected {case.expected_termination_reason}, "
            f"got {trace.get('agent_termination_reason')}"
        )

    guidance = _as_dict(output.get("guidance"))
    if mode == "safety":
        if guidance:
            failures.append("safety output contained guidance")
        if tools:
            failures.append("safety output called guidance tools")
    elif mode == "reflection":
        if guidance or output.get("decision_state") is not None:
            failures.append("reflection output entered the guidance path")
    elif mode == "guidance":
        required = case.required_guidance_fields or [
            "options",
            "recommendation",
            "uncertainty",
            "evidence",
            "next_actions",
        ]
        missing = [key for key in required if key not in guidance]
        if missing:
            failures.append(f"guidance fields missing: {', '.join(missing)}")
        options = guidance.get("options") or []
        if not options:
            failures.append("guidance contained no options")
        for scored in options:
            option = _as_dict(_as_dict(scored).get("option"))
            if "benefits" not in option or "costs" not in option:
                failures.append("an option lacked explicit trade-offs")
                break
        if not guidance.get("next_actions"):
            failures.append("guidance contained no next actions")
        if not case.history_available and _claims_personal_history(
            guidance.get("evidence") or []
        ):
            failures.append("empty history became personal evidence")
    return failures


def compare_outputs(
    *,
    prompt: str,
    legacy_response: str,
    guidance_output: dict[str, Any],
) -> dict[str, Any]:
    """Stable artifact demonstrating the user-visible output change."""

    guidance = _as_dict(guidance_output.get("guidance"))
    return {
        "prompt": prompt,
        "before": {
            "mode": "reflection",
            "response": legacy_response,
            "structured_guidance": None,
        },
        "after": {
            "mode": guidance_output.get("mode"),
            "response": guidance_output.get("response"),
            "structured_guidance": {
                key: guidance.get(key)
                for key in (
                    "options",
                    "recommended_option_id",
                    "recommendation",
                    "confidence",
                    "evidence",
                    "uncertainty",
                    "next_actions",
                )
            },
        },
    }


def results_json(results: list[PipelineEvalResult]) -> str:
    return json.dumps(
        {
            "passed": sum(result.passed for result in results),
            "total": len(results),
            "cases": [asdict(result) | {"passed": result.passed} for result in results],
        },
        indent=2,
    )


def _memory_ids(output: dict[str, Any]) -> list[str]:
    ids = []
    for item in output.get("retrieved_memories") or []:
        metadata = _as_dict(_as_dict(item).get("metadata"))
        memory_id = metadata.get("id") or metadata.get("entry_hash")
        if memory_id:
            ids.append(str(memory_id))
    return ids


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value if isinstance(value, dict) else {}


def _claims_personal_history(evidence: list[Any]) -> bool:
    claims = " ".join(str(item).casefold() for item in evidence)
    return any(
        cue in claims
        for cue in (
            "you previously",
            "your past",
            "a similar entry was retrieved",
            "your history shows",
        )
    )
