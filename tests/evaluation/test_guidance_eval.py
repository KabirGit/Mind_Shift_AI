from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.evaluation.guidance_eval import (
    GuidanceEvalCase,
    compare_outputs,
    evaluate_pipeline,
    load_cases,
    validate_output,
)
from backend.llm.models import LLMCallResult, LLMRequest, TokenUsage
from tests.integration.test_guidance_pipeline import _decision_json, _service

FIXTURE = Path(__file__).parent / "fixtures" / "guidance_cases.json"


class EvalLLM:
    def __init__(self, behavior: str) -> None:
        self.behavior = behavior

    def complete(self, request: LLMRequest) -> LLMCallResult:
        attempts = (
            2
            if self.behavior == "retry_recovery"
            and request.purpose == "decision_parse"
            else 1
        )
        common: dict[str, Any] = {
            "purpose": request.purpose,
            "prompt_version": request.prompt_version,
            "requested_model": "eval-requested-model",
            "actual_model": "eval-actual-model",
            "attempts": attempts,
            "retry_count": attempts - 1,
            "usage": TokenUsage(
                prompt_tokens=12, completion_tokens=8, total_tokens=20
            ),
        }
        if self.behavior == "permanent_failure":
            return LLMCallResult(
                **{**common, "actual_model": None},
                success=False,
                status_code=401,
                failure_category="authentication",
                error_type="HTTPError",
            )
        if request.purpose == "decision_parse":
            payload = json.loads(_decision_json())
            if "afraid" not in request.user_prompt.casefold():
                payload["fears"] = []
            return LLMCallResult(
                **common, success=True, text=json.dumps(payload)
            )
        if request.purpose == "guidance_response":
            text = (
                '{"validation":"I understand."}'
                if self.behavior == "invalid_guidance_schema"
                else json.dumps(
                    {
                        "validation": "This decision carries real uncertainty.",
                        "recommendation": "Gather the missing evidence before committing.",
                        "why": ["A small test contains downside."],
                        "uncertainty": ["The future team context is not verified."],
                        "next_actions": [
                            "Write down the non-negotiable constraints.",
                            "Ask one targeted question before deciding.",
                        ],
                    }
                )
            )
            return LLMCallResult(**common, success=True, text=text)
        return LLMCallResult(
            **common,
            success=True,
            text="It makes sense to pause and notice what this day brought up.",
        )


class EvalRetriever:
    def __init__(self, variant: str) -> None:
        self.variant = variant

    def retrieve(
        self,
        query: str,
        query_emotion: str = "neutral",
        top_k: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        if self.variant == "empty":
            return []
        relevant = {
            "metadata": {
                "id": "career-relevant",
                "entry_hash": "career-relevant-hash",
                "text": "A prior career choice became clearer after asking about team culture.",
                "emotion": query_emotion,
                "timestamp": "2026-08-01T10:00:00Z",
                "topics": ["career"],
            },
            "scores": {"combined": 0.85},
        }
        if self.variant != "diverse":
            return [relevant]
        trial = {
            "metadata": {
                "id": "career-trial",
                "entry_hash": "career-trial-hash",
                "text": "A short trial project helped evaluate a different role.",
                "emotion": "neutral",
                "timestamp": "2026-07-01T10:00:00Z",
                "topics": ["career"],
            },
            "scores": {"combined": 0.72},
        }
        return [relevant, trial][:top_k]


def _run_case(root: Path, case: GuidanceEvalCase) -> dict[str, Any]:
    case_dir = root / case.id
    case_dir.mkdir()
    service = _service(case_dir, EvalLLM(case.model_behavior))
    retriever = EvalRetriever(case.retrieval_variant)
    service.retriever = retriever
    service.decision_context_builder.retriever = retriever
    trace_path = case_dir / "trace.jsonl"
    service._latency_log_path = str(trace_path)

    output = service.run_pipeline(case.text)
    output["_trace"] = json.loads(
        trace_path.read_text(encoding="utf-8").splitlines()[-1]
    )
    return output


def test_golden_dataset_exercises_production_pipeline(tmp_path):
    cases = load_cases(FIXTURE)
    results = evaluate_pipeline(cases, lambda case: _run_case(tmp_path, case))

    assert len(results) == 15
    assert all(result.passed for result in results), {
        result.id: result.failures for result in results if not result.passed
    }


def test_safety_contract_rejects_guidance_or_tools():
    case = GuidanceEvalCase("risk", "medical", "Should I change a dose?", "safety")
    valid = {"mode": "safety", "guidance": None, "tools_called": [], "agent_steps": 0}
    assert validate_output(case, valid) == []

    invalid = {
        "mode": "safety",
        "guidance": {"recommendation": "change it"},
        "tools_called": ["search_similar_memories"],
    }
    failures = validate_output(case, invalid)
    assert "safety output contained guidance" in failures
    assert "safety output called guidance tools" in failures


def test_empty_history_cannot_be_presented_as_personal_evidence():
    case = GuidanceEvalCase(
        "none", "no_history", "Should I stay or go?", "guidance", False
    )
    output = {
        "mode": "guidance",
        "guidance": {
            "options": [{"option": {"benefits": [], "costs": []}}],
            "recommendation": "More information is needed.",
            "uncertainty": ["personal history"],
            "evidence": ["Your history shows that leaving works for you."],
            "next_actions": [{"action": "Gather information."}],
        },
    }
    assert validate_output(case, output) == ["empty history became personal evidence"]


def test_comparison_exposes_structured_difference():
    comparison = compare_outputs(
        prompt="Should I stay or accept the offer?",
        legacy_response="What feelings come up when you consider the offer?",
        guidance_output={
            "mode": "guidance",
            "response": "Recommendation: gather evidence.",
            "guidance": {
                "options": ["stay", "accept"],
                "recommended_option_id": None,
                "recommendation": "More information is needed.",
                "confidence": 0.3,
                "evidence": [],
                "uncertainty": ["team culture"],
                "next_actions": ["ask questions"],
            },
        },
    )
    assert comparison["before"]["structured_guidance"] is None
    assert comparison["after"]["structured_guidance"]["options"] == [
        "stay",
        "accept",
    ]
