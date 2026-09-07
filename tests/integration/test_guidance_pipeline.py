from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backend.api.rag_service import RAGService
from backend.llm.prompt_builder import PromptBuilder
from backend.storage.db import JournalDB


class FakeVectorStore:
    @staticmethod
    def _hash_entry(text, timestamp):
        normalized = " ".join(str(text).strip().casefold().split())
        return hashlib.sha256(normalized.encode()).hexdigest()


class FakeEmotion:
    def detect(self, text):
        emotion = "fear" if "afraid" in text.casefold() else "neutral"
        return {"emotion": emotion, "confidence": 0.8 if emotion == "fear" else 0.2}


class FakeRetriever:
    def __init__(self):
        self.calls = 0

    def retrieve(self, query, query_emotion="neutral", top_k=5):
        self.calls += 1
        return [
            {
                "metadata": {
                    "text": (
                        "I am afraid. Should I stay in my current role or accept the "
                        "new job?"
                    ),
                    "emotion": query_emotion,
                    "timestamp": "2026-09-05T10:00:00Z",
                },
                "scores": {"combined": 1.0},
            },
            {
                "metadata": {
                    "text": "A prior career choice improved after a small trial.",
                    "emotion": "fear",
                    "timestamp": "2026-08-01T10:00:00Z",
                    "topics": ["career"],
                },
                "scores": {"combined": 0.82},
            },
        ]


class FakeMemory:
    def store_entry(self, text, tags=None, emotion_signal=None, topics=None):
        return {
            "text": text,
            "emotion": (emotion_signal or {}).get("emotion", "neutral"),
            "timestamp": "2026-09-05T10:00:00Z",
            "topics": topics or [],
        }

    def get_recent_memory(self, limit=3):
        return []


class FakeTextProcessor:
    def extract(self, text):
        return {
            "entities_people": [],
            "entities_places": [],
            "entities_orgs": [],
            "keywords": ["career"],
            "topics": ["career"],
            "habits": [],
            "sentiment_compound": -0.2,
            "sentiment_valence": -0.2,
        }


class QueueLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FailingGuidancePrompt(PromptBuilder):
    def build_guidance(self, **kwargs):
        raise RuntimeError("guidance prompt unavailable")


def _decision_json():
    return json.dumps(
        {
            "is_decision": True,
            "problem": "Choose between staying and accepting a new job",
            "desired_outcome": "make a sustainable career move",
            "available_options": ["Stay in my current role", "Accept the new job"],
            "fears": ["I am afraid of making the wrong move"],
            "constraints": ["I need stable income"],
            "relevant_goals": ["career growth"],
            "missing_information": ["actual team culture"],
            "uncertainty": 0.5,
            "suggested_options": [],
            "confidence": 0.9,
        }
    )


def _service(tmp_path, llm, prompt_builder=None):
    return RAGService(
        vector_store=FakeVectorStore(),
        emotion_detector=FakeEmotion(),
        retriever=FakeRetriever(),
        prompt_builder=prompt_builder or PromptBuilder(),
        llm_client=llm,
        memory_manager=FakeMemory(),
        journal_db=JournalDB(str(tmp_path / "guidance.db")),
        text_processor=FakeTextProcessor(),
    )


def test_decision_uses_two_llm_calls_and_filters_current_memory(tmp_path):
    final = json.dumps(
        {
            "validation": "This is understandably difficult.",
            "recommendation": "Gather evidence first.",
            "why": ["A trial limits downside."],
            "uncertainty": ["Team culture is unknown."],
            "next_actions": ["Ask questions.", "Run a small trial."],
        }
    )
    llm = QueueLLM([_decision_json(), final])
    service = _service(tmp_path, llm)

    output = service.run_pipeline(
        "I am afraid. Should I stay in my current role or accept the new job?"
    )

    assert output["mode"] == "guidance"
    assert output["decision_state"] is not None
    assert output["guidance"] is not None
    assert output["response"] == (
        "Validation:\nThis is understandably difficult.\n\n"
        "Recommendation:\nGather evidence first.\n\n"
        "Why:\n- A trial limits downside.\n\n"
        "Uncertainty:\n- Team culture is unknown.\n\n"
        "Next steps:\n1. Ask questions.\n2. Run a small trial."
    )
    assert len(llm.prompts) == 2
    assert output["agent_steps"] <= 3
    assert len(output["tools_called"]) == output["agent_steps"]
    assert len(output["retrieved_memories"]) == 1
    assert "prior career choice" in output["retrieved_memories"][0]["metadata"]["text"]


def test_normal_journaling_keeps_one_call_reflection_path(tmp_path):
    llm = QueueLLM(["legacy reflection"])
    service = _service(tmp_path, llm)

    output = service.run_pipeline("I had a calm walk and felt better today.")

    assert output["mode"] == "reflection"
    assert output["response"] == "legacy reflection"
    assert len(llm.prompts) == 1
    assert llm.prompts[0].startswith("You are a journaling companion.")


def test_high_risk_decision_uses_no_llm_or_retrieval(tmp_path):
    llm = QueueLLM([])
    service = _service(tmp_path, llm)

    output = service.run_pipeline("Should I double my antidepressant dose tonight?")

    assert output["mode"] == "safety"
    assert output["decision_state"] is None
    assert output["guidance"] is None
    assert output["retrieved_memories"] == []
    assert llm.prompts == []
    assert service.retriever.calls == 0


def test_unexpected_guidance_failure_falls_back_to_reflection(tmp_path):
    llm = QueueLLM([_decision_json(), "legacy reflection"])
    service = _service(tmp_path, llm, FailingGuidancePrompt())

    output = service.run_pipeline("Should I stay in my job or accept the offer?")

    assert output["mode"] == "reflection"
    assert output["response"] == "legacy reflection"
    assert output["guidance"] is None
    assert len(llm.prompts) == 2


def test_malformed_final_guidance_uses_structured_local_renderer(tmp_path):
    llm = QueueLLM([_decision_json(), "A vague answer without the contract."])
    service = _service(tmp_path, llm)

    output = service.run_pipeline("Should I stay in my job or accept the offer?")

    assert output["mode"] == "guidance"
    for section in ("Validation:", "Recommendation:", "Why:", "Uncertainty:", "Next steps:"):
        assert section in output["response"]


def _last_trace(path: Path):
    return json.loads(path.read_text(encoding="utf-8").splitlines()[-1])


def test_guidance_trace_is_structured_and_redacted(tmp_path):
    final = json.dumps(
        {
            "validation": "This is difficult.",
            "recommendation": "Gather evidence.",
            "why": ["It limits downside."],
            "uncertainty": ["Culture is unknown."],
            "next_actions": ["Ask.", "Review."],
        }
    )
    service = _service(tmp_path, QueueLLM([_decision_json(), final]))
    trace_path = tmp_path / "guidance.jsonl"
    service._latency_log_path = str(trace_path)

    output = service.run_pipeline("Should I stay in my job or accept the offer?")
    trace = _last_trace(trace_path)

    assert trace["trace_id"] == output["trace_id"]
    assert trace["mode"] == "guidance"
    assert trace["agent_steps"] <= 3
    assert trace["llm_calls"] == 2
    assert trace["logical_llm_calls"] == 2
    assert trace["provider_attempts"] == 2
    assert trace["retry_count"] == 0
    assert trace["prompt_versions"] == ["decision-parse-v2", "guidance-response-v2"]
    assert trace["status"] == "success"
    assert trace["agent_termination_reason"] in {
        "sufficient_context",
        "max_calls",
        "no_additional_tool_needed",
    }
    assert trace["context_selection"]["estimated_tokens"] <= 1500
    assert "total" in trace["stage_latencies_ms"]
    assert trace["latency_ms"] == trace["elapsed_ms"]
    serialized = json.dumps(trace)
    assert "Should I stay" not in serialized
    assert "prior career choice" not in serialized


def test_reflection_safety_and_fallback_trace_outcomes(tmp_path):
    reflection = _service(tmp_path, QueueLLM(["legacy reflection"]))
    reflection_path = tmp_path / "reflection.jsonl"
    reflection._latency_log_path = str(reflection_path)
    reflection.run_pipeline("I had a calm day.")
    assert _last_trace(reflection_path)["outcome"] == "reflected"

    safety = _service(tmp_path, QueueLLM([]))
    safety_path = tmp_path / "safety.jsonl"
    safety._latency_log_path = str(safety_path)
    safety.run_pipeline("Should I double my prescription dose?")
    safety_trace = _last_trace(safety_path)
    assert safety_trace["mode"] == "safety"
    assert safety_trace["tools_called"] == []
    assert safety_trace["llm_calls"] == 0

    fallback = _service(
        tmp_path,
        QueueLLM([_decision_json(), "legacy reflection"]),
        FailingGuidancePrompt(),
    )
    fallback_path = tmp_path / "fallback.jsonl"
    fallback._latency_log_path = str(fallback_path)
    fallback.run_pipeline("Should I stay or accept the offer?")
    assert _last_trace(fallback_path)["outcome"] == (
        "guidance_failure_reflection_fallback"
    )


def test_trace_append_failure_never_breaks_pipeline(tmp_path):
    service = _service(tmp_path, QueueLLM(["legacy reflection"]))
    service._latency_log_path = str(tmp_path)
    output = service.run_pipeline("I had a calm day.")
    assert output["response"] == "legacy reflection"
