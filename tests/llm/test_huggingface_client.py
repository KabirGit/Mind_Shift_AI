from __future__ import annotations

from typing import Any

import requests

from backend.llm.huggingface_client import HuggingFaceInferenceClient
from backend.llm.models import LLMRequest


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self.payload = payload or {}
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError("provider request failed")
            error.response = self  # type: ignore[assignment]
            raise error

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.response


def _request() -> LLMRequest:
    return LLMRequest(
        purpose="decision_parse",
        system_prompt="Return bounded JSON.",
        user_prompt="Should I resign or ask for feedback?",
        prompt_version="test-hf-v1",
        response_format="json_object",
        temperature=0.0,
        max_tokens=300,
    )


def _success_payload() -> dict[str, Any]:
    return {
        "model": "arsoban/ocd-therapist-27b-v0.3",
        "choices": [
            {
                "message": {"content": '{"problem":"work decision"}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 8,
            "total_tokens": 28,
        },
    }


def test_routes_requested_model_through_featherless() -> None:
    session = FakeSession(FakeResponse(payload=_success_payload()))
    client = HuggingFaceInferenceClient(
        "arsoban/ocd-therapist-27b-v0.3",
        "hf_secret",
        provider="featherless-ai",
        session=session,
    )

    result = client.complete(_request())

    assert result.success is True
    assert result.requested_model == (
        "arsoban/ocd-therapist-27b-v0.3:featherless-ai"
    )
    assert result.actual_model == "arsoban/ocd-therapist-27b-v0.3"
    assert result.usage.total_tokens == 28
    call = session.calls[0]
    assert call["args"][0] == (
        "https://router.huggingface.co/v1/chat/completions"
    )
    assert call["kwargs"]["json"]["model"] == (
        "arsoban/ocd-therapist-27b-v0.3:featherless-ai"
    )
    assert call["kwargs"]["json"]["response_format"] == {
        "type": "json_object"
    }
    assert call["kwargs"]["headers"]["Authorization"] == "Bearer hf_secret"


def test_does_not_append_a_second_provider_suffix() -> None:
    session = FakeSession(FakeResponse(payload=_success_payload()))
    client = HuggingFaceInferenceClient(
        "arsoban/ocd-therapist-27b-v0.3:featherless-ai",
        "hf_secret",
        provider="featherless-ai",
        session=session,
    )

    result = client.complete(_request())

    assert result.requested_model == (
        "arsoban/ocd-therapist-27b-v0.3:featherless-ai"
    )


def test_missing_hugging_face_token_fails_before_network() -> None:
    session = FakeSession(FakeResponse(payload=_success_payload()))
    result = HuggingFaceInferenceClient(
        "arsoban/ocd-therapist-27b-v0.3",
        None,
        session=session,
    ).complete(_request())

    assert result.success is False
    assert result.failure_category == "configuration"
    assert result.attempts == 0
    assert session.calls == []
