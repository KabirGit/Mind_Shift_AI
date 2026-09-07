from __future__ import annotations

from typing import Any

import requests

from backend.llm.mistral_client import MistralInferenceClient
from backend.llm.models import LLMRequest


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError("provider request failed")
            error.response = self  # type: ignore[assignment]
            raise error

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeSession:
    def __init__(self, results: list[Any]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
        self.calls.append({"args": args, "kwargs": kwargs})
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _request(*, output_format: str = "text") -> LLMRequest:
    return LLMRequest(
        purpose="reflection",
        system_prompt="System policy",
        user_prompt="Untrusted input",
        prompt_version="test-v1",
        response_format=output_format,  # type: ignore[arg-type]
        temperature=0.0,
        max_tokens=123,
    )


def _success_payload() -> dict[str, Any]:
    return {
        "model": "mistral-small-latest",
        "choices": [
            {
                "message": {"content": "A useful response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
        },
    }


def test_complete_returns_text_metadata_usage_and_separate_timeouts() -> None:
    session = FakeSession([FakeResponse(payload=_success_payload())])
    client = MistralInferenceClient(
        "requested-model",
        "secret",
        connect_timeout_s=2,
        read_timeout_s=9,
        session=session,
    )

    result = client.complete(_request(output_format="json_object"))

    assert result.success is True
    assert result.text == "A useful response"
    assert result.actual_model == "mistral-small-latest"
    assert result.usage.total_tokens == 14
    assert result.attempts == 1
    sent = session.calls[0]["kwargs"]
    assert sent["timeout"] == (2.0, 9.0)
    assert sent["json"]["response_format"] == {"type": "json_object"}
    assert sent["json"]["temperature"] == 0.0
    assert sent["json"]["messages"][0]["role"] == "system"


def test_missing_key_fails_before_network_call() -> None:
    session = FakeSession([])
    result = MistralInferenceClient("model", None, session=session).complete(_request())

    assert result.success is False
    assert result.attempts == 0
    assert result.failure_category == "configuration"
    assert session.calls == []


def test_timeout_retries_once_then_recovers() -> None:
    session = FakeSession([requests.Timeout(), FakeResponse(payload=_success_payload())])
    sleeps: list[float] = []
    client = MistralInferenceClient(
        "model",
        "secret",
        session=session,
        retry_backoff_s=0.1,
        sleep_fn=sleeps.append,
    )

    result = client.complete(_request())

    assert result.success is True
    assert result.attempts == 2
    assert result.retry_count == 1
    assert len(session.calls) == 2
    assert sleeps == [0.1]


def test_retryable_http_status_retries_once() -> None:
    session = FakeSession(
        [
            FakeResponse(status_code=429, headers={"Retry-After": "0"}),
            FakeResponse(status_code=503),
        ]
    )
    result = MistralInferenceClient("model", "secret", session=session).complete(
        _request()
    )

    assert result.success is False
    assert result.attempts == 2
    assert result.retry_count == 1
    assert result.failure_category == "provider_unavailable"
    assert len(session.calls) == 2


def test_permanent_http_failure_is_not_retried() -> None:
    session = FakeSession([FakeResponse(status_code=401)])
    result = MistralInferenceClient("model", "secret", session=session).complete(
        _request()
    )

    assert result.success is False
    assert result.attempts == 1
    assert result.retry_count == 0
    assert result.failure_category == "authentication"
    assert len(session.calls) == 1


def test_invalid_provider_shape_is_a_failure() -> None:
    session = FakeSession([FakeResponse(payload={"choices": []})])
    result = MistralInferenceClient("model", "secret", session=session).complete(
        _request()
    )

    assert result.success is False
    assert result.failure_category == "invalid_response"
    assert result.text == ""


def test_max_attempts_is_unconditionally_capped_at_two() -> None:
    session = FakeSession([requests.ConnectionError(), requests.ConnectionError()])
    result = MistralInferenceClient(
        "model", "secret", session=session, max_attempts=99, retry_backoff_s=0
    ).complete(_request())

    assert result.attempts == 2
    assert result.retry_count == 1
    assert result.failure_category == "network"


def test_legacy_generate_returns_generic_fallback_on_failure() -> None:
    client = MistralInferenceClient("model", None)

    assert client.generate("hello") == (
        "I'm here with you. Tell me more about what you're feeling."
    )
