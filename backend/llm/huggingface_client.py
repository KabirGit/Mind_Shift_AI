"""Hugging Face Inference Providers chat-completions boundary."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from backend.llm.mistral_client import MistralInferenceClient


class HuggingFaceInferenceClient(MistralInferenceClient):
    """Typed client for Hugging Face's OpenAI-compatible inference router.

    The router selects a concrete provider from a model suffix. MindShift pins
    the requested OCD model to Featherless AI because it is the provider listed
    for that model on the Hub. All status handling, retry bounds, response-shape
    validation, and usage capture are inherited from the common chat boundary.
    """

    endpoint = "https://router.huggingface.co/v1/chat/completions"
    provider_name = "Hugging Face Inference Providers"

    def __init__(
        self,
        model_name: str,
        api_token: str | None,
        max_new_tokens: int = 220,
        timeout_s: float | None = None,
        temperature: float = 0.2,
        *,
        provider: str | None = "featherless-ai",
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 30.0,
        max_attempts: int = 2,
        retry_backoff_s: float = 0.25,
        session: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        normalized_model = model_name.strip()
        normalized_provider = provider.strip() if provider else ""
        routed_model = (
            f"{normalized_model}:{normalized_provider}"
            if normalized_provider and ":" not in normalized_model.rsplit("/", 1)[-1]
            else normalized_model
        )
        self.hub_model_name = normalized_model
        self.inference_provider = normalized_provider or None
        super().__init__(
            model_name=routed_model,
            api_token=api_token,
            max_new_tokens=max_new_tokens,
            timeout_s=timeout_s,
            temperature=temperature,
            connect_timeout_s=connect_timeout_s,
            read_timeout_s=read_timeout_s,
            max_attempts=max_attempts,
            retry_backoff_s=retry_backoff_s,
            session=session,
            sleep_fn=sleep_fn,
        )


__all__ = ["HuggingFaceInferenceClient"]
