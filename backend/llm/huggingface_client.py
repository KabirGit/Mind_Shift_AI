from __future__ import annotations

import json
import logging
import socket
import urllib.error
import urllib.request
from typing import Any
import requests

logger = logging.getLogger(__name__)


class HuggingFaceInferenceClient:
    """
    Hugging Face Inference API client (local process).
    Contract:
      generate(prompt: str) -> str
    """

    def __init__(
        self,
        model_name: str,
        api_token: str | None,
        max_new_tokens: int = 220,
        timeout_s: int = 30,
        temperature: float = 0.2,
    ) -> None:
        self.model_name = model_name
        self.api_token = api_token
        self.max_new_tokens = max_new_tokens
        self.timeout_s = timeout_s
        self.temperature = temperature

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    """def generate(self, prompt: str) -> str:
        url =f"https://api-inference.huggingface.co/models/{self.model_name}"
        payload: dict[str, Any] = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                # Keep response consistent across models.
                "return_full_text": False,
            },
        }

        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8")
            data = json.loads(body)
            return self._extract_text(data)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            logger.exception("HF Inference API call failed, using fallback: %s", exc)
            return (
                "I’m here with you. I may not have immediate access to generate a full reply right now, "
                "but it helps to notice what you’re feeling and what you need in this moment."
            )
        except Exception as exc:
            logger.exception("HF Inference API parsing failed, using fallback: %s", exc)
            return (
                "I’m here with you. Tell me a bit more about what triggered this feeling, and we can explore it gently."
            )"""

    
    def generate(self, prompt: str) -> str:
        url = "https://api.mistral.ai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mistral-small",
            "messages": [
                {"role": "system", "content": "You are an empathetic journaling assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens
        }

        try:
            res = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
            data = res.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.exception("Mistral API failed: %s", e)
            return "I'm here with you. Tell me more about what you're feeling."

    def _extract_text(self, data: Any) -> str:
        # HF often returns either:
        # - list[{"generated_text": "..."}]
        # - dict{"error": "..."} on failures
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"HF error: {data['error']}")

        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                # Most generation models
                text = first.get("generated_text") or first.get("summary_text") or ""
                return str(text).strip() or self._fallback_from_data(first)
        return self._fallback_from_data(data)

    def _fallback_from_data(self, data: Any) -> str:
        # Try to use any text-like value.
        if isinstance(data, dict):
            for key in ("generated_text", "summary_text", "text"):
                if key in data and data[key]:
                    return str(data[key]).strip()
        return (
            "I hear you. Take a slow breath, and if you can, tell me what part feels the heaviest right now."
        )

