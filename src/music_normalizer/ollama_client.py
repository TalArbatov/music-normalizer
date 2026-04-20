"""Thin HTTP client for the Ollama /api/chat endpoint.

We deliberately avoid the ``ollama`` Python package so that the only runtime
dependency on the LLM side is ``httpx``. Ollama's built-in JSON format mode is
used to coerce the model to structured output.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from .config import OllamaConfig
from .utils import get_logger

log = get_logger(__name__)


class OllamaError(RuntimeError):
    """Raised when Ollama responds with an error or unparseable output."""


class OllamaClient:
    def __init__(self, config: OllamaConfig):
        self.config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.request_timeout_s),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        """Call /api/chat with format=json and return the parsed object.

        :raises OllamaError: on HTTP error, invalid JSON, or missing 'message.content'.
        """
        body = {
            "model": self.config.model,
            "format": "json",
            "stream": False,
            "options": {"temperature": self.config.temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            resp = self._client.post("/api/chat", json=body)
        except httpx.HTTPError as e:
            raise OllamaError(f"HTTP error calling Ollama: {e}") from e

        if resp.status_code != 200:
            raise OllamaError(
                f"Ollama returned {resp.status_code}: {resp.text[:500]}"
            )
        data = resp.json()
        content = data.get("message", {}).get("content")
        if not content:
            raise OllamaError(f"Ollama response missing message.content: {data}")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            log.error("Ollama returned non-JSON content:\n%s", content)
            raise OllamaError(f"Ollama returned invalid JSON: {e}") from e

    def health_check(self) -> bool:
        try:
            resp = self._client.get("/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
