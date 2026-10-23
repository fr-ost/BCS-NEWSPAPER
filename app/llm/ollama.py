"""Local model provider via Ollama's HTTP API (experimental).

Not used by default and not exercised against a real Ollama server in this
repository's tests. It exists to show that a local model plugs into the same
interface: set ``LLM_PROVIDER=ollama`` and ``OLLAMA_MODEL`` and run the
pipeline on a machine where Ollama is running. GitHub-hosted runners cannot
reach a model on your own computer.
"""

from __future__ import annotations

import requests

from .base import InvalidResponseError, LLMProvider, LLMRequest, ModelUnavailableError, PermanentLLMError, TransientLLMError


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str, base_url: str, temperature: float, max_output_tokens: int, timeout: float, session: requests.Session | None = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.session = session or requests.Session()

    def generate_raw(self, request: LLMRequest) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": request.max_output_tokens or self.max_output_tokens},
        }
        try:
            resp = self.session.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout)
        except requests.RequestException as exc:
            raise TransientLLMError(f"Ollama unreachable: {type(exc).__name__}") from exc
        if resp.status_code == 404:
            raise ModelUnavailableError(f"Ollama model {self.model!r} not found")
        if resp.status_code >= 500:
            raise TransientLLMError(f"Ollama error {resp.status_code}")
        if resp.status_code != 200:
            raise PermanentLLMError(f"Ollama request failed ({resp.status_code})")
        try:
            content = resp.json()["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise InvalidResponseError("Unexpected Ollama response shape") from exc
        return content
