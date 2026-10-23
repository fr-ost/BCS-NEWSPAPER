"""OpenAI provider (Chat Completions REST API, JSON mode, no SDK).

The API key is sent in the Authorization header, never in the URL. OpenAI is
pay-as-you-go: when the prepaid credit runs out it returns HTTP 429 with
code ``insufficient_quota``, which is mapped to QuotaExhaustedError so the
run falls back to the next provider (Gemini) instead of failing.
"""

from __future__ import annotations

import re

import requests

from .base import (
    InvalidResponseError,
    LLMProvider,
    LLMRequest,
    ModelUnavailableError,
    PermanentLLMError,
    QuotaExhaustedError,
    RateLimitError,
    TransientLLMError,
)


def _retry_after(headers) -> float | None:
    for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        value = headers.get(name) if headers else None
        try:
            if value is not None:
                return float(value) * scale
        except (TypeError, ValueError):
            continue
    return None


def classify_error(status: int, body: object, headers=None) -> Exception:
    error = body.get("error", {}) if isinstance(body, dict) else {}
    if not isinstance(error, dict):
        error = {}
    code = str(error.get("code") or "")
    etype = str(error.get("type") or "")
    message = str(error.get("message", ""))[:300]
    if status == 429:
        if code == "insufficient_quota" or etype == "insufficient_quota":
            return QuotaExhaustedError(f"OpenAI credit exhausted: {message}")
        return RateLimitError(f"OpenAI rate limited: {message}", retry_after=_retry_after(headers))
    if status == 404 or code == "model_not_found":
        return ModelUnavailableError(f"OpenAI model unavailable: {message}")
    if status in (401, 403):
        return PermanentLLMError(f"OpenAI rejected credentials ({status}): {message}")
    if status in (408, 409) or status >= 500:
        return TransientLLMError(f"OpenAI temporary error ({status}): {message}")
    return PermanentLLMError(f"OpenAI request failed ({status}): {message}")


def extract_text(data: object) -> tuple[str, dict]:
    """Return (text, usage) from a Chat Completions response."""
    if not isinstance(data, dict):
        raise InvalidResponseError("OpenAI response is not an object")
    choices = data.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise InvalidResponseError("OpenAI returned no choices")
    choice = choices[0]
    message = choice.get("message") or {}
    if message.get("refusal"):
        raise InvalidResponseError("OpenAI refused the request")
    finish = choice.get("finish_reason")
    if finish == "length":
        raise InvalidResponseError("OpenAI output truncated (length)")
    if finish not in (None, "stop"):
        raise InvalidResponseError(f"OpenAI stopped early: {finish}")
    text = message.get("content")
    if not isinstance(text, str) or not text.strip():
        raise InvalidResponseError("OpenAI returned empty text")
    usage = data.get("usage") or {}
    return text, {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),  # includes reasoning tokens
    }


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: str,
        max_output_tokens: int,
        timeout: float,
        reasoning_effort: str = "",
        session: requests.Session | None = None,
    ):
        if not api_key:
            raise PermanentLLMError("OPENAI_API_KEY is not set")
        if not re.fullmatch(r"[A-Za-z0-9._:-]+", model or ""):
            raise PermanentLLMError(f"Invalid model id {model!r}")
        self._api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.session = session or requests.Session()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def check_model(self) -> None:
        try:
            resp = self.session.get(f"{self.api_base}/models/{self.model}", headers=self._headers(), timeout=30)
        except requests.RequestException as exc:
            raise TransientLLMError(f"Network error checking model: {type(exc).__name__}") from exc
        if resp.status_code != 200:
            raise classify_error(resp.status_code, _safe_json(resp), resp.headers)

    def generate_raw(self, request: LLMRequest) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            "response_format": {"type": "json_object"},
            # Reasoning tokens count against this limit and are billed as output.
            "max_completion_tokens": request.max_output_tokens or self.max_output_tokens,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        try:
            resp = self.session.post(f"{self.api_base}/chat/completions", headers=self._headers(), json=body, timeout=self.timeout)
        except requests.Timeout as exc:
            raise TransientLLMError("OpenAI request timed out") from exc
        except requests.RequestException as exc:
            raise TransientLLMError(f"OpenAI network error: {type(exc).__name__}") from exc
        data = _safe_json(resp)
        if resp.status_code != 200:
            raise classify_error(resp.status_code, data, resp.headers)
        text, usage = extract_text(data)
        self.record_usage(usage["input_tokens"], usage["output_tokens"])
        return text


def _safe_json(resp: requests.Response) -> object:
    try:
        return resp.json()
    except ValueError:
        return {}
