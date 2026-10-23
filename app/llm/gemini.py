"""Gemini Developer API provider (REST, no SDK).

Uses ``generateContent`` with JSON output. The API key is sent in the
``x-goog-api-key`` header, never in the URL, so it cannot leak into logged
URLs or exception messages.

This module never enables billing or switches models. If the configured
model has no free-tier quota, Google returns 429 and the run degrades to a
partial edition instead of falling back to anything paid.
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

_DELAY = re.compile(r"^(\d+(?:\.\d+)?)s$")


def _retry_delay(details: list) -> float | None:
    for d in details:
        if isinstance(d, dict) and d.get("@type", "").endswith("RetryInfo"):
            m = _DELAY.match(str(d.get("retryDelay", "")))
            if m:
                return float(m.group(1))
    return None


def _is_daily_or_zero_quota(error: dict) -> bool:
    message = str(error.get("message", ""))
    if "limit: 0" in message:  # model has no free-tier quota for this project
        return True
    for d in error.get("details") or []:
        if not isinstance(d, dict) or not d.get("@type", "").endswith("QuotaFailure"):
            continue
        for v in d.get("violations") or []:
            quota_id = str(v.get("quotaId", "")) + str(v.get("quotaMetric", ""))
            if "PerDay" in quota_id or "per_day" in quota_id.lower():
                return True
    return False


def classify_error(status: int, body: object) -> Exception:
    error = body.get("error", {}) if isinstance(body, dict) else {}
    if not isinstance(error, dict):
        error = {}
    message = str(error.get("message", ""))[:300]
    api_status = str(error.get("status", ""))
    if status == 429 or api_status == "RESOURCE_EXHAUSTED":
        if _is_daily_or_zero_quota(error):
            return QuotaExhaustedError(f"Gemini quota exhausted: {message}")
        return RateLimitError(f"Gemini rate limited: {message}", retry_after=_retry_delay(error.get("details") or []))
    if status == 404 or api_status == "NOT_FOUND":
        return ModelUnavailableError(f"Gemini model unavailable: {message}")
    if status in (401, 403) or api_status in ("PERMISSION_DENIED", "UNAUTHENTICATED"):
        return PermanentLLMError(f"Gemini rejected credentials ({status}): {message}")
    if status in (500, 502, 503, 504) or api_status in ("UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"):
        return TransientLLMError(f"Gemini temporary error ({status}): {message}")
    return PermanentLLMError(f"Gemini request failed ({status}): {message}")


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        api_base: str,
        temperature: float,
        max_output_tokens: int,
        timeout: float,
        session: requests.Session | None = None,
    ):
        if not api_key:
            raise PermanentLLMError("GEMINI_API_KEY is not set")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", model or ""):
            raise PermanentLLMError(f"Invalid model id {model!r}")
        self._api_key = api_key
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.session = session or requests.Session()

    def _headers(self) -> dict:
        return {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}

    def check_model(self) -> None:
        try:
            resp = self.session.get(f"{self.api_base}/models/{self.model}", headers=self._headers(), timeout=30)
        except requests.RequestException as exc:
            raise TransientLLMError(f"Network error checking model: {type(exc).__name__}") from exc
        if resp.status_code != 200:
            raise classify_error(resp.status_code, _safe_json(resp))

    def generate_raw(self, request: LLMRequest) -> str:
        body = {
            "systemInstruction": {"parts": [{"text": request.system}]},
            "contents": [{"role": "user", "parts": [{"text": request.prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": request.max_output_tokens or self.max_output_tokens,
                "responseMimeType": "application/json",
            },
        }
        try:
            resp = self.session.post(
                f"{self.api_base}/models/{self.model}:generateContent",
                headers=self._headers(),
                json=body,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise TransientLLMError("Gemini request timed out") from exc
        except requests.RequestException as exc:
            raise TransientLLMError(f"Gemini network error: {type(exc).__name__}") from exc

        data = _safe_json(resp)
        if resp.status_code != 200:
            raise classify_error(resp.status_code, data)
        return extract_text(data)


def _safe_json(resp: requests.Response) -> object:
    try:
        return resp.json()
    except ValueError:
        return {}


def extract_text(data: object) -> str:
    if not isinstance(data, dict):
        raise InvalidResponseError("Gemini response is not an object")
    feedback = data.get("promptFeedback") or {}
    if isinstance(feedback, dict) and feedback.get("blockReason"):
        raise InvalidResponseError(f"Prompt blocked: {feedback.get('blockReason')}")
    candidates = data.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        raise InvalidResponseError("Gemini returned no candidates")
    cand = candidates[0]
    finish = cand.get("finishReason")
    parts = (cand.get("content") or {}).get("parts") or []
    # Skip "thought" parts some models return; keep only answer text.
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
    if finish == "MAX_TOKENS":
        raise InvalidResponseError("Gemini output truncated (MAX_TOKENS)")
    if finish not in (None, "STOP", "FINISH_REASON_UNSPECIFIED"):
        raise InvalidResponseError(f"Gemini stopped early: {finish}")
    if not text.strip():
        raise InvalidResponseError("Gemini returned empty text")
    return text
