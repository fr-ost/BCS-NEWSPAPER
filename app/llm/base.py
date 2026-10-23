"""Provider-neutral LLM interface, error taxonomy and retry/budget control.

Adding a provider (e.g. a local model) means implementing
:class:`LLMProvider.generate_raw` and mapping its failures onto the error
classes below. Retries, spacing and the per-run call budget live in
:class:`LLMClient` and are shared by every provider.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Base class. Messages must never contain credentials."""


class QuotaExhaustedError(LLMError):
    """Daily/free-tier quota is used up. Stop calling the model this run."""


class BudgetExceededError(QuotaExhaustedError):
    """Our own per-run call budget is used up (protects the free quota)."""


class RateLimitError(LLMError):
    """Short-term rate limit (per minute). Retry after ``retry_after`` seconds."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class TransientLLMError(LLMError):
    """Network failure or 5xx. Worth a bounded retry."""


class PermanentLLMError(LLMError):
    """Bad credentials, bad request, etc. Never retried."""


class ModelUnavailableError(PermanentLLMError):
    """The configured model id does not exist or is not available to this key."""


class InvalidResponseError(LLMError):
    """The model replied, but not with usable JSON (blocked, truncated, malformed)."""


@dataclass
class LLMRequest:
    task: str  # "screening" | "analysis" | "mcq" | "ping"
    system: str
    prompt: str
    # Structured copy of the input, used only by offline/mock providers.
    payload: dict = field(default_factory=dict)
    max_output_tokens: int | None = None


class LLMProvider(ABC):
    name = "base"

    @abstractmethod
    def generate_raw(self, request: LLMRequest) -> str:
        """Return the model's raw text reply or raise an :class:`LLMError`."""

    def check_model(self) -> None:
        """Optional preflight check that does not consume generation quota."""


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def parse_json_text(text: str) -> object:
    """Parse model output as JSON, tolerating code fences and surrounding prose."""
    if not isinstance(text, str) or not text.strip():
        raise InvalidResponseError("Empty model response")
    cleaned = _FENCE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(open_ch), cleaned.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise InvalidResponseError("Model response is not valid JSON")


class LLMClient:
    """Wraps a provider with budget, spacing, bounded retries and backoff+jitter."""

    def __init__(
        self,
        provider: LLMProvider,
        max_calls: int,
        min_interval: float,
        max_retries: int,
        backoff_base: float,
        backoff_max: float,
        max_retry_after: float,
        sleep=time.sleep,
        clock=time.monotonic,
        rng: random.Random | None = None,
    ):
        self.provider = provider
        self.max_calls = max_calls
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.max_retry_after = max_retry_after
        self._sleep = sleep
        self._clock = clock
        self._rng = rng or random.Random()
        self._last_call: float | None = None
        self.calls_made = 0
        self.quota_exhausted = False

    @property
    def remaining_calls(self) -> int:
        return max(0, self.max_calls - self.calls_made)

    def _wait_for_spacing(self) -> None:
        if self._last_call is None:
            return
        elapsed = self._clock() - self._last_call
        if elapsed < self.min_interval:
            self._sleep(self.min_interval - elapsed)

    def _backoff(self, attempt: int) -> float:
        delay = min(self.backoff_max, self.backoff_base * (2**attempt))
        return delay * (0.5 + self._rng.random() / 2)  # "equal jitter"

    def generate_json(self, request: LLMRequest) -> object:
        if self.quota_exhausted:
            raise QuotaExhaustedError("Model quota already exhausted in this run")
        invalid_retries = 1
        attempt = 0
        while True:
            if self.calls_made >= self.max_calls:
                self.quota_exhausted = True
                raise BudgetExceededError(f"Per-run call budget of {self.max_calls} reached")
            self._wait_for_spacing()
            self.calls_made += 1
            self._last_call = self._clock()
            try:
                return parse_json_text(self.provider.generate_raw(request))
            except QuotaExhaustedError:
                self.quota_exhausted = True
                raise
            except PermanentLLMError:
                raise
            except InvalidResponseError as exc:
                if invalid_retries <= 0:
                    raise
                invalid_retries -= 1
                log.warning("Invalid %s response (%s); retrying once", request.task, exc)
                continue
            except RateLimitError as exc:
                if attempt >= self.max_retries:
                    # Persistent rate limiting: stop using the model for this run.
                    self.quota_exhausted = True
                    raise QuotaExhaustedError(f"Rate limited after {attempt + 1} attempts") from exc
                wait = exc.retry_after if exc.retry_after is not None else self._backoff(attempt)
                if wait > self.max_retry_after:
                    self.quota_exhausted = True
                    raise QuotaExhaustedError(f"Server asked to wait {wait:.0f}s; treating as exhausted") from exc
                log.warning("Rate limited on %s; waiting %.1fs", request.task, wait)
                self._sleep(wait + self._rng.uniform(0, 1))
            except TransientLLMError as exc:
                if attempt >= self.max_retries:
                    raise
                wait = self._backoff(attempt)
                log.warning("Transient error on %s (%s); retrying in %.1fs", request.task, exc, wait)
                self._sleep(wait)
            attempt += 1
