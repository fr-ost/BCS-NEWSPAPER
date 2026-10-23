"""Model error handling, quota exhaustion and retry limits (all mocked)."""

import random

import pytest

from app.llm import (
    BudgetExceededError,
    InvalidResponseError,
    LLMClient,
    LLMRequest,
    ModelUnavailableError,
    PermanentLLMError,
    QuotaExhaustedError,
    RateLimitError,
    TransientLLMError,
)
from app.llm.gemini import GeminiProvider, classify_error, extract_text
from app.llm.mock import ScriptedProvider

REQ = LLMRequest(task="analysis", system="s", prompt="p")


def client(provider, max_calls=10, max_retries=3, sleeps=None, max_retry_after=90):
    sleeps = sleeps if sleeps is not None else []
    return LLMClient(provider, max_calls=max_calls, min_interval=0, max_retries=max_retries, backoff_base=4, backoff_max=60,
                     max_retry_after=max_retry_after, sleep=sleeps.append, rng=random.Random(1))


def quota_body(quota_id, message="Quota exceeded", delay=None):
    details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id}]}]
    if delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay})
    return {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": message, "details": details}}


def test_gemini_error_classification():
    assert isinstance(classify_error(429, quota_body("GenerateRequestsPerDayPerProjectPerModel-FreeTier")), QuotaExhaustedError)
    rl = classify_error(429, quota_body("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", delay="17s"))
    assert isinstance(rl, RateLimitError) and rl.retry_after == 17
    assert isinstance(classify_error(429, quota_body("X", message="limit: 0, model: m")), QuotaExhaustedError)
    assert isinstance(classify_error(404, {"error": {"status": "NOT_FOUND", "message": "models/x not found"}}), ModelUnavailableError)
    assert isinstance(classify_error(403, {}), PermanentLLMError)
    assert isinstance(classify_error(400, {"error": {"message": "API key not valid"}}), PermanentLLMError)
    assert isinstance(classify_error(503, {}), TransientLLMError)
    assert isinstance(classify_error(500, "not json"), TransientLLMError)


def test_gemini_response_extraction():
    ok = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "thinking", "thought": True}, {"text": '{"a":1}'}]}}]}
    assert extract_text(ok) == '{"a":1}'
    for bad in ({"promptFeedback": {"blockReason": "SAFETY"}}, {"candidates": []},
                {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "{"}]}}]},
                {"candidates": [{"finishReason": "SAFETY", "content": {"parts": []}}]}):
        with pytest.raises(InvalidResponseError):
            extract_text(bad)


def test_gemini_key_sent_in_header_not_url():
    seen = {}

    class Resp:
        status_code = 200

        def json(self):
            return {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "{}"}]}}]}

    class Session:
        def post(self, url, headers, json, timeout):
            seen.update(url=url, headers=headers, body=json)
            return Resp()

    p = GeminiProvider("secret-key-xyz", "gemini-test", "https://api.example", 0.2, 100, 10, session=Session())
    p.generate_raw(REQ)
    assert "secret-key-xyz" not in seen["url"]
    assert seen["headers"]["x-goog-api-key"] == "secret-key-xyz"
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"
    with pytest.raises(PermanentLLMError):
        GeminiProvider("", "m", "https://x", 0, 1, 1)
    with pytest.raises(PermanentLLMError):
        GeminiProvider("k", "../evil", "https://x", 0, 1, 1)


def test_daily_quota_stops_immediately_and_for_rest_of_run():
    prov = ScriptedProvider([QuotaExhaustedError("daily")])
    c = client(prov)
    with pytest.raises(QuotaExhaustedError):
        c.generate_json(REQ)
    assert c.calls_made == 1 and c.quota_exhausted
    with pytest.raises(QuotaExhaustedError):
        c.generate_json(REQ)
    assert len(prov.requests) == 1  # no further calls once exhausted


def test_rate_limit_retries_with_server_delay_then_succeeds():
    sleeps = []
    c = client(ScriptedProvider([RateLimitError("rpm", retry_after=10), '{"ok": true}']), sleeps=sleeps)
    assert c.generate_json(REQ) == {"ok": True}
    assert len(sleeps) == 1 and 10 <= sleeps[0] <= 11


def test_persistent_rate_limit_becomes_quota_exhaustion():
    c = client(ScriptedProvider([RateLimitError("rpm")] * 10), max_retries=2)
    with pytest.raises(QuotaExhaustedError):
        c.generate_json(REQ)
    assert c.calls_made == 3 and c.quota_exhausted


def test_huge_retry_after_is_not_waited():
    sleeps = []
    c = client(ScriptedProvider([RateLimitError("rpm", retry_after=3600)]), sleeps=sleeps)
    with pytest.raises(QuotaExhaustedError):
        c.generate_json(REQ)
    assert sleeps == []


def test_transient_errors_have_bounded_retries_with_backoff():
    sleeps = []
    prov = ScriptedProvider([TransientLLMError("503")] * 10)
    c = client(prov, max_retries=3, sleeps=sleeps)
    with pytest.raises(TransientLLMError):
        c.generate_json(REQ)
    assert len(prov.requests) == 4
    assert all(0 < s <= 60 for s in sleeps) and len(sleeps) == 3


def test_permanent_errors_are_never_retried():
    prov = ScriptedProvider([ModelUnavailableError("404"), '{"never": 1}'])
    with pytest.raises(ModelUnavailableError):
        client(prov).generate_json(REQ)
    assert len(prov.requests) == 1


def test_invalid_json_retried_once_only():
    prov = ScriptedProvider(["not json", "still not json", '{"x": 1}'])
    with pytest.raises(InvalidResponseError):
        client(prov).generate_json(REQ)
    assert len(prov.requests) == 2
    assert client(ScriptedProvider(["oops", '{"x": 1}'])).generate_json(REQ) == {"x": 1}


def test_per_run_budget_caps_all_calls():
    prov = ScriptedProvider(['{"a":1}'] * 10)
    c = client(prov, max_calls=2)
    c.generate_json(REQ)
    c.generate_json(REQ)
    with pytest.raises(BudgetExceededError):
        c.generate_json(REQ)
    assert len(prov.requests) == 2 and c.quota_exhausted
