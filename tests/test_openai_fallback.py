"""OpenAI provider and OpenAI → Gemini fallback (all mocked)."""

import json
import random

import pytest

from app.config import ConfigError, load_settings
from app.llm import (
    FallbackClient,
    InvalidResponseError,
    LLMClient,
    LLMRequest,
    ModelUnavailableError,
    PermanentLLMError,
    QuotaExhaustedError,
    RateLimitError,
    TransientLLMError,
    build_llm,
)
from app.llm.mock import MockProvider, ScriptedProvider
from app.llm.openai import OpenAIProvider, classify_error, extract_text

REQ = LLMRequest(task="analysis", system="s", prompt="Return JSON")


def client(provider, max_calls=10, max_retries=2):
    return LLMClient(provider, max_calls=max_calls, min_interval=0, max_retries=max_retries, backoff_base=1, backoff_max=2,
                     max_retry_after=90, sleep=lambda s: None, rng=random.Random(0))


def named(provider, name):
    provider.name = name
    return provider


# ------------------------------------------------------------ OpenAI provider


def test_openai_error_classification():
    assert isinstance(classify_error(429, {"error": {"code": "insufficient_quota", "message": "You exceeded your current quota"}}), QuotaExhaustedError)
    rl = classify_error(429, {"error": {"code": "rate_limit_exceeded"}}, {"retry-after-ms": "1500"})
    assert isinstance(rl, RateLimitError) and rl.retry_after == 1.5
    assert classify_error(429, {}, {"retry-after": "7"}).retry_after == 7
    assert isinstance(classify_error(404, {"error": {"code": "model_not_found"}}), ModelUnavailableError)
    assert isinstance(classify_error(401, {"error": {"message": "Incorrect API key"}}), PermanentLLMError)
    assert isinstance(classify_error(400, {"error": {"message": "bad param"}}), PermanentLLMError)
    assert isinstance(classify_error(503, "not json"), TransientLLMError)


def test_openai_response_extraction():
    ok = {"choices": [{"finish_reason": "stop", "message": {"content": '{"a": 1}'}}], "usage": {"prompt_tokens": 100, "completion_tokens": 40}}
    assert extract_text(ok) == ('{"a": 1}', {"input_tokens": 100, "output_tokens": 40})
    for bad in ({"choices": []}, {"choices": [{"finish_reason": "length", "message": {"content": "{"}}]},
                {"choices": [{"finish_reason": "stop", "message": {"refusal": "no", "content": None}}]},
                {"choices": [{"finish_reason": "content_filter", "message": {"content": ""}}]}):
        with pytest.raises(InvalidResponseError):
            extract_text(bad)


def test_openai_request_shape_and_secret_handling():
    seen = {}

    class Resp:
        status_code = 200
        headers = {}

        def json(self):
            return {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}

    class Session:
        def post(self, url, headers, json, timeout):
            seen.update(url=url, headers=headers, body=json)
            return Resp()

    p = OpenAIProvider("sk-secret-123", "gpt-test-mini", "https://api.example/v1", 4000, 30, reasoning_effort="low", session=Session())
    p.generate_raw(REQ)
    assert seen["url"] == "https://api.example/v1/chat/completions" and "sk-secret" not in seen["url"]
    assert seen["headers"]["Authorization"] == "Bearer sk-secret-123"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["body"]["max_completion_tokens"] == 4000 and seen["body"]["reasoning_effort"] == "low"
    assert "temperature" not in seen["body"]  # reasoning models reject custom temperature
    assert p.usage == {"requests": 1, "input_tokens": 5, "output_tokens": 3}
    with pytest.raises(PermanentLLMError):
        OpenAIProvider("", "m", "https://x", 1, 1)


# ------------------------------------------------------------ fallback chain


def test_out_of_credit_switches_to_gemini_for_rest_of_run():
    openai = named(ScriptedProvider([QuotaExhaustedError("insufficient_quota"), '{"never": true}']), "openai")
    gemini = named(ScriptedProvider(['{"from": "gemini"}', '{"again": 1}']), "gemini")
    fc = FallbackClient([client(openai), client(gemini)])
    assert fc.generate_json(REQ) == {"from": "gemini"}
    assert fc.generate_json(REQ) == {"again": 1}
    assert len(openai.requests) == 1  # never asked again after running out of credit
    assert fc.fallback_used and "openai" in fc.disabled and not fc.quota_exhausted


def test_bad_key_or_model_falls_back():
    for err in (PermanentLLMError("401"), ModelUnavailableError("404")):
        fc = FallbackClient([client(named(ScriptedProvider([err]), "openai")), client(named(ScriptedProvider(['{"ok": 1}']), "gemini"))])
        assert fc.generate_json(REQ) == {"ok": 1}


def test_outage_after_retries_falls_back():
    openai = named(ScriptedProvider([TransientLLMError("503")] * 10), "openai")
    fc = FallbackClient([client(openai, max_retries=2), client(named(ScriptedProvider(['{"ok": 1}']), "gemini"))])
    assert fc.generate_json(REQ) == {"ok": 1} and len(openai.requests) == 3


def test_invalid_output_tries_fallback_without_disabling_primary():
    openai = named(ScriptedProvider(["junk", "junk", '{"primary": 1}']), "openai")
    gemini = named(ScriptedProvider(['{"fallback": 1}']), "gemini")
    fc = FallbackClient([client(openai), client(gemini)])
    assert fc.generate_json(REQ) == {"fallback": 1}
    assert fc.generate_json(REQ) == {"primary": 1}  # primary still in use
    assert fc.disabled == {}


def test_all_providers_exhausted_is_quota_exhaustion():
    fc = FallbackClient([client(named(ScriptedProvider([QuotaExhaustedError("credit")]), "openai")),
                         client(named(ScriptedProvider([QuotaExhaustedError("daily")]), "gemini"))])
    with pytest.raises(QuotaExhaustedError):
        fc.generate_json(REQ)
    assert fc.quota_exhausted
    with pytest.raises(QuotaExhaustedError):
        fc.generate_json(REQ)


def test_usage_reported_per_provider():
    fc = FallbackClient([client(named(MockProvider(), "openai")), client(named(MockProvider(), "gemini"))])
    fc.generate_json(REQ)
    assert set(fc.usage()) == {"openai", "gemini"} and fc.calls_made == 1


# ------------------------------------------------------------ configuration


def test_chain_built_from_config_and_missing_keys_skipped():
    both = build_llm(load_settings(env={"OPENAI_API_KEY": "sk-x", "GEMINI_API_KEY": "g-x"}))
    assert isinstance(both, FallbackClient) and both.provider_names == ["openai", "gemini"]
    only_gemini = build_llm(load_settings(env={"GEMINI_API_KEY": "g-x"}))
    assert isinstance(only_gemini, LLMClient) and only_gemini.provider_names == ["gemini"]
    with pytest.raises(PermanentLLMError):
        build_llm(load_settings(env={}))
    no_fallback = build_llm(load_settings(env={"OPENAI_API_KEY": "sk-x", "GEMINI_API_KEY": "g", "LLM_FALLBACK": "none"}))
    assert no_fallback.provider_names == ["openai"]
    forced = build_llm(load_settings(env={"OPENAI_API_KEY": "sk-x", "GEMINI_API_KEY": "g"}), "gemini")
    assert forced.provider_names == ["gemini"]


def test_env_overrides_and_validation():
    s = load_settings(env={"OPENAI_MODEL": "gpt-other", "LLM_FALLBACK": "gemini,ollama"})
    assert s.llm.openai_model == "gpt-other" and s.llm.fallback_providers == ("gemini", "ollama")
    with pytest.raises(ConfigError):
        load_settings(env={"LLM_FALLBACK": "paidthing"})
    assert "sk-topsecret" not in repr(load_settings(env={"OPENAI_API_KEY": "sk-topsecret"}))
    assert load_settings(env={"OPENAI_API_KEY": "sk-topsecret"}).secrets.redact("key sk-topsecret") == "key ***"
