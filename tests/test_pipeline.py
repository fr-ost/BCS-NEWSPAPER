"""End-to-end pipeline behaviour with mocked model, feeds and Telegram."""

import json
from dataclasses import replace
from datetime import date

from app.http import FetchError
from app.llm import InvalidResponseError, QuotaExhaustedError
from app.llm.mock import MockProvider
from app.pipeline import Pipeline
from app.state import StateStore

from .conftest import NOW, FakeTelegram, no_sleep, start_update


class Programmable(MockProvider):
    """Mock model that records tasks and can fail on chosen calls."""

    def __init__(self, quota_after: int | None = None, reject_all=False, invalid_tasks=()):
        self.tasks: list[str] = []
        self.quota_after = quota_after
        self.reject_all = reject_all
        self.invalid_tasks = set(invalid_tasks)

    def generate_raw(self, request):
        self.tasks.append(request.task)
        if self.quota_after is not None and len(self.tasks) > self.quota_after:
            raise QuotaExhaustedError("daily quota exhausted (simulated)")
        if request.task in self.invalid_tasks:
            raise InvalidResponseError("garbage (simulated)")
        if self.reject_all and request.task == "screening":
            return json.dumps({"results": [{"id": c["id"], "decision": "reject"} for c in request.payload["candidates"]]})
        return super().generate_raw(request)


def load_state(settings):
    return json.loads(settings.state_path.read_text(encoding="utf-8"))


def test_full_run_delivers_once_and_never_duplicates(make_pipeline, settings):
    tg = FakeTelegram()
    result = make_pipeline(provider=Programmable(), telegram=tg).run(now=NOW)
    assert result.outcome == "delivered"
    assert result.stories == 3 and result.mcqs == 3 and not result.partial
    sent_count = len(tg.sent)
    assert sent_count >= 1
    edition = load_state(settings)["editions"]["2026-10-23"]
    assert edition["status"] == "delivered" and edition["recipients"] == 1
    record = edition["deliveries"]["101"]
    assert record["status"] == "done" and len(record["sent"]) == len(edition["chunks"]) == sent_count
    assert load_state(settings)["last_successful_edition"] == "2026-10-23"
    text = "\n".join(tg.sent)
    assert "Cabinet approves" in text and "উত্তর ও ব্যাখ্যা" in text and "tg-spoiler" in text

    # Retrying the same day sends nothing and calls no model.
    prov = Programmable()
    tg2 = FakeTelegram()
    again = make_pipeline(provider=prov, telegram=tg2).run(now=NOW)
    assert again.outcome == "already_delivered" and tg2.sent == [] and prov.tasks == []


def test_duplicate_reports_become_one_story_with_both_links(make_pipeline):
    tg = FakeTelegram()
    make_pipeline(telegram=tg).run(now=NOW)
    text = "\n".join(tg.sent)
    assert text.count("Cabinet approves draft amendment") == 1
    assert ">The Daily Star</a> | <a" in text and ">Dhaka Tribune</a>" in text


def test_dry_run_sends_nothing_and_writes_output(make_pipeline, settings):
    tg = FakeTelegram()
    result = make_pipeline(telegram=tg, dry_run=True).run(now=NOW)
    assert result.outcome == "dry_run" and tg.calls == 0
    assert result.output_path and result.output_path.exists()
    assert "বিসিএস দৈনিক ব্রিফিং" in result.output_path.read_text(encoding="utf-8")
    assert load_state(settings)["editions"]["2026-10-23"]["status"] == "ready"


def test_quota_exhaustion_gives_partial_edition_without_fabrication(make_pipeline, settings):
    # Call 1 = screening, call 2 = first analysis, then quota runs out.
    prov = Programmable(quota_after=2)
    tg = FakeTelegram()
    result = make_pipeline(provider=prov, telegram=tg).run(now=NOW)
    assert result.outcome == "delivered" and result.partial
    assert result.stories == 1 and result.mcqs == 0
    text = "\n".join(tg.sent)
    assert "আংশিক সংস্করণ" in text and "বিশ্লেষণ বাকি" in text
    state = load_state(settings)
    statuses = sorted(ev["status"] for ev in state["events"].values())
    assert statuses.count("included") == 1
    pending = [ev for ev in state["events"].values() if ev["status"] in ("selected", "maybe")]
    assert pending and all(ev["attempts"] == 0 for ev in pending)  # quota is not the story's fault
    assert state["editions"]["2026-10-23"]["unprocessed_candidates"] >= 1

    # A later run resumes: no re-screening, no re-analysis of finished work.
    included = [eid for eid, ev in state["events"].items() if ev["status"] == "included"]
    prov2 = Programmable()
    tg2 = FakeTelegram()
    later = make_pipeline(provider=prov2, telegram=tg2).run(now=NOW, edition_date=date(2026, 10, 24))
    assert later.outcome == "delivered"
    assert "screening" not in prov2.tasks
    # Each pending story is tried at most once; one has no usable evidence and needs no model call.
    assert 1 <= prov2.tasks.count("analysis") <= len(pending)
    state2 = load_state(settings)
    assert later.stories == prov2.tasks.count("analysis")
    assert not set(included) & set(state2["editions"]["2026-10-24"]["story_event_ids"])
    assert state2["events"][included[0]]["analysis"] == state["events"][included[0]]["analysis"]


def test_quota_before_any_story_sends_one_failure_notice_then_recovers(make_pipeline, settings):
    tg = FakeTelegram()
    r1 = make_pipeline(provider=Programmable(quota_after=0), telegram=tg).run(now=NOW)
    assert r1.outcome == "failure_notified" and len(tg.sent) == 1 and "তৈরি করা যায়নি" in tg.sent[0]
    r2 = make_pipeline(provider=Programmable(quota_after=0), telegram=tg).run(now=NOW)
    assert r2.outcome == "failure_notified" and len(tg.sent) == 1  # not notified twice
    r3 = make_pipeline(provider=Programmable(), telegram=tg).run(now=NOW)
    assert r3.outcome == "delivered" and load_state(settings)["editions"]["2026-10-23"]["status"] == "delivered"


def test_interrupted_delivery_resumes_without_resending(make_pipeline, settings, sources):
    small = replace(settings, telegram=replace(settings.telegram, max_message_chars=900))
    store = StateStore(small.state_path)

    def pipe(tg, provider=None):
        from app.__main__ import SAMPLE_DIR, offline_article_fetcher
        from app.llm import build_client

        return Pipeline(small, sources, store, build_client(small, provider or Programmable(), sleep=no_sleep),
                        offline_article_fetcher(small, SAMPLE_DIR / "articles"), telegram=tg, offline_feed_dir=SAMPLE_DIR / "feeds")

    tg1 = FakeTelegram(fail_on={2})
    r1 = pipe(tg1).run(now=NOW)
    assert r1.outcome == "failed" and len(tg1.sent) == 1
    edition = load_state(small)["editions"]["2026-10-23"]
    total = len(edition["chunks"])
    assert total >= 3 and edition["status"] == "delivering"

    prov = Programmable()
    tg2 = FakeTelegram()
    r2 = pipe(tg2, prov).run(now=NOW)
    assert r2.outcome == "delivered" and len(tg2.sent) == total - 1 and prov.tasks == []
    assert tg2.sent[0] not in tg1.sent
    assert pipe(FakeTelegram()).run(now=NOW).outcome == "already_delivered"


def test_low_relevance_day_is_not_padded(make_pipeline):
    tg = FakeTelegram()
    result = make_pipeline(provider=Programmable(reject_all=True), telegram=tg).run(now=NOW)
    assert result.outcome == "delivered" and result.stories == 0
    assert len(tg.sent) == 1 and "পূরণ করা হয়নি" in tg.sent[0]


def test_all_sources_failing_sends_failure_notice(settings, sources):
    class DeadFetcher:
        def get(self, url, **kw):
            raise FetchError("HTTP 503")

    from app.llm import build_client

    tg = FakeTelegram()
    pipe = Pipeline(settings, sources, StateStore(settings.state_path), build_client(settings, Programmable(), sleep=no_sleep),
                    lambda url: None, feed_fetcher=DeadFetcher(), telegram=tg)
    result = pipe.run(now=NOW)
    assert result.outcome == "failure_notified" and "উৎস" in tg.sent[0]
    health = json.loads(settings.state_path.read_text())["source_health"]
    assert all(h["consecutive_failures"] == 1 for h in health.values())


def test_failed_story_retries_are_limited(make_pipeline, settings):
    max_attempts = settings.analysis.max_attempts_per_story
    for day in range(max_attempts + 2):
        prov = Programmable(invalid_tasks={"analysis"})
        make_pipeline(provider=prov, telegram=FakeTelegram()).run(now=NOW, edition_date=date(2026, 11, 1 + day))
    state = load_state(settings)
    analysed_events = [ev for ev in state["events"].values() if ev.get("attempts")]
    assert analysed_events
    assert all(ev["attempts"] <= max_attempts for ev in analysed_events)
    assert all(ev["status"] == "failed" for ev in analysed_events)
    # The final run made no analysis calls at all.
    assert "analysis" not in prov.tasks


def test_state_is_checkpointed_even_when_run_crashes(make_pipeline, settings):
    class Boom(Programmable):
        def generate_raw(self, request):
            if request.task == "mcq":
                raise RuntimeError("unexpected bug")
            return super().generate_raw(request)

    import pytest

    with pytest.raises(RuntimeError):
        make_pipeline(provider=Boom(), telegram=FakeTelegram()).run(now=NOW)
    state = load_state(settings)
    assert sum(1 for ev in state["events"].values() if ev["status"] == "analysed") == 3
    assert state["runs"][-1]["outcome"] == "failed"


# --------------------------------------------------------------- subscribers


def test_every_subscriber_gets_the_whole_edition(make_pipeline, settings):
    tg = FakeTelegram(subscribers=4)
    result = make_pipeline(telegram=tg).run(now=NOW)
    assert result.outcome == "delivered"
    per_chat = {}
    for chat, text in tg.sent_to:
        per_chat.setdefault(chat, []).append(text)
    assert set(per_chat) == {"101", "102", "103", "104"}
    assert len({tuple(v) for v in per_chat.values()}) == 1  # identical edition for everyone
    assert len([r for r in tg.replies if "যুক্ত করা হয়েছে" in r[1]]) == 4  # each got a welcome
    assert load_state(settings)["editions"]["2026-10-23"]["recipients"] == 4


def test_blocked_subscriber_is_removed_without_blocking_others(make_pipeline, settings):
    tg = FakeTelegram(subscribers=3, gone={"102"})
    result = make_pipeline(telegram=tg).run(now=NOW)
    assert result.outcome == "delivered"
    assert {c for c, _ in tg.sent_to} == {"101", "103"}
    subs = load_state(settings)["subscribers"]
    assert subs["102"]["status"] == "inactive" and subs["101"]["status"] == "active"


def test_one_subscribers_failure_resends_only_their_missing_messages(make_pipeline, settings):
    # Two subscribers; the 2nd edition message overall fails (subscriber 101, chunk 2).
    tg1 = FakeTelegram(subscribers=2, fail_on={2})
    r1 = make_pipeline(telegram=tg1).run(now=NOW)
    assert r1.outcome == "failed"
    edition = load_state(settings)["editions"]["2026-10-23"]
    n = len(edition["chunks"])
    assert edition["status"] == "delivering"
    assert edition["deliveries"]["102"]["status"] == "done"  # 102 was not held back
    assert len(edition["deliveries"]["101"]["sent"]) == 1

    prov = Programmable()
    tg2 = FakeTelegram(subscribers=2)
    r2 = make_pipeline(provider=prov, telegram=tg2).run(now=NOW)
    assert r2.outcome == "delivered" and prov.tasks == []
    assert [c for c, _ in tg2.sent_to] == ["101"] * (n - 1)  # only what 101 was missing
    assert make_pipeline(telegram=FakeTelegram(subscribers=2)).run(now=NOW).outcome == "already_delivered"


def test_no_subscribers_spends_no_model_quota(make_pipeline):
    prov = Programmable()
    result = make_pipeline(provider=prov, telegram=FakeTelegram(subscribers=0)).run(now=NOW)
    assert result.outcome == "no_subscribers" and result.ok and prov.tasks == []


def test_new_subscriber_after_delivery_gets_next_edition(make_pipeline, settings):
    make_pipeline(telegram=FakeTelegram(subscribers=1)).run(now=NOW)
    late = FakeTelegram(updates=[start_update(1, 101), start_update(2, 202)])
    assert make_pipeline(telegram=late).run(now=NOW).outcome == "already_delivered"
    assert late.sent == [] and ("202", late.replies[0][1]) in late.replies
    nxt = FakeTelegram(updates=[start_update(1, 101), start_update(2, 202)])
    make_pipeline(telegram=nxt).run(now=NOW, edition_date=date(2026, 10, 24))
    assert {c for c, _ in nxt.sent_to} <= {"101", "202"}
    assert "202" in load_state(settings)["editions"]["2026-10-24"]["deliveries"]


def test_failure_notice_goes_to_all_subscribers_once(make_pipeline):
    tg = FakeTelegram(subscribers=3)
    make_pipeline(provider=Programmable(quota_after=0), telegram=tg).run(now=NOW)
    assert sorted(c for c, _ in tg.sent_to) == ["101", "102", "103"]
    make_pipeline(provider=Programmable(quota_after=0), telegram=tg).run(now=NOW)
    assert len(tg.sent_to) == 3


# --------------------------------------------------------------- OpenAI -> Gemini fallback


def test_openai_running_out_mid_run_completes_edition_with_gemini(settings, sources):
    from app.__main__ import SAMPLE_DIR, offline_article_fetcher
    from app.llm import FallbackClient, build_client

    class OpenAIStub(Programmable):
        name = "openai"

        def generate_raw(self, request):
            out = super().generate_raw(request)
            self.record_usage(1000, 500)
            return out

    class GeminiStub(Programmable):
        name = "gemini"

    openai = OpenAIStub(quota_after=2)  # screening + 1 analysis, then credit runs out
    gemini = GeminiStub()
    llm = FallbackClient([build_client(settings, openai, sleep=no_sleep), build_client(settings, gemini, sleep=no_sleep)])
    tg = FakeTelegram()
    pipe = Pipeline(settings, sources, StateStore(settings.state_path), llm, offline_article_fetcher(settings, SAMPLE_DIR / "articles"),
                    telegram=tg, offline_feed_dir=SAMPLE_DIR / "feeds")
    result = pipe.run(now=NOW)
    assert result.outcome == "delivered" and not result.partial
    assert result.stories == 3 and result.mcqs == 3
    assert openai.tasks.count("analysis") == 2 and "mcq" in gemini.tasks  # 2nd analysis call hit the limit
    assert "বিকল্প মডেল" in "\n".join(tg.sent)  # readers are told a fallback model was used
    assert result.llm_usage["openai"]["input_tokens"] == 2000
    assert result.estimated_cost_usd == round(2000 / 1e6 * 0.75 + 1000 / 1e6 * 4.50, 4)
    run = json.loads(settings.state_path.read_text())["runs"][-1]
    assert "openai" in run["providers_disabled"] and run["estimated_cost_usd"] == result.estimated_cost_usd
