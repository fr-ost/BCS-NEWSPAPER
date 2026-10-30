"""The scheduled-tick gate: what each (late, unpredictable) GitHub tick does."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from app import __main__ as cli
from app.schedule_gate import decide, main as gate_main, record, tick_lag

from .conftest import FakeTelegram

UTC = timezone.utc


def dhaka(hh: int, mm: int = 0, day: int = 31) -> datetime:
    """A UTC instant for HH:MM Asia/Dhaka on 2026-10-<day>."""
    return datetime(2026, 10, day, hh, mm, tzinfo=UTC) - timedelta(hours=6)


def state_with(edition: dict | None = None, runs: int = 0, crash_notified: bool = False, key: str = "2026-10-31") -> dict:
    state = {"schema_version": 2, "editions": {}, "schedule": {"days": {}}}
    if edition is not None:
        state["editions"][key] = {"date": key, **edition}
    if runs or crash_notified:
        state["schedule"]["days"][key] = {"runs": runs, "crash_notified": crash_notified}
    return state


READY = {"status": "ready", "chunks": [{"index": 0, "text": "x"}]}


@pytest.mark.parametrize(
    "when, edition, expected",
    [
        (dhaka(1, 30), None, False),  # too early: tonight's news isn't in
        (dhaka(4, 59), None, False),
        (dhaka(5, 7), None, True),  # prepare and hold
        (dhaka(6, 37), READY, False),  # built, release 83 min away: nothing to do
        (dhaka(7, 41), READY, True),  # built, release 19 min away: wait and send on time
        (dhaka(7, 41), None, True),  # not built yet: build now (sends at 08:00 if done in time)
        (dhaka(8, 7), READY, True),  # deliver
        (dhaka(8, 7), None, True),  # nothing prepared: build and send
        (dhaka(8, 7), {"status": "delivered"}, False),
        (dhaka(9, 30), {"status": "delivering", "chunks": [{"index": 0}]}, True),  # finish a partial delivery
        (dhaka(9, 30), {"status": "failed", "attempts": 1}, True),  # retry a failed build
        (dhaka(9, 30), {"status": "building"}, True),  # a crashed build
        (dhaka(22, 5), None, False),  # too late today
    ],
)
def test_decision_table(settings, when, edition, expected):
    d = decide(state_with(edition), when, settings)
    assert d.run is expected, d.reason


def test_runs_are_capped_per_day(settings):
    assert decide(state_with(None, runs=4), dhaka(9), settings).run is True
    d = decide(state_with(None, runs=5), dhaka(9), settings)
    assert d.run is False and "5 scheduled runs" in d.reason
    # A new day starts a fresh count.
    assert decide(state_with(None, runs=5, key="2026-10-30"), dhaka(9), settings).run is True


def test_crash_notice_only_on_the_last_chance(settings):
    assert decide(state_with(None), dhaka(8, 7), settings).notify_on_failure is False
    assert decide(state_with(None, runs=4), dhaka(8, 7), settings).notify_on_failure is True  # 5th = last run
    assert decide(state_with(None, runs=1), dhaka(10, 30), settings).notify_on_failure is True  # 2h+ late
    assert decide(state_with(None, runs=4, crash_notified=True), dhaka(8, 7), settings).notify_on_failure is False


def test_missing_or_garbage_state_fails_open_inside_the_window(settings):
    assert decide(None, dhaka(8, 7), settings).run is True
    assert decide(None, dhaka(3), settings).run is False
    assert decide({"editions": "garbage", "schedule": []}, dhaka(8, 7), settings).run is True


def test_manual_editions_do_not_count_as_the_scheduled_one(settings):
    state = state_with(None)
    state["editions"]["2026-10-31-m1"] = {"date": "2026-10-31-m1", "status": "delivered"}
    assert decide(state, dhaka(8, 7), settings).run is True


def test_tick_lag():
    now = datetime(2026, 10, 31, 2, 11, tzinfo=UTC)
    assert tick_lag("7 21 * * *", now) == timedelta(hours=5, minutes=4)  # yesterday 21:07 UTC
    assert tick_lag("7 2 * * *", now) == timedelta(minutes=4)
    assert tick_lag("7,37 2 * * *", now) is None
    assert tick_lag("", now) is None


def test_check_writes_github_outputs(settings, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BCS_STATE_PATH", str(settings.state_path))
    out = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("TICK_CRON", "7 21 * * *")
    assert gate_main(["check", "--now", dhaka(8, 11).isoformat()]) == 0
    values = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert values == {"run": "true", "notify": "false", "lag_minutes": "304"}
    assert "5 h 04 min after it was due" in capsys.readouterr().out


def test_record_counts_runs_and_keeps_state_valid(settings):
    settings.state_path.parent.mkdir(parents=True)
    settings.state_path.write_text(json.dumps({**state_with(READY), "articles": {}, "events": {}}), encoding="utf-8")
    record(settings, dhaka(8, 9), "failure", False, 301)
    msg = record(settings, dhaka(8, 40), "success", True, 290)
    assert "run 2 of 5" in msg
    data = json.loads(settings.state_path.read_text(encoding="utf-8"))
    day = data["schedule"]["days"]["2026-10-31"]
    assert day["runs"] == 2 and day["crash_notified"] is True and day["lag_minutes"] == [301, 290]
    assert data["editions"]["2026-10-31"]["status"] == "ready"  # nothing else touched


def test_record_never_creates_or_rewrites_unknown_state(settings):
    assert "not recorded" in record(settings, dhaka(8), "success", False, None)
    assert not settings.state_path.exists()
    settings.state_path.parent.mkdir(parents=True)
    settings.state_path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    assert "not recorded" in record(settings, dhaka(8), "success", False, None)
    assert json.loads(settings.state_path.read_text()) == {"schema_version": 99}


def test_record_prunes_old_days(settings):
    state = state_with(None)
    state["schedule"]["days"] = {"2026-10-01": {"runs": 3}, "2026-10-30": {"runs": 2}}
    settings.state_path.parent.mkdir(parents=True)
    settings.state_path.write_text(json.dumps({**state, "articles": {}, "events": {}}), encoding="utf-8")
    record(settings, dhaka(8), "success", False, None)
    days = json.loads(settings.state_path.read_text())["schedule"]["days"]
    assert set(days) == {"2026-10-30", "2026-10-31"}


def test_hold_waits_and_sends_on_time_when_release_is_close(monkeypatch, tmp_path):
    """`run --hold` landing at 07:45 builds, waits until 08:00, then sends."""
    clock = {"now": dhaka(7, 45, day=23) + timedelta(days=0)}
    slept = []

    def fake_sleep(seconds):
        slept.append(seconds)
        clock["now"] += timedelta(seconds=seconds)

    monkeypatch.setattr(cli, "utcnow", lambda: clock["now"])
    monkeypatch.setattr(cli, "_sleep", fake_sleep)
    tg = FakeTelegram()
    monkeypatch.setattr(cli, "make_telegram", lambda settings: tg)
    monkeypatch.setattr(cli, "make_fetcher", lambda settings: None)
    _offline_env(monkeypatch, tmp_path)
    assert cli.main(["run", "--hold", "--no-pdf"]) == 0
    result = json.loads((tmp_path / "out" / "run-result.json").read_text())
    assert result["outcome"] == "delivered" and tg.sent
    assert len(slept) == 1 and 14 * 60 <= slept[0] <= 15 * 60 + 1


def test_hold_does_not_wait_when_release_is_far(monkeypatch, tmp_path):
    clock = {"now": dhaka(6, 0, day=23)}
    monkeypatch.setattr(cli, "utcnow", lambda: clock["now"])
    monkeypatch.setattr(cli, "_sleep", lambda s: pytest.fail("must not wait 2 hours in a job"))
    tg = FakeTelegram()
    monkeypatch.setattr(cli, "make_telegram", lambda settings: tg)
    monkeypatch.setattr(cli, "make_fetcher", lambda settings: None)
    _offline_env(monkeypatch, tmp_path)
    assert cli.main(["run", "--hold", "--no-pdf"]) == 0
    assert json.loads((tmp_path / "out" / "run-result.json").read_text())["outcome"] == "prepared" and tg.sent == []


def _offline_env(monkeypatch, tmp_path):
    """Real CLI wiring, but sample feeds/articles and the mock model."""
    from app.__main__ import SAMPLE_DIR, offline_article_fetcher
    from app.collector import load_fixture_items

    monkeypatch.setenv("BCS_STATE_PATH", str(tmp_path / "state" / "state.json"))
    monkeypatch.setenv("BCS_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_FALLBACK", "none")
    monkeypatch.setenv("TELE_BOT_TOKEN", "123456:test-telegram-token")
    import app.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "collect", lambda sources, fetcher, now, max_items, max_bytes: load_fixture_items(sources, SAMPLE_DIR / "feeds", now, max_items))
    monkeypatch.setattr(cli, "extract_article", lambda fetcher, url, *a: offline_article_fetcher(cli.load_settings(), SAMPLE_DIR / "articles")(url))
