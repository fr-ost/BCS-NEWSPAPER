"""Prepare early, release at 08:00: the scheduled run split."""

import json
from datetime import datetime, timedelta, timezone

from app.__main__ import release_time

from .conftest import NOW, FakeTelegram
from .test_pipeline import Programmable

HOLD = NOW + timedelta(hours=1)  # NOW is 02:00 UTC; pretend 08:00 Dhaka is an hour away


def load_state(settings):
    return json.loads(settings.state_path.read_text(encoding="utf-8"))


def test_prepare_builds_but_does_not_send_before_release_time(make_pipeline, settings):
    prov = Programmable()
    tg = FakeTelegram()
    r = make_pipeline(provider=prov, telegram=tg).run(now=NOW, hold_until=HOLD)
    assert r.outcome == "prepared" and r.ok and r.stories == 3
    assert tg.sent == [] and "analysis" in prov.tasks
    ed = load_state(settings)["editions"]["2026-10-23"]
    assert ed["status"] == "ready" and ed["chunks"] and ed["hold_until"]


def test_deliver_run_sends_prepared_edition_without_model_calls(make_pipeline, settings):
    make_pipeline(telegram=FakeTelegram()).run(now=NOW, hold_until=HOLD)
    prov = Programmable()
    tg = FakeTelegram()
    r = make_pipeline(provider=prov, telegram=tg).run(now=HOLD)  # the 08:00 trigger
    assert r.outcome == "delivered" and tg.sent and prov.tasks == []
    # Backup triggers afterwards are no-ops.
    tg2 = FakeTelegram()
    assert make_pipeline(telegram=tg2).run(now=HOLD + timedelta(minutes=20)).outcome == "already_delivered" and tg2.sent == []


def test_second_prepare_run_does_not_rebuild_or_send(make_pipeline):
    make_pipeline(telegram=FakeTelegram()).run(now=NOW, hold_until=HOLD)
    prov = Programmable()
    tg = FakeTelegram()
    r = make_pipeline(provider=prov, telegram=tg).run(now=NOW + timedelta(minutes=30), hold_until=HOLD)
    assert r.outcome == "prepared" and prov.tasks == [] and tg.sent == []


def test_late_prepare_run_after_release_time_sends_immediately(make_pipeline):
    tg = FakeTelegram()
    r = make_pipeline(telegram=tg).run(now=NOW, hold_until=NOW - timedelta(minutes=5))
    assert r.outcome == "delivered" and tg.sent


def test_deliver_without_prepare_builds_and_sends(make_pipeline):
    tg = FakeTelegram()
    assert make_pipeline(telegram=tg).run(now=HOLD).outcome == "delivered" and tg.sent


def test_release_time_is_8am_dhaka(settings):
    now = datetime(2026, 10, 25, 0, 13, tzinfo=timezone.utc)  # 06:13 Dhaka
    rt = release_time(settings, now)
    assert rt.astimezone(timezone.utc) == datetime(2026, 10, 25, 2, 0, tzinfo=timezone.utc)
