import json
from datetime import datetime, timedelta, timezone

import pytest

from app.state import SCHEMA_VERSION, StateStore, StateVersionError, empty_state


@pytest.fixture
def store(tmp_path):
    return StateStore(tmp_path / "state.json", retention_days=14)


def test_missing_state_starts_fresh(store):
    state = store.load()
    assert state["schema_version"] == SCHEMA_VERSION and state["articles"] == {}
    assert not store.recovered


def test_save_is_atomic_and_round_trips(store, tmp_path):
    state = store.load()
    state["last_successful_edition"] = "2026-10-22"
    store.save(state)
    assert store.load()["last_successful_edition"] == "2026-10-22"
    store.save(state)
    assert store.backup_path.exists()
    assert not list(tmp_path.glob(".state-*.tmp"))  # no temp files left behind


def test_failed_write_keeps_previous_state(store, monkeypatch):
    state = store.load()
    state["last_successful_edition"] = "good"
    store.save(state)
    import app.state as st

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(st.os, "replace", boom)
    state["last_successful_edition"] = "bad"
    with pytest.raises(OSError):
        store.save(state)
    monkeypatch.undo()
    assert store.load()["last_successful_edition"] == "good"


def test_corrupt_state_recovers_from_backup(store):
    state = store.load()
    state["last_successful_edition"] = "2026-10-21"
    store.save(state)
    store.save(state)  # creates .bak with the good content
    store.path.write_text("{ this is not json", encoding="utf-8")
    recovered = store.load()
    assert recovered["last_successful_edition"] == "2026-10-21"
    assert store.recovered
    assert list(store.path.parent.glob("state.json.corrupt-*"))  # corrupt copy preserved


def test_corrupt_state_without_backup_starts_fresh_but_preserves_copy(store):
    store.path.write_text('["wrong", "root"]', encoding="utf-8")
    state = store.load()
    assert state["articles"] == {} and store.recovered
    assert any("corrupt" in n for n in state["notes"])
    assert list(store.path.parent.glob("state.json.corrupt-*"))


def test_structurally_invalid_state_is_corrupt(store):
    store.path.write_text(json.dumps({"schema_version": 1, "articles": []}), encoding="utf-8")
    assert store.load()["articles"] == {} and store.recovered


def test_newer_schema_is_never_overwritten(store):
    store.path.write_text(json.dumps({"schema_version": SCHEMA_VERSION + 1}), encoding="utf-8")
    with pytest.raises(StateVersionError):
        store.load()
    assert json.loads(store.path.read_text())["schema_version"] == SCHEMA_VERSION + 1


def test_migration_from_v0(store):
    store.path.write_text(json.dumps({"processed_urls": ["https://www.example.com/a/1", "javascript:bad"]}), encoding="utf-8")
    state = store.load()
    assert state["schema_version"] == SCHEMA_VERSION
    assert [a["canonical_url"] for a in state["articles"].values()] == ["https://example.com/a/1"]


def test_malformed_records_are_dropped_not_fatal(store):
    data = empty_state()
    data["articles"] = {"a1": {"id": "a1", "canonical_url": "https://e.com/1"}, "a2": {"id": "WRONG"}, "a3": "junk"}
    data["events"] = {"e1": {"event_id": "e1", "status": "bogus", "article_ids": []}}
    data["editions"] = {"2026-10-22": {"date": "2026-10-22", "status": "delivered"}, "x": {"date": "y", "status": "delivered"}}
    store.path.write_text(json.dumps(data), encoding="utf-8")
    state = store.load()
    assert list(state["articles"]) == ["a1"] and state["events"] == {} and list(state["editions"]) == ["2026-10-22"]


def test_prune_removes_old_articles_and_orphan_events(store):
    now = datetime(2026, 10, 23, tzinfo=timezone.utc)
    state = empty_state()
    old, new = (now - timedelta(days=20)).isoformat(), now.isoformat()
    state["articles"] = {
        "a_old": {"id": "a_old", "canonical_url": "u1", "retrieved_at": old},
        "a_new": {"id": "a_new", "canonical_url": "u2", "retrieved_at": new},
    }
    state["events"] = {
        "e_old": {"event_id": "e_old", "status": "new", "article_ids": ["a_old"]},
        "e_mix": {"event_id": "e_mix", "status": "new", "article_ids": ["a_old", "a_new"]},
    }
    store.prune(state, now)
    assert list(state["articles"]) == ["a_new"]
    assert list(state["events"]) == ["e_mix"] and state["events"]["e_mix"]["article_ids"] == ["a_new"]


def test_migration_from_v1_adds_subscribers_and_keeps_delivery_records(store):
    v1 = empty_state()
    v1["schema_version"] = 1
    del v1["subscribers"], v1["telegram"]
    v1["editions"] = {"2026-10-22": {"date": "2026-10-22", "status": "delivered", "chunks": [{"index": 0, "message_id": 55}]}}
    store.path.write_text(json.dumps(v1), encoding="utf-8")
    state = store.load()
    assert state["schema_version"] == SCHEMA_VERSION == 2
    assert state["subscribers"] == {} and state["telegram"] == {"last_update_id": None}
    assert state["editions"]["2026-10-22"]["deliveries"]["legacy"] == {"sent": {"0": 55}, "status": "done", "attempts": 1}
