"""Versioned JSON state with atomic writes, validation and migration.

The state file is committed to the private repository by the workflow, so
it is the memory between runs: what was seen, analysed, selected and
delivered. Writes are atomic (temp file + fsync + rename) and the previous
good version is kept as ``.bak`` for recovery within a run/host.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from .util import iso, parse_iso, utcnow

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

EVENT_STATUSES = {
    "new",  # awaiting screening
    "selected",  # screening: select
    "maybe",  # screening: maybe
    "rejected",  # screening: reject
    "analysed",  # analysis complete and usable
    "analysis_rejected",  # model/grounding judged it unusable
    "failed",  # gave up after max attempts or no accessible evidence
    "included",  # shown in a delivered edition
}
EDITION_STATUSES = {"building", "ready", "delivering", "delivered", "failed"}


class StateError(Exception):
    """State is unreadable or structurally invalid."""


class StateVersionError(StateError):
    """State was written by a newer schema. Never overwrite it."""


def empty_state() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": iso(utcnow()),
        "updated_at": None,
        "articles": {},
        "events": {},
        "editions": {},
        "last_successful_edition": None,
        "source_health": {},
        "runs": [],
        "notes": [],
    }


def migrate(data: dict) -> dict:
    """Upgrade older state formats in place, one version at a time."""
    version = data.get("schema_version", 0)
    if not isinstance(version, int):
        raise StateError("schema_version is not an integer")
    if version > SCHEMA_VERSION:
        raise StateVersionError(f"State schema {version} is newer than supported {SCHEMA_VERSION}; refusing to touch it")
    if version == 0:
        data = _migrate_v0_to_v1(data)
    return data


def _migrate_v0_to_v1(old: dict) -> dict:
    """v0 = prototype-era state: at most a list of processed URLs."""
    from .collector import article_id_for
    from .util import canonicalize_url, is_valid_http_url

    new = empty_state()
    now = iso(utcnow())
    for url in old.get("processed_urls") or []:
        if not is_valid_http_url(url):
            continue
        canonical = canonicalize_url(url)
        aid = article_id_for(canonical)
        new["articles"][aid] = {
            "id": aid,
            "source_id": "legacy",
            "source_name": "legacy",
            "title": "",
            "url": url,
            "canonical_url": canonical,
            "excerpt": "",
            "published_at": None,
            "retrieved_at": now,
            "date_status": "missing",
            "language": "",
            "event_id": None,
            "legacy": True,
        }
    new["notes"].append(f"Migrated from schema 0 at {now}")
    return new


def validate(data: object) -> dict:
    """Check structure; drop individual malformed records instead of failing."""
    if not isinstance(data, dict):
        raise StateError("State root is not an object")
    for key, typ in (("articles", dict), ("events", dict), ("editions", dict), ("source_health", dict), ("runs", list)):
        if not isinstance(data.get(key), typ):
            raise StateError(f"State field {key!r} missing or wrong type")
    dropped = 0
    for aid, art in list(data["articles"].items()):
        if not isinstance(art, dict) or art.get("id") != aid or not isinstance(art.get("canonical_url"), str):
            del data["articles"][aid]
            dropped += 1
    for eid, ev in list(data["events"].items()):
        if not isinstance(ev, dict) or ev.get("event_id") != eid or ev.get("status") not in EVENT_STATUSES or not isinstance(ev.get("article_ids"), list):
            del data["events"][eid]
            dropped += 1
    for date, ed in list(data["editions"].items()):
        if not isinstance(ed, dict) or ed.get("date") != date or ed.get("status") not in EDITION_STATUSES:
            del data["editions"][date]
            dropped += 1
    if dropped:
        log.warning("Dropped %d malformed state records", dropped)
        data.setdefault("notes", []).append(f"{iso(utcnow())}: dropped {dropped} malformed records on load")
    data.setdefault("notes", [])
    data.setdefault("last_successful_edition", None)
    return data


class StateStore:
    def __init__(self, path: Path, retention_days: int = 14, max_run_history: int = 30):
        self.path = Path(path)
        self.retention_days = retention_days
        self.max_run_history = max_run_history
        self.recovered = False  # True when state had to be recovered/reset

    @property
    def backup_path(self) -> Path:
        return self.path.with_name(self.path.name + ".bak")

    def _read(self, path: Path) -> dict:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StateError(f"Cannot read {path.name}: {type(exc).__name__}") from exc
        if not isinstance(raw, dict):
            raise StateError("State root is not an object")
        return validate(migrate(raw))

    def load(self) -> dict:
        if not self.path.exists():
            if self.backup_path.exists():
                log.warning("State file missing; trying backup")
                try:
                    data = self._read(self.backup_path)
                    self.recovered = True
                    return data
                except StateVersionError:
                    raise
                except StateError:
                    pass
            log.info("No state file; starting fresh")
            return empty_state()
        try:
            return self._read(self.path)
        except StateVersionError:
            raise
        except StateError as exc:
            log.error("State is corrupt (%s); preserving a copy and attempting recovery", exc)
            stamp = utcnow().strftime("%Y%m%dT%H%M%SZ")
            shutil.copy2(self.path, self.path.with_name(f"{self.path.name}.corrupt-{stamp}"))
            self.recovered = True
            if self.backup_path.exists():
                try:
                    data = self._read(self.backup_path)
                    data["notes"].append(f"{iso(utcnow())}: recovered from backup after corruption")
                    return data
                except StateError:
                    pass
            data = empty_state()
            data["notes"].append(f"{iso(utcnow())}: state was corrupt and no backup was usable; started fresh")
            return data

    def save(self, state: dict) -> None:
        state["schema_version"] = SCHEMA_VERSION
        state["updated_at"] = iso(utcnow())
        payload = json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True)
        json.loads(payload)  # never write something we cannot read back
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            if self.path.exists():
                shutil.copy2(self.path, self.backup_path)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def prune(self, state: dict, now: datetime | None = None) -> int:
        """Remove old articles/events/editions/runs to keep the file small.

        Delivered edition records and events still referenced by an edition
        within the retention window are kept.
        """
        now = now or utcnow()
        cutoff = now - timedelta(days=self.retention_days)
        removed = 0
        for aid, art in list(state["articles"].items()):
            seen = parse_iso(art.get("retrieved_at"))
            if seen and seen < cutoff:
                del state["articles"][aid]
                removed += 1
        for eid, ev in list(state["events"].items()):
            alive = [a for a in ev["article_ids"] if a in state["articles"]]
            if not alive:
                del state["events"][eid]
                removed += 1
            else:
                ev["article_ids"] = alive
        edition_cutoff = (now - timedelta(days=max(self.retention_days, 30))).date().isoformat()
        for date in list(state["editions"]):
            if date < edition_cutoff:
                del state["editions"][date]
                removed += 1
        state["runs"] = state["runs"][-self.max_run_history :]
        state["notes"] = state.get("notes", [])[-20:]
        return removed
