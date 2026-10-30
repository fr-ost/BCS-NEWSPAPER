"""Scheduled-tick gate. Standard library only: it runs before ``pip install``.

GitHub starts scheduled workflows late (on this repository 4-6 hours late,
and by a different amount every day), so instead of a few exact times the
workflow fires a cheap "tick" every 30 minutes through the night. Each tick
runs ``check``, which reads the committed state and the local clock
(edition.timezone) and says whether the real briefing job should start:

- before ``schedule.prepare_from``: no, the morning's news isn't in yet;
- until ``edition.delivery_time``: yes if today's edition isn't built yet
  (the job builds it and holds it), or if it is built and the release time is
  at most ``schedule.max_wait_minutes`` away (the job waits, then sends);
- from ``delivery_time`` to ``schedule.deliver_until``: yes, unless today's
  edition was delivered or the day's run limit is used up;
- after that: no.

The briefing job is idempotent (it re-reads the latest state and never sends
an edition twice), so an unnecessary "yes" costs a minute, never a duplicate.
If anything here goes wrong, the gate says yes and lets the app decide.

``record`` runs at the end of every gated (and ``deliver``) job, even a failed
one, and counts the day's runs and failures, so a persistent failure stops
after ``schedule.max_runs_per_day`` failed runs instead of retrying every 30
minutes. A built edition is never stranded by that limit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone

from .config import ConfigError, Settings, load_settings
from .util import get_tz, iso, utcnow

KEEP_DAYS = 10
MAX_LAGS_KEPT = 12


@dataclass(frozen=True)
class Decision:
    run: bool
    reason: str
    # A crash in this run should be reported to subscribers (last chance today).
    notify_on_failure: bool = False


def _at(day: date, hhmm: str, tz) -> datetime:
    hh, mm = (int(x) for x in hhmm.split(":"))
    return datetime.combine(day, dtime(hh, mm), tzinfo=tz)


def day_log(state: dict, key: str) -> dict:
    days = (state.get("schedule") or {}).get("days") if isinstance(state.get("schedule"), dict) else None
    entry = days.get(key) if isinstance(days, dict) else None
    return entry if isinstance(entry, dict) else {}


def _int(value) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def decide(state: dict | None, now: datetime, settings: Settings) -> Decision:
    """What a scheduled tick at ``now`` should do, given the committed state."""
    cfg = settings.schedule
    tz = get_tz(settings.edition.timezone)
    local = now.astimezone(tz)
    day = local.date()
    key = day.isoformat()
    start = _at(day, cfg.prepare_from, tz)
    release = _at(day, settings.edition.delivery_time, tz)
    until = _at(day, cfg.deliver_until, tz)
    late = local >= release + timedelta(hours=2)
    if not isinstance(state, dict):
        if start <= local < until:
            return Decision(True, "saved state is missing or unreadable; letting the app decide", late)
        return Decision(False, "outside the briefing window")

    editions = state.get("editions") if isinstance(state.get("editions"), dict) else {}
    edition = editions.get(key) if isinstance(editions.get(key), dict) else {}
    status = edition.get("status")
    log = day_log(state, key)
    failed = _int(log.get("failed_runs"))
    notify = not log.get("crash_notified") and (failed + 1 >= cfg.max_runs_per_day or late)

    if status == "delivered":
        return Decision(False, f"today's edition ({key}) was already delivered")
    if local < start:
        return Decision(False, f"too early: today's edition is built from {cfg.prepare_from}")
    if local >= until:
        return Decision(False, f"too late: an unsent edition is skipped after {cfg.deliver_until}")
    built = status in ("ready", "delivering") and bool(edition.get("chunks"))
    # Stop retrying a failing build; a built edition gets twice as many tries
    # to go out (e.g. Telegram briefly down) before the day is given up.
    if failed >= cfg.max_runs_per_day * (2 if built else 1):
        return Decision(False, f"{failed} failed runs today (limit {cfg.max_runs_per_day}); see the last run's log")
    if local < release:
        if not built:
            return Decision(True, f"prepare: build today's edition now and hold it until {settings.edition.delivery_time}", notify)
        left = (release - local).total_seconds() / 60
        if left <= cfg.max_wait_minutes:
            return Decision(True, f"edition ready, release in {left:.0f} min: the job waits and sends it on time", notify)
        return Decision(False, f"edition ready and held until {settings.edition.delivery_time} ({left:.0f} min)")
    return Decision(True, "deliver the held edition" if built else "build and deliver today's edition", notify)


def tick_lag(cron: str, now: datetime) -> timedelta | None:
    """How late this tick started: ``now`` minus the latest time a daily
    ``"M H * * *"`` cron (UTC) was due. None for any other cron shape."""
    try:
        minute, hour, dom, month, dow = (cron or "").split()
        if (dom, month, dow) != ("*", "*", "*"):
            return None
        m, h = int(minute), int(hour)
        due = now.astimezone(timezone.utc).replace(hour=h, minute=m, second=0, microsecond=0)
    except ValueError:
        return None
    if due > now:
        due -= timedelta(days=1)
    return now - due


def read_state(path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def record(settings: Settings, now: datetime, job_status: str, crash_notified: bool, lag_minutes: int | None) -> str:
    """Count a finished gated run in state (committed by the next step)."""
    from .state import SCHEMA_VERSION, StateStore

    path = settings.state_path
    state = read_state(path)
    if state is None:
        return "no readable state file; run not recorded"
    if state.get("schema_version") != SCHEMA_VERSION:
        return f"state schema {state.get('schema_version')!r} != {SCHEMA_VERSION}; run not recorded"
    key = now.astimezone(get_tz(settings.edition.timezone)).date().isoformat()
    sched = state.get("schedule") if isinstance(state.get("schedule"), dict) else {}
    days = sched.get("days") if isinstance(sched.get("days"), dict) else {}
    entry = days.get(key) if isinstance(days.get(key), dict) else {}
    entry["runs"] = _int(entry.get("runs")) + 1
    if job_status != "success":
        entry["failed_runs"] = _int(entry.get("failed_runs")) + 1
    entry["last_run_at"] = iso(now)
    entry["last_job_status"] = (job_status or "unknown")[:20]
    if lag_minutes is not None:
        entry["lag_minutes"] = ([x for x in entry.get("lag_minutes") or [] if isinstance(x, int)] + [lag_minutes])[-MAX_LAGS_KEPT:]
    if crash_notified:
        entry["crash_notified"] = True
    days[key] = entry
    cutoff = (date.fromisoformat(key) - timedelta(days=KEEP_DAYS)).isoformat()
    sched["days"] = {k: v for k, v in days.items() if k >= cutoff}
    state["schedule"] = sched
    StateStore(path).save(state)
    return f"run {entry['runs']} today recorded ({entry['last_job_status']}; {_int(entry.get('failed_runs'))} failed of {settings.schedule.max_runs_per_day} allowed)"


def _fmt(delta: timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    return f"{minutes // 60} h {minutes % 60:02d} min"


def _write_outputs(values: dict[str, str]) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for k, v in values.items():
            fh.write(f"{k}={v}\n")


def cmd_check(settings: Settings | None, now: datetime) -> int:
    cron = os.environ.get("TICK_CRON", "")
    lag = tick_lag(cron, now)
    if settings is None:
        decision = Decision(True, "configuration unreadable; letting the app decide", True)
    else:
        try:
            decision = decide(read_state(settings.state_path), now, settings)
        except Exception as exc:  # the gate must never block the briefing
            decision = Decision(True, f"gate error ({type(exc).__name__}); letting the app decide", True)
    local = now.astimezone(get_tz(settings.edition.timezone if settings else "Asia/Dhaka"))
    started = f"started {now.astimezone(timezone.utc):%H:%M} UTC ({local:%H:%M} local)"
    when = f"tick '{cron}' {started}, {_fmt(lag)} after it was due" if lag is not None else f"tick {started}"
    verdict = "RUN" if decision.run else "skip"
    print(f"::notice title=Schedule gate::{verdict}: {decision.reason} [{when}]")
    _write_outputs(
        {
            "run": "true" if decision.run else "false",
            "notify": "true" if decision.notify_on_failure else "false",
            "lag_minutes": str(int(lag.total_seconds() // 60)) if lag is not None else "",
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m app.schedule_gate")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="decide whether this scheduled tick should run the briefing")
    check.add_argument("--now", help="pretend time (ISO 8601, for testing)")
    rec = sub.add_parser("record", help="count a finished gated run in state")
    rec.add_argument("--job-status", default="unknown")
    rec.add_argument("--crash-notified", default="false")
    rec.add_argument("--lag-minutes", default="")
    args = parser.parse_args(argv)
    now = datetime.fromisoformat(args.now) if getattr(args, "now", None) else utcnow()
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        settings = None
    if args.command == "check":
        return cmd_check(settings, now)
    if settings is None:
        return 0  # never fail the job over bookkeeping
    lag = int(args.lag_minutes) if args.lag_minutes.strip().lstrip("-").isdigit() else None
    try:
        print(record(settings, now, args.job_status, args.crash_notified.strip().lower() == "true", lag))
    except Exception as exc:  # bookkeeping must never fail the job
        print(f"Could not record the run: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
