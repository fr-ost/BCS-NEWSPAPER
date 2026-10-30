"""Command-line entry point: ``python -m app <command>``.

Commands:
  run              Build today's edition and deliver it to Telegram.
  run --hold       Prepare today's edition and hold it until
                   edition.delivery_time (08:00); a later run sends it. If the
                   release is at most schedule.max_wait_minutes away, wait and
                   send it now. After the release time it sends at once.
  run --manual     Extra edition with whatever is new since the last one
                   (used by manual "live" runs; can be run any number of times).
  run --dry-run    Build with real feeds and the real model, but never send.
                   Uses a scratch state file, so production state is untouched.
  run --offline    Fully offline dry run: bundled sample feeds + mock model.
  subscribers      Read new /start and /stop messages now and list subscribers.
  test-telegram    Register new subscribers, then send each one a test message.
  verify-sources   Check every configured feed live (robots.txt, HTTP, parse).
  check-model      Check the configured Gemini model id is available to the key.
  status           Print a summary of the persisted state.
  notify-failure   Send a failure notice unless the run already reported one
                   (used by the workflow when a run crashes).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from .collector import parse_feed
from .config import PROJECT_ROOT, ConfigError, Settings, load_settings
from .extractor import Extraction, clean_article_html, extract_article
from .http import FetchError, Fetcher, RobotsDisallowed
from .llm import LLMError, build_llm, build_provider
from .pdf import make_pdf_builder
from .pipeline import Pipeline
from .sources import load_sources
from .state import StateError, StateStore, StateVersionError
from .subscribers import active_subscribers, deactivate, sync_subscribers
from .telegram_delivery import TelegramClient, TelegramError
from .util import get_tz, utcnow

SAMPLE_DIR = PROJECT_ROOT / "sample_data"
RESULT_FILE = "run-result.json"
EXIT_OK, EXIT_FAILED, EXIT_REPORTED_FAILURE, EXIT_CONFIG = 0, 1, 2, 3

log = logging.getLogger("app")
_sleep = time.sleep  # replaced in tests


class RedactingFilter(logging.Filter):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = self.settings.secrets.redact(message)
        if redacted != message:
            record.msg, record.args = redacted, None
        return True


def setup_logging(settings: Settings | None, verbose: bool) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    if settings is not None:
        handler.addFilter(RedactingFilter(settings))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    # urllib3 debug logs would include request URLs (Telegram's contains the token).
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def make_fetcher(settings: Settings) -> Fetcher:
    c = settings.collection
    return Fetcher(c.user_agent, timeout=c.request_timeout_seconds, retries=c.request_retries, respect_robots=c.respect_robots_txt)


def make_telegram(settings: Settings) -> TelegramClient:
    t = settings.telegram
    return TelegramClient(
        settings.secrets.telegram_bot_token,
        api_base=t.api_base,
        timeout=t.request_timeout_seconds,
        max_retries=t.max_retries,
        disable_preview=t.disable_web_page_preview,
        min_interval_per_chat=t.min_seconds_between_messages,
    )


def production_store(settings: Settings) -> StateStore:
    return StateStore(settings.state_path, settings.state.retention_days, settings.state.max_run_history)


def sync_now(settings: Settings, client: TelegramClient, state: dict):
    return sync_subscribers(client, state, settings.secrets.bot_invite_code, settings.telegram.max_subscribers, utcnow())


def offline_article_fetcher(settings: Settings, articles_dir: Path):
    def fetch(url: str) -> Extraction:
        slug = urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1] or "index"
        path = articles_dir / f"{slug}.html"
        if not path.exists():
            return Extraction(status="failed", reason="No offline fixture for this article")
        text = clean_article_html(path.read_bytes(), settings.extraction.max_article_chars)
        if len(text) < settings.extraction.min_article_chars:
            return Extraction(status="too_short", text=text, reason="Fixture too short")
        return Extraction(status="ok", text=text)

    return fetch


def release_time(settings: Settings, now):
    """Today's delivery_time in the edition timezone, as an aware datetime."""
    from datetime import datetime, time as dtime

    tz = get_tz(settings.edition.timezone)
    hh, mm = (int(x) for x in settings.edition.delivery_time.split(":"))
    local_day = now.astimezone(tz).date()
    return datetime.combine(local_day, dtime(hh, mm), tzinfo=tz)


def write_result(settings: Settings, payload: dict) -> None:
    try:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        (settings.output_dir / RESULT_FILE).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        log.warning("Could not write run result file")


def cmd_run(settings: Settings, args) -> int:
    sources = load_sources(settings.config_dir)
    (settings.output_dir / RESULT_FILE).unlink(missing_ok=True)  # never act on a stale result
    offline = args.offline
    dry_run = args.dry_run or offline  # offline output is mock text: never send it
    provider_name = "mock" if offline else args.provider  # None -> configured chain with fallback

    if dry_run:
        state_path = settings.output_dir / "dry-run-state.json"
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        for p in (state_path, state_path.with_name(state_path.name + ".bak")):
            p.unlink(missing_ok=True)  # every dry run starts from a clean scratch state
    else:
        state_path = settings.state_path
    store = StateStore(state_path, settings.state.retention_days, settings.state.max_run_history)

    try:
        llm = build_llm(settings, provider_name)
    except LLMError as exc:
        log.error("Model provider unavailable: %s", exc)
        llm = None

    telegram = None
    if not dry_run:
        try:
            telegram = make_telegram(settings)
        except TelegramError as exc:
            log.error("%s", exc)
            write_result(settings, {"outcome": "failed", "reason": str(exc), "failure_notified": False})
            return EXIT_CONFIG

    if llm is None:
        # The workflow's notify-failure step tells subscribers; nothing is sent from here.
        write_result(settings, {"outcome": "failed", "reason": "provider config", "failure_notified": False})
        return EXIT_CONFIG

    if offline:
        feed_dir = SAMPLE_DIR / "feeds"
        article_fetcher = offline_article_fetcher(settings, SAMPLE_DIR / "articles")
        feed_fetcher = None
    else:
        feed_dir = None
        fetcher = make_fetcher(settings)
        x = settings.extraction
        feed_fetcher = fetcher

        def article_fetcher(url: str) -> Extraction:
            return extract_article(fetcher, url, settings.collection.max_article_bytes, x.min_article_chars, x.max_article_chars)

    pdf_builder = None if args.no_pdf else make_pdf_builder()

    def make_pipeline(client) -> Pipeline:
        return Pipeline(
            settings,
            sources,
            store,
            client,
            article_fetcher,
            feed_fetcher=feed_fetcher,
            telegram=telegram,
            dry_run=dry_run,
            offline_feed_dir=feed_dir,
            pdf_builder=pdf_builder,
        )

    pipeline = make_pipeline(llm)
    edition_date = date.fromisoformat(args.date) if args.date else None
    now = utcnow()
    if offline and edition_date is None:
        # Sample feeds are dated; pin "now" so the offline run is reproducible.
        from datetime import datetime, timezone

        now = datetime(2026, 10, 23, 2, 0, tzinfo=timezone.utc)
    hold_until = release_time(settings, now) if args.hold else None
    result = pipeline.run(now=now, edition_date=edition_date, manual=args.manual, hold_until=hold_until)
    if result.outcome == "prepared" and hold_until is not None:
        # A tick that lands shortly before the release time waits here and sends
        # on time rather than relying on another (late) tick after it.
        wait = (hold_until - utcnow()).total_seconds()
        if 0 < wait <= settings.schedule.max_wait_minutes * 60:
            log.info("Edition ready; waiting %d s to release it at %s", wait, hold_until.isoformat(timespec="minutes"))
            _sleep(wait + 1)
            # A fresh pipeline so this run's record and cost exclude the build above.
            result = make_pipeline(build_llm(settings, provider_name)).run(now=utcnow(), edition_date=edition_date, hold_until=hold_until)
    payload = {
        "outcome": result.outcome,
        "edition_date": result.edition_date,
        "stories": result.stories,
        "mcqs": result.mcqs,
        "partial": result.partial,
        "llm_calls": result.llm_calls,
        "llm_usage": result.llm_usage,
        "estimated_openai_cost_usd": result.estimated_cost_usd,
        "errors": result.errors[-10:],
        "output": str(result.output_path) if result.output_path else None,
        "dry_run": dry_run,
        "failure_notified": result.outcome == "failure_notified",
    }
    write_result(settings, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    if result.outcome == "no_subscribers":
        print("WARNING: nobody has subscribed yet. Share t.me/<your_bot>?start=<BOT_INVITE_CODE> and have people press Start.")
    if result.outcome == "failure_notified":
        return EXIT_REPORTED_FAILURE
    return EXIT_OK if result.ok else EXIT_FAILED


def _print_subscribers(state: dict) -> None:
    subs = active_subscribers(state)
    print(f"Active subscribers: {len(subs)}")
    for sub in subs:
        print(f"  - {sub['name']} ({sub['type']}), joined {sub['joined_at']}")


def cmd_subscribers(settings: Settings, args) -> int:
    try:
        client = make_telegram(settings)
        me = client.get_me()
    except TelegramError as exc:
        print(f"Telegram FAILED: {exc}")
        return EXIT_FAILED
    store = production_store(settings)
    state = store.load()
    summary = sync_now(settings, client, state)
    store.save(state)
    print(f"Bot: @{me.get('username', '?')}")
    code = "<BOT_INVITE_CODE>" if settings.secrets.bot_invite_code else ""
    if code:
        print(f"Invite link: https://t.me/{me.get('username', '<bot>')}?start={code}  (replace with your code; it is secret)")
    else:
        print(f"Invite link: https://t.me/{me.get('username', '<bot>')}  (no BOT_INVITE_CODE set: anyone can join, up to {settings.telegram.max_subscribers})")
    print(f"New: {summary.added or 'none'} · Left: {summary.removed or 'none'} · Rejected: {summary.rejected}")
    _print_subscribers(state)
    return EXIT_OK


def cmd_test_telegram(settings: Settings, args) -> int:
    try:
        client = make_telegram(settings)
    except TelegramError as exc:
        print(f"Telegram test FAILED: {exc}")
        return EXIT_FAILED
    store = production_store(settings)
    state = store.load()
    try:
        sync_now(settings, client, state)
    except TelegramError as exc:
        print(f"Could not read new subscribers: {exc}")
    store.save(state)
    subs = active_subscribers(state)
    if not subs:
        print("No subscribers yet. Open your bot in Telegram (via the invite link) and press Start, then run this again.")
        return EXIT_FAILED
    failures = 0
    for sub in subs:
        try:
            client.send_message(sub["chat_id"], "✅ <b>BCS Newspaper Engine</b>\nটেস্ট মেসেজ: টেলিগ্রাম সংযোগ কাজ করছে। (এটি ব্রিফিং নয়)")
            print(f"  ok   {sub['name']}")
        except TelegramError as exc:
            failures += 1
            if exc.recipient_gone:
                deactivate(state, sub["chat_id"], "unreachable", utcnow())
            print(f"  FAIL {sub['name']}: {exc}")
    store.save(state)
    print(f"Test message delivered to {len(subs) - failures}/{len(subs)} subscribers.")
    return EXIT_OK if failures == 0 else EXIT_FAILED


def cmd_verify_sources(settings: Settings, args) -> int:
    sources = load_sources(settings.config_dir)
    fetcher = make_fetcher(settings)
    now = utcnow()
    rows = []
    for src in sources:
        if not src.feed_url:
            rows.append((src.id, "enabled" if src.enabled else "disabled", "SKIP", "no feed_url (reference only)"))
            continue
        try:
            res = fetcher.get(src.feed_url, max_bytes=settings.collection.max_feed_bytes, timeout=src.timeout_seconds, retries=1)
            items, report = parse_feed(res.content, src, now, 100)
            dated = sorted((i.published_at for i in items if i.published_at), reverse=True)
            ok = report.ok and items
            detail = f"HTTP {res.status}, {len(items)} items, newest {dated[0] if dated else 'undated'}"
            if report.warnings:
                detail += f", warnings: {'; '.join(report.warnings[:2])}"
            rows.append((src.id, "enabled" if src.enabled else "disabled", "OK" if ok else "FAIL", detail if ok else (report.error or detail)))
        except RobotsDisallowed as exc:
            rows.append((src.id, "enabled" if src.enabled else "disabled", "BLOCKED", str(exc)))
        except FetchError as exc:
            rows.append((src.id, "enabled" if src.enabled else "disabled", "FAIL", str(exc)))
    print(f"{'source':<24}{'config':<10}{'result':<9}detail")
    for row in rows:
        print(f"{row[0]:<24}{row[1]:<10}{row[2]:<9}{row[3]}")
    print("\nOnly keep sources enabled that report OK. Update config/sources.toml accordingly.")
    failed_enabled = [r for r in rows if r[1] == "enabled" and r[2] not in ("OK", "SKIP")]
    return EXIT_FAILED if failed_enabled else EXIT_OK


def cmd_check_model(settings: Settings, args) -> int:
    names = [args.provider] if args.provider else list(dict.fromkeys([settings.llm.provider, *settings.llm.fallback_providers]))
    failed = 0
    for name in names:
        model = {"openai": settings.llm.openai_model, "gemini": settings.llm.gemini_model, "ollama": settings.llm.ollama_model}.get(name, name)
        try:
            build_provider(settings, name).check_model()
            print(f"OK    {name}: model '{model}' is available")
        except LLMError as exc:
            failed += 1
            print(f"FAIL  {name}: {settings.secrets.redact(str(exc))}")
    print("Availability is not proof of credit/free quota. Prices: https://openai.com/api/pricing · https://ai.google.dev/gemini-api/docs/pricing")
    return EXIT_FAILED if failed == len(names) else EXIT_OK


def cmd_status(settings: Settings, args) -> int:
    state = production_store(settings).load()
    by_status: dict[str, int] = {}
    for ev in state["events"].values():
        by_status[ev["status"]] = by_status.get(ev["status"], 0) + 1
    summary = {
        "schema_version": state["schema_version"],
        "last_successful_edition": state.get("last_successful_edition"),
        "articles": len(state["articles"]),
        "events_by_status": by_status,
        "recent_editions": {
            d: {k: e.get(k) for k in ("status", "partial", "attempts", "failure_reason", "delivered_at")}
            for d, e in sorted(state["editions"].items())[-5:]
        },
        "source_health": state["source_health"],
        # Scheduled runs per day and how late GitHub started them (minutes).
        "schedule_days": dict(sorted(((state.get("schedule") or {}).get("days") or {}).items())[-5:]),
        "active_subscribers": [s["name"] for s in active_subscribers(state)],
        "last_run": state["runs"][-1] if state["runs"] else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return EXIT_OK


def cmd_notify_failure(settings: Settings, args) -> int:
    path = settings.output_dir / RESULT_FILE
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        result = {}
    if result.get("failure_notified") or result.get("dry_run"):
        print("Failure already reported (or dry run); not sending another notice.")
        return EXIT_OK
    try:
        client = make_telegram(settings)
        subs = active_subscribers(production_store(settings).load())
    except (TelegramError, StateError) as exc:
        print(f"Could not send failure notice: {exc}")
        return EXIT_FAILED
    sent = 0
    for sub in subs:
        try:
            client.send_message(
                sub["chat_id"],
                "⚠️ <b>আজকের বিসিএস ব্রিফিং পাঠানো যায়নি</b>\n"
                "একটি কারিগরি সমস্যা হয়েছে; বটের মালিককে জানানো হয়েছে। কোনো খবর বানিয়ে পাঠানো হয়নি।",
            )
            sent += 1
        except TelegramError as exc:
            print(f"Could not notify {sub['name']}: {exc}")
    print(f"Failure notice sent to {sent}/{len(subs)} subscribers. Details are in the GitHub Actions log.")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app", description="BCS Newspaper Engine")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="build (and deliver) today's edition")
    run.add_argument("--dry-run", action="store_true", help="never send to Telegram; use scratch state")
    run.add_argument("--offline", action="store_true", help="sample feeds + mock model; implies --dry-run")
    run.add_argument("--provider", choices=["openai", "gemini", "ollama", "mock"], help="use only this provider (no fallback)")
    run.add_argument("--date", help="edition date YYYY-MM-DD (testing)")
    run.add_argument("--hold", action="store_true", help="prepare now but don't send before edition.delivery_time (early scheduled run)")
    run.add_argument("--manual", action="store_true", help="extra edition even if today's was already sent (only new stories)")
    run.add_argument("--no-pdf", action="store_true", help="don't create/send the PDF")
    sub.add_parser("test-telegram", help="register new subscribers and send each a test message")
    sub.add_parser("subscribers", help="read /start and /stop now and list subscribers")
    sub.add_parser("verify-sources", help="check configured feeds live")
    cm = sub.add_parser("check-model", help="check the configured model is available")
    cm.add_argument("--provider", choices=["openai", "gemini", "ollama"])
    sub.add_parser("status", help="summarise persisted state")
    sub.add_parser("notify-failure", help="send a crash notice if none was sent")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except ConfigError as exc:
        setup_logging(None, args.verbose)
        log.error("Configuration error: %s", exc)
        return EXIT_CONFIG
    setup_logging(settings, args.verbose)

    handlers = {
        "run": cmd_run,
        "test-telegram": cmd_test_telegram,
        "subscribers": cmd_subscribers,
        "verify-sources": cmd_verify_sources,
        "check-model": cmd_check_model,
        "status": cmd_status,
        "notify-failure": cmd_notify_failure,
    }
    try:
        return handlers[args.command](settings, args)
    except StateVersionError as exc:
        log.error("%s", exc)
        return EXIT_CONFIG
    except ConfigError as exc:
        log.error("Configuration error: %s", exc)
        return EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
