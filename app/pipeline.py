"""Daily edition pipeline: collect → group → screen → analyse → rank → MCQ →
render → deliver, with a state checkpoint after every stage and story.

Idempotency rules:
- The scheduled run makes one edition per Asia/Dhaka date and never re-sends it.
- A manual run (``manual=True``) always makes an extra edition with whatever
  is new since the last one; stories already sent are never repeated.
- An edition is rendered once and its chunks stored before sending. Delivery
  is tracked per subscriber and per chunk; a retry sends only the chunks a
  subscriber has not yet received.
- Completed analyses are stored and never recomputed.
- Failed stories are retried at most ``max_attempts_per_story`` times;
  quota exhaustion does not count as an attempt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

from .analysis import AnalysisInput, AnalysisRejected, analyse
from .collector import FeedItem, SourceReport, collect, load_fixture_items
from .config import Settings
from .deduplicator import assign_events, distinctive_numbers, merge_groups, same_story
from .enrichment import enrich_static_gk
from .extractor import FULL, Extraction, choose_analysis_input
from .llm import (
    FallbackClient,
    InvalidResponseError,
    LLMClient,
    LLMError,
    PermanentLLMError,
    QuotaExhaustedError,
    TransientLLMError,
)
from .questions import generate_mcqs
from .render import bn_date, render_edition, render_failure_notice, render_pdf_html
from .scoring import rank_stories
from .screening import screen_events
from .sources import Source
from .state import StateStore
from .syllabus import SyllabusProfile, load_syllabus
from .subscribers import active_subscribers, deactivate, ensure_structures, sync_subscribers
from .telegram_delivery import TelegramClient, TelegramError, chunk_sections
from .util import get_tz, iso, normalize_text, parse_iso, sha256, utcnow

log = logging.getLogger(__name__)

DECISION_RANK = {"select": 2, "maybe": 1, "reject": 0}
MAX_SCREENING_ATTEMPTS = 3
MAX_LINKS_PER_STORY = 4
SOURCE_FAILURE_ALERT = 3
# Stories delivered within this many days are shown to screening so repeats
# of them are rejected (genuine follow-ups with a new development are kept).
RECENT_DAYS = 7
MAX_RECENT = 60

ArticleFetcher = Callable[[str], Extraction]
PdfBuilder = Callable[[str], bytes]  # HTML -> PDF bytes


@dataclass
class RunResult:
    outcome: str
    edition_date: str
    stories: int = 0
    mcqs: int = 0
    partial: bool = False
    llm_calls: int = 0
    llm_usage: dict = field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    errors: list[str] = field(default_factory=list)
    output_path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.outcome in {"delivered", "already_delivered", "dry_run", "failure_notified", "no_subscribers", "prepared"}


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        sources: list[Source],
        store: StateStore,
        llm: LLMClient | FallbackClient,
        article_fetcher: ArticleFetcher,
        feed_fetcher=None,
        telegram: TelegramClient | None = None,
        dry_run: bool = False,
        offline_feed_dir: Path | None = None,
        pdf_builder: PdfBuilder | None = None,
        profile: SyllabusProfile | None = None,
    ):
        self.s = settings
        self.sources = sources
        self.sources_by_id = {src.id: src for src in sources}
        self.store = store
        self.llm = llm
        self.fetch_article = article_fetcher
        self.feed_fetcher = feed_fetcher
        self.telegram = telegram
        self.dry_run = dry_run
        self.offline_feed_dir = offline_feed_dir
        self.pdf_builder = pdf_builder
        self.profile = profile or load_syllabus(settings.config_dir)
        self._title_filters = [re.compile(p, re.I) for p in settings.filters.exclude_title_patterns]
        self.prefiltered = 0
        self.screen_stats: dict[str, int] = {}
        self._pdf_cache: dict[str, bytes | None] = {}
        self.state: dict = {}
        self.errors: list[str] = []
        self.llm_stop_reason: str | None = None  # "quota" | "permanent: ..." when the model became unusable
        self.screening_incomplete = False
        self._started = time.monotonic()
        self._hold_until: datetime | None = None

    # ------------------------------------------------------------------ helpers
    def _save(self) -> None:
        self.store.save(self.state)

    def _error(self, msg: str) -> None:
        msg = self.s.secrets.redact(msg)
        log.warning(msg)
        self.errors.append(msg)

    def _fresh(self, dt_value: str | None, now: datetime) -> bool:
        dt = parse_iso(dt_value)
        return dt is not None and dt >= now - timedelta(hours=self.s.edition.freshness_hours)

    def _event_time(self, ev: dict) -> str | None:
        """Newest known publication time, else first-seen time."""
        times = []
        for aid in ev["article_ids"]:
            art = self.state["articles"].get(aid)
            if art:
                times.append(art.get("published_at") or art.get("retrieved_at"))
        times = [t for t in times if t]
        return max(times) if times else ev.get("created_at")

    def _articles_of(self, ev: dict) -> list[dict]:
        arts = [self.state["articles"][a] for a in ev["article_ids"] if a in self.state["articles"]]
        # Primary article: one we may fetch, then the richest excerpt.
        def key(a: dict):
            src = self.sources_by_id.get(a["source_id"])
            return (0 if src and src.fetch_articles else 1, -len(a.get("excerpt", "")), a["id"])

        return sorted(arts, key=key)

    # ------------------------------------------------------------------ run
    def run(self, now: datetime | None = None, edition_date: date | None = None, manual: bool = False, hold_until: datetime | None = None) -> RunResult:
        """Build and deliver an edition.

        ``hold_until``: build everything now but don't send before this time
        (UTC). A later run (the 08:00 delivery trigger) sends the stored edition.
        """
        now = now or utcnow()
        self._started = time.monotonic()
        self._hold_until = hold_until
        tz = get_tz(self.s.edition.timezone)
        ed_date = edition_date or now.astimezone(tz).date()
        key = ed_date.isoformat()
        self.state = self.store.load()
        run_record = {"run_id": uuid.uuid4().hex[:10], "started_at": iso(now), "edition_date": key, "dry_run": self.dry_run, "manual": manual}
        if self.store.recovered:
            self._error("State had to be recovered or reset; duplicate protection may be weaker for this run")
        result = RunResult(outcome="failed", edition_date=key)
        try:
            result = self._run(now, ed_date, key, manual)
        except Exception as exc:
            self._error(f"Unexpected error: {type(exc).__name__}: {exc}")
            result = RunResult(outcome="failed", edition_date=key)
            raise
        finally:
            result.errors = self.errors
            result.llm_calls = self.llm.calls_made
            result.llm_usage = self.llm.usage()
            result.estimated_cost_usd = self._estimated_cost(result.llm_usage)
            run_record.update(
                edition_date=result.edition_date,  # a manual run's own key (<date>-mN)
                llm_usage=result.llm_usage,
                estimated_cost_usd=result.estimated_cost_usd,
                providers_disabled=dict(getattr(self.llm, "disabled", {})),
                finished_at=iso(utcnow()),
                outcome=result.outcome,
                stories=result.stories,
                mcqs=result.mcqs,
                partial=result.partial,
                llm_calls=self.llm.calls_made,
                prefiltered=self.prefiltered,
                errors=self.errors[-20:],
            )
            self.state.setdefault("runs", []).append(run_record)
            self.store.prune(self.state, now)
            self._save()
        return result

    def _estimated_cost(self, usage: dict) -> float:
        """Rough OpenAI spend for this run from reported tokens (Gemini free tier = 0)."""
        u = usage.get("openai")
        if not u:
            return 0.0
        cfg = self.s.llm
        cost = u["input_tokens"] / 1e6 * cfg.openai_usd_per_million_input + u["output_tokens"] / 1e6 * cfg.openai_usd_per_million_output
        return round(cost, 4)

    def _model_notices(self) -> list[str]:
        if getattr(self.llm, "fallback_used", False):
            primary = self.llm.provider_names[0]
            return [f"আজ {primary} সাময়িকভাবে ব্যবহার করা যায়নি, তাই কিছু বা সব বিশ্লেষণ বিকল্প মডেল দিয়ে করা হয়েছে।"]
        return []

    def _sync_subscribers(self, now: datetime) -> None:
        ensure_structures(self.state)
        if self.dry_run or self.telegram is None:
            return
        try:
            sync_subscribers(self.telegram, self.state, self.s.secrets.bot_invite_code, self.s.telegram.max_subscribers, now)
        except TelegramError as exc:
            self._error(f"Could not read new subscribers this run: {exc}")
        self._save()

    def _run(self, now: datetime, ed_date: date, key: str, manual: bool = False) -> RunResult:
        self._sync_subscribers(now)
        manual = manual and not self.dry_run
        if manual:
            # Finish any interrupted edition from today before starting another.
            for ed in self.state["editions"].values():
                if ed.get("edition_date", ed["date"][:10]) == key and ed["status"] in ("ready", "delivering") and ed.get("chunks"):
                    log.info("Resuming delivery of edition %s before a new manual edition", ed["date"])
                    return self._deliver(ed)
            n = 1 + sum(1 for k in self.state["editions"] if k.startswith(f"{key}-m"))
            key = f"{key}-m{n}"
        edition = self.state["editions"].get(key)
        if edition and edition["status"] == "delivered":
            log.info("Edition %s already delivered; nothing to do", key)
            return RunResult(outcome="already_delivered", edition_date=key, stories=len(edition.get("story_event_ids", [])))
        if edition and edition["status"] in ("ready", "delivering") and edition.get("chunks"):
            if self._hold_until is not None and not self.dry_run and now < self._hold_until:
                return RunResult(outcome="prepared", edition_date=key, stories=len(edition.get("story_event_ids", [])))
            if self.dry_run:
                return RunResult(outcome="dry_run", edition_date=key, stories=len(edition.get("story_event_ids", [])))
            log.info("Resuming delivery of edition %s", key)
            return self._deliver(edition)
        if not self.dry_run and not active_subscribers(self.state):
            # Don't spend model quota on an edition nobody will receive.
            self._error("No subscribers yet: share the bot's invite link and have people press Start")
            return RunResult(outcome="no_subscribers", edition_date=key)

        previous = edition or {}
        edition = {
            "date": key,
            "edition_date": ed_date.isoformat(),
            "manual": manual,
            "status": "building",
            "created_at": iso(now),
            "attempts": int(previous.get("attempts", 0)) + 1,
            "failure_notified": bool(previous.get("failure_notified", False)),
            "dry_run": self.dry_run,
        }
        self.state["editions"][key] = edition

        reports = self._collect(now)
        self._save()
        self._screen(now)
        self._save()
        self._analyse(now)
        self._save()
        stories = self._select(now)
        self._enrich(stories)
        self._save()
        self._log_selection(stories)
        mcqs, mcq_problem = self._questions(stories)

        unprocessed = self._unprocessed_count(now)
        partial_reasons: list[str] = []
        if self.llm_stop_reason == "quota":
            partial_reasons.append("AI মডেলের ফ্রি কোটা/অনুরোধ-সীমা শেষ হওয়ায় সব খবর বিশ্লেষণ করা যায়নি।")
        elif self.llm_stop_reason:
            partial_reasons.append("AI মডেলে ত্রুটির কারণে বিশ্লেষণ অসম্পূর্ণ।")
        if self.screening_incomplete and not self.llm_stop_reason:
            partial_reasons.append("প্রাথমিক বাছাই (screening) সম্পূর্ণ হয়নি।")
        if mcq_problem:
            partial_reasons.append(mcq_problem)

        collectable = [src for src in self.sources if src.collectable]
        all_sources_failed = bool(collectable) and reports and all(not r.ok for r in reports)
        if not stories and (self.llm_stop_reason or all_sources_failed or not collectable):
            if not collectable:
                reason = "কোনো সংবাদ উৎস চালু (enabled) নেই।"
            elif all_sources_failed:
                reason = "কোনো সংবাদ উৎস থেকে খবর আনা যায়নি।"
            elif self.llm_stop_reason == "quota":
                reason = "AI মডেলের ফ্রি কোটা শেষ, তাই আজ বিশ্লেষণ করা যায়নি।"
            else:
                reason = "AI মডেলে ত্রুটি, তাই আজ বিশ্লেষণ করা যায়নি।"
            return self._fail(edition, ed_date, reason, unprocessed)

        earlier_today = any(
            e.get("edition_date") == ed_date.isoformat() and e["status"] == "delivered" for e in self.state["editions"].values()
        )
        if not stories and not manual and earlier_today and not partial_reasons:
            # A manual edition already went out today and nothing new turned up:
            # don't message everyone just to say so.
            edition.update(status="delivered", delivered_at=iso(utcnow()), recipients=0, story_event_ids=[], mcqs=[], chunks=[], skipped="nothing new since an earlier edition today")
            self._save()
            return RunResult(outcome="delivered", edition_date=key)

        sections = render_edition(
            ed_date,
            stories,
            mcqs,
            partial_reasons,
            unprocessed if partial_reasons else 0,
            self.sources,
            self.s.edition.mcq_count,
            notices=self._model_notices() + self._source_notices(),
            extra=manual,
            since_last=earlier_today,
        )
        chunks = chunk_sections(sections, self.s.telegram.max_message_chars)
        edition.update(
            status="ready",
            partial=bool(partial_reasons),
            partial_reasons=partial_reasons,
            unprocessed_candidates=unprocessed,
            story_event_ids=[st["event_id"] for st in stories],
            mcqs=mcqs,
            chunks=[{"index": i, "text": c, "sha": sha256(c)[:16], "status": "pending", "message_id": None, "attempts": 0} for i, c in enumerate(chunks)],
        )
        output_path = self._write_output(key, chunks, edition)
        if stories:
            self._build_pdf(edition, ed_date, stories, mcqs, partial_reasons, manual)
        self._save()

        if self._hold_until is not None and not self.dry_run:
            effective_now = now + timedelta(seconds=time.monotonic() - self._started)
            if effective_now < self._hold_until:
                log.info("Edition %s prepared; holding until %s for delivery", key, iso(self._hold_until))
                edition["hold_until"] = iso(self._hold_until)
                self._save()
                return RunResult(outcome="prepared", edition_date=key, stories=len(stories), mcqs=len(mcqs), partial=bool(partial_reasons), output_path=output_path)

        if self.dry_run:
            log.info("Dry run: edition rendered to %s, not sent", output_path)
            res = RunResult(outcome="dry_run", edition_date=key, stories=len(stories), mcqs=len(mcqs), partial=bool(partial_reasons), output_path=output_path)
            return res
        res = self._deliver(edition)
        res.output_path = output_path
        res.mcqs = len(mcqs)
        return res

    # ------------------------------------------------------------------ Stage A+B
    def _collect(self, now: datetime) -> list[SourceReport]:
        max_items = self.s.collection.max_items_per_source
        if self.offline_feed_dir is not None:
            items, reports = load_fixture_items(self.sources, self.offline_feed_dir, now, max_items)
        else:
            items, reports = collect(self.sources, self.feed_fetcher, now, max_items, self.s.collection.max_feed_bytes)
        self._update_source_health(reports, now)
        for r in reports:
            if not r.ok:
                self._error(f"Source {r.source_id} failed: {r.error}")

        new_items: list[FeedItem] = []
        for item in items:
            if item.id in self.state["articles"]:
                continue
            if item.date_status == "ok" and not self._fresh(item.published_at, now):
                continue  # old item still in the feed
            if self._excluded(item):
                self.prefiltered += 1
                continue  # off-syllabus section (sports, opinion, district, ...); never costs a model call
            new_items.append(item)
        if self.prefiltered:
            log.info("Pre-filter dropped %d off-syllabus items (sections/titles)", self.prefiltered)
        if not new_items:
            log.info("No new articles")
            return reports

        new_dicts = [i.to_dict() for i in new_items]
        horizon = now - timedelta(hours=self.s.edition.freshness_hours * 2)
        existing = [
            a for a in self.state["articles"].values()
            if a.get("event_id") and (parse_iso(a.get("retrieved_at")) or horizon) >= horizon
        ]
        mapping = assign_events(new_dicts, existing)
        for art in new_dicts:
            eid = mapping[art["id"]]
            art["event_id"] = eid
            art["content_hash"] = sha256(normalize_text(art["title"] + " " + art["excerpt"]))[:16]
            art["extraction"] = None
            self.state["articles"][art["id"]] = art
            ev = self.state["events"].get(eid)
            if ev is None:
                self.state["events"][eid] = {
                    "event_id": eid,
                    "article_ids": [art["id"]],
                    "status": "new",
                    "created_at": iso(now),
                    "updated_at": iso(now),
                    "screening": None,
                    "screening_attempts": 0,
                    "analysis": None,
                    "attempts": 0,
                    "last_error": None,
                    "included_in": None,
                }
            elif art["id"] not in ev["article_ids"]:
                ev["article_ids"].append(art["id"])
                ev["updated_at"] = iso(now)
        log.info("Collected %d new articles into %d events", len(new_dicts), len(set(mapping.values())))
        return reports

    def _excluded(self, item: FeedItem) -> bool:
        path = (urlsplit(item.url).path or "/").lower()
        if not path.endswith("/"):
            path += "/"
        if any(seg in path for seg in self.s.filters.exclude_url_paths):
            return True
        return any(p.search(item.title) for p in self._title_filters)

    def _update_source_health(self, reports: list[SourceReport], now: datetime) -> None:
        health = self.state["source_health"]
        for r in reports:
            h = health.setdefault(r.source_id, {"consecutive_failures": 0})
            h["last_attempt"] = iso(now)
            if r.ok:
                h.update(last_success=iso(now), consecutive_failures=0, last_item_count=r.item_count, last_warnings=r.warnings[:5])
            else:
                h.update(last_failure=iso(now), last_error=self.s.secrets.redact(str(r.error))[:300])
                h["consecutive_failures"] = int(h.get("consecutive_failures", 0)) + 1

    def _source_notices(self) -> list[str]:
        notices = []
        for src in self.sources:
            h = self.state["source_health"].get(src.id, {})
            if src.collectable and h.get("consecutive_failures", 0) >= SOURCE_FAILURE_ALERT:
                notices.append(f"উৎস সমস্যা: {src.name} পরপর {h['consecutive_failures']} বার ব্যর্থ — কনফিগারেশন যাচাই করুন।")
        return notices

    # ------------------------------------------------------------------ Stage C
    def _screen(self, now: datetime) -> None:
        events = [
            ev for ev in self.state["events"].values()
            if (ev["status"] == "new" or self._legacy_screening(ev)) and self._fresh(self._event_time(ev), now)
        ]
        events.sort(key=lambda ev: (self._event_time(ev) or "", ev["event_id"]), reverse=True)
        events = events[: self.s.screening.max_candidates]
        if not events:
            return
        payload = []
        for ev in events:
            primary = self._articles_of(ev)[0]
            payload.append(
                {
                    "event_id": ev["event_id"],
                    "title": primary["title"],
                    "excerpt": primary.get("excerpt", ""),
                    "source_name": ", ".join(sorted({a["source_name"] for a in self._articles_of(ev)})),
                    "published_at": primary.get("published_at"),
                }
            )
        recent = [{k: r[k] for k in ("event_id", "date", "headline", "title")} for r in self._recent_sent(now)]
        outcome = None
        try:
            outcome = screen_events(
                self.llm, payload, self.s.screening.batch_size, self.profile, self.s.selection.maybe_min_priority, recent=recent
            )
        except QuotaExhaustedError as exc:
            self.llm_stop_reason = "quota"
            self._error(f"Screening stopped: {exc}")
            outcome = getattr(exc, "partial", None)
        except PermanentLLMError as exc:
            self.llm_stop_reason = f"permanent: {exc}"
            self._error(f"Screening failed permanently: {exc}")
            outcome = getattr(exc, "partial", None)
        except (InvalidResponseError, TransientLLMError, LLMError) as exc:
            self._error(f"Screening failed: {exc}")
            outcome = getattr(exc, "partial", None)

        decided = outcome.decisions if outcome else {}
        if outcome and outcome.merges:
            self._apply_merges(outcome.merges, decided)
        for ev in events:
            eid = ev["event_id"]
            if eid not in self.state["events"]:
                continue  # merged into another event
            dec = decided.get(eid)
            if dec is None:
                ev["screening_attempts"] = ev.get("screening_attempts", 0) + (0 if self.llm_stop_reason == "quota" else 1)
                self.screening_incomplete = True
                if ev["screening_attempts"] >= MAX_SCREENING_ATTEMPTS:
                    ev["status"] = "failed"
                    ev["last_error"] = "Screening failed repeatedly"
                continue
            ev["screening"] = dec
            ev["status"] = {"select": "selected", "maybe": "maybe", "reject": "rejected"}[dec["decision"]]
            reason = dec.get("forced_reject") or dec["decision"]
            self.screen_stats[reason] = self.screen_stats.get(reason, 0) + 1
            ev["updated_at"] = iso(now)

    def _recent_sent(self, now: datetime) -> list[dict]:
        """Stories delivered in the last RECENT_DAYS days, newest first."""
        tz = get_tz(self.s.edition.timezone)
        cutoff = (now.astimezone(tz).date() - timedelta(days=RECENT_DAYS)).isoformat()
        out: list[dict] = []
        editions = sorted(self.state["editions"].values(), key=lambda e: e.get("delivered_at") or "", reverse=True)
        for ed in editions:
            day = ed.get("edition_date") or ed["date"][:10]
            if ed.get("status") != "delivered" or day < cutoff:
                continue
            for eid in ed.get("story_event_ids") or []:
                ev = self.state["events"].get(eid)
                note = (ev or {}).get("analysis") or {}
                if not note.get("headline_bn"):
                    continue
                arts = self._articles_of(ev)
                out.append(
                    {
                        "event_id": eid,
                        "date": day,
                        "headline": note["headline_bn"],
                        "title": arts[0]["title"] if arts else "",
                        "topic": note.get("syllabus_topic"),
                        "numbers": distinctive_numbers([f["fact"] for f in note.get("key_facts", [])] + [note.get("what_happened", "")]),
                    }
                )
        return out[:MAX_RECENT]

    @staticmethod
    def _legacy_screening(ev: dict) -> bool:
        """Selected/maybe under the pre-syllabus rules: screen again."""
        return ev["status"] in ("selected", "maybe") and "story_type" not in (ev.get("screening") or {})

    def _apply_merges(self, merges: list[tuple[str, str]], decided: dict[str, dict]) -> None:
        events = self.state["events"]
        groups = {
            eid: events[eid]["article_ids"]
            for pair in merges for eid in pair
            if eid in events and (events[eid]["status"] == "new" or self._legacy_screening(events[eid]))
        }
        merged = merge_groups(groups, [(a, b) for a, b in merges if a in groups and b in groups])
        for root, members in merged.items():
            absorbed = [eid for eid in groups if eid != root and set(groups[eid]) <= set(members)]
            if not absorbed:
                continue
            decisions = [decided[e] for e in [root, *absorbed] if e in decided]
            if decisions:
                best = max(decisions, key=lambda d: (DECISION_RANK[d["decision"]], d["priority"]))
                cats = list(dict.fromkeys(c for d in decisions for c in d["categories"]))
                decided[root] = {**best, "categories": cats}
            events[root]["article_ids"] = members
            for eid in absorbed:
                for aid in events[eid]["article_ids"]:
                    if aid in self.state["articles"]:
                        self.state["articles"][aid]["event_id"] = root
                decided.pop(eid, None)
                del events[eid]
            log.info("Model grouped %s into %s", absorbed, root)

    # ------------------------------------------------------------------ Stage D-F
    def _pending_analysis(self, now: datetime) -> list[dict]:
        maxa = self.s.analysis.max_attempts_per_story
        pending = [
            ev for ev in self.state["events"].values()
            if ev["status"] in ("selected", "maybe") and ev.get("attempts", 0) < maxa and self._fresh(self._event_time(ev), now)
        ]
        # Highest expected exam value first; at equal priority, "select" before "maybe".
        pending.sort(key=lambda ev: (-(ev.get("screening") or {}).get("priority", 0), ev["status"] != "selected", ev["event_id"]))
        return pending

    def _unprocessed_count(self, now: datetime) -> int:
        return sum(1 for ev in self._pending_analysis(now) if ev["status"] == "selected")

    def _analysis_input(self, ev: dict) -> AnalysisInput | None:
        excerpt_fallback: AnalysisInput | None = None
        for art in self._articles_of(ev):
            src = self.sources_by_id.get(art["source_id"])
            if src and src.fetch_articles:
                extraction = self.fetch_article(art["url"])
            else:
                extraction = Extraction(status="failed", reason="Article fetching disabled for source")
            art["extraction"] = {"status": extraction.status, "reason": extraction.reason[:200], "chars": len(extraction.text)}
            if extraction.status == "ok":
                art["content_hash"] = sha256(normalize_text(extraction.text))[:16]
            choice = choose_analysis_input(extraction, art.get("excerpt", ""), self.s.extraction.min_excerpt_chars_for_analysis)
            if choice is None:
                continue
            mode, text = choice
            if mode == FULL:
                return AnalysisInput(mode=mode, text=text, article_id=art["id"])
            if excerpt_fallback is None:
                excerpt_fallback = AnalysisInput(mode=mode, text=text, article_id=art["id"])
        return excerpt_fallback

    def _analyse(self, now: datetime) -> None:
        if self.llm_stop_reason:
            return
        budget = self.s.analysis.max_analyses_per_run
        for ev in self._pending_analysis(now)[:budget]:
            if self.llm.quota_exhausted:
                self.llm_stop_reason = self.llm_stop_reason or "quota"
                break
            inp = self._analysis_input(ev)
            if inp is None:
                ev["status"] = "failed"
                ev["last_error"] = "No accessible article text or sufficient excerpt"
                self._save()
                continue
            art = self.state["articles"][inp.article_id]
            ev["attempts"] = ev.get("attempts", 0) + 1
            screening = ev.get("screening") or {}
            followup = {**screening["followup_of"], "new_development": screening.get("new_development", "")} if screening.get("followup_of") else None
            try:
                note = analyse(
                    self.llm, inp, art["title"], art["source_name"], art.get("published_at"), self.s.weights,
                    self.profile, self.s.selection.min_syllabus_relevance, followup=followup,
                )
                if note.get("syllabus_topic"):
                    note["syllabus_label"] = self.profile.topics[note["syllabus_topic"]].label
            except AnalysisRejected as exc:
                ev["status"] = "analysis_rejected"
                ev["last_error"] = str(exc)[:300]
            except QuotaExhaustedError as exc:
                ev["attempts"] -= 1  # not the story's fault
                self.llm_stop_reason = "quota"
                self._error(f"Analysis stopped: {exc}")
                break
            except PermanentLLMError as exc:
                ev["attempts"] -= 1
                self.llm_stop_reason = f"permanent: {exc}"
                self._error(f"Analysis stopped: {exc}")
                break
            except Exception as exc:  # invalid output, exhausted retries, or a reply our checks didn't expect
                ev["last_error"] = self.s.secrets.redact(f"{type(exc).__name__}: {exc}")[:300]
                if ev["attempts"] >= self.s.analysis.max_attempts_per_story:
                    ev["status"] = "failed"
                self._error(f"Analysis of {ev['event_id']} failed: {type(exc).__name__}: {exc}")
            else:
                ev["analysis"] = note
                ev["status"] = "analysed"
                ev["last_error"] = None
                ev["analysed_at"] = iso(now)
            ev["updated_at"] = iso(now)
            self._save()  # checkpoint after every story

    # ------------------------------------------------------------------ Stage E
    def _log_selection(self, stories: list[dict]) -> None:
        """Human-readable selection report in the run log (for tuning)."""
        log.info("Selection report: pre-filtered %d items; screening %s", self.prefiltered, dict(sorted(self.screen_stats.items())))
        rejected = [ev for ev in self.state["events"].values() if ev["status"] == "analysis_rejected" and ev.get("last_error")]
        for ev in rejected[-10:]:
            title = self._articles_of(ev)[0]["title"] if self._articles_of(ev) else ev["event_id"]
            log.info("  analysis rejected: %s :: %s", title[:80], ev["last_error"][:120])
        for st in stories:
            a = st["analysis"]
            log.info("  SELECTED [%s] %s :: %s", a["score"], a.get("syllabus_topic"), a["headline_bn"][:90])

    def _story_for_event(self, ev: dict) -> dict:
        arts = self._articles_of(ev)
        analysed = self.state["articles"].get(ev["analysis"].get("analysed_article_id")) or arts[0]
        links, names = [], []
        for a in [analysed, *[x for x in arts if x is not analysed]]:
            if a["source_name"] not in names and len(links) < MAX_LINKS_PER_STORY:
                names.append(a["source_name"])
                links.append((a["source_name"], a["url"]))
        return {
            "event_id": ev["event_id"],
            "analysis": ev["analysis"],
            "score": ev["analysis"]["score"],
            "published_at": analysed.get("published_at"),
            "date_status": analysed.get("date_status"),
            "source_names": names,
            "links": links,
            "primary_url": analysed["url"],
            "primary_source": analysed["source_name"],
        }

    def _select(self, now: datetime) -> list[dict]:
        recent = self._recent_sent(now)
        stories = []
        for ev in self.state["events"].values():
            if not (
                ev["status"] == "analysed" and ev.get("analysis") and self._articles_of(ev) and self._fresh(self._event_time(ev), now)
                and ev["analysis"].get("syllabus_topic")  # analyses made before the syllabus calibration are ignored
            ):
                continue
            earlier = self._repeat_of_sent(ev, recent)
            if earlier:
                ev["status"] = "analysis_rejected"
                ev["last_error"] = f"Repeat of a story already sent on {earlier['date']}: {earlier['headline'][:80]}"
                log.info("Skipping repeat: %s (sent %s)", ev["analysis"]["headline_bn"][:80], earlier["date"])
                continue
            stories.append(self._story_for_event(ev))
        return rank_stories(stories, self.s.edition.min_story_score, self.s.edition.max_stories, self.s.selection.max_stories_per_topic)

    @staticmethod
    def _repeat_of_sent(ev: dict, recent: list[dict]) -> dict | None:
        """Safety net behind the screening model: same syllabus topic and the
        same distinctive figures as a story already sent. Accepted follow-ups
        (screening named a new development) are never blocked."""
        if (ev.get("screening") or {}).get("followup_of"):
            return None
        note = ev["analysis"]
        numbers = distinctive_numbers([f["fact"] for f in note.get("key_facts", [])] + [note.get("what_happened", "")])
        for r in recent:
            if r["event_id"] != ev["event_id"] and r["topic"] == note.get("syllabus_topic") and same_story(numbers, note["headline_bn"], r["numbers"], r["headline"]):
                return r
        return None

    # ------------------------------------------------------------------ Stage F2
    def _enrich(self, stories: list[dict]) -> None:
        """Related static GK for the stories being sent (optional extra: two
        calls, the second a blind check). Never blocks or delays the edition."""
        todo = [st for st in stories if "static_gk" not in st["analysis"]]
        if not todo or self.llm_stop_reason or self.llm.quota_exhausted:
            return
        if getattr(self.llm, "remaining_calls", 3) < 3:  # keep one call for the MCQs
            log.info("Static GK skipped: model call budget nearly used")
            return
        try:
            outcome = enrich_static_gk(self.llm, todo)
        except Exception as exc:  # an optional section must never cost the edition
            self._error(f"Static GK skipped: {type(exc).__name__}: {exc}")
            return
        for st in todo:
            st["analysis"]["static_gk"] = outcome.items.get(st["event_id"], [])

    # ------------------------------------------------------------------ Stage G
    def _questions(self, stories: list[dict]) -> tuple[list[dict], str | None]:
        count = self.s.edition.mcq_count
        if not stories or count <= 0:
            return [], None
        if self.llm_stop_reason or self.llm.quota_exhausted:
            return [], "AI কোটা/ত্রুটির কারণে MCQ তৈরি করা যায়নি।"
        try:
            outcome = generate_mcqs(self.llm, stories, count)
        except QuotaExhaustedError as exc:
            self.llm_stop_reason = "quota"
            self._error(f"MCQ generation stopped: {exc}")
            return [], "AI কোটা শেষ হওয়ায় MCQ তৈরি করা যায়নি।"
        except Exception as exc:  # LLMError, or a reply our checks didn't expect
            self._error(f"MCQ generation failed: {type(exc).__name__}: {exc}")
            return [], "MCQ তৈরিতে ত্রুটি হয়েছে।"
        return outcome.questions, None

    # ------------------------------------------------------------------ delivery
    def _write_output(self, key: str, chunks: list[str], edition: dict) -> Path | None:
        try:
            out_dir = self.s.output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"edition-{key}{'-dry-run' if self.dry_run else ''}.txt"
            body = "\n\n".join(f"===== message {i + 1}/{len(chunks)} =====\n{c}" for i, c in enumerate(chunks))
            path.write_text(body + "\n", encoding="utf-8")
            meta = {k: v for k, v in edition.items() if k != "chunks"}
            (out_dir / f"edition-{key}{'-dry-run' if self.dry_run else ''}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            return path
        except OSError as exc:
            self._error(f"Could not write edition output: {exc}")
            return None

    def _pdf_path(self, key: str) -> Path:
        return self.s.output_dir / f"edition-{key}{'-dry-run' if self.dry_run else ''}.pdf"

    def _build_pdf(self, edition: dict, ed_date: date, stories: list[dict], mcqs: list[dict], partial_reasons: list[str], extra: bool) -> bytes | None:
        """Render the edition as a PDF. Best effort: never blocks the text edition."""
        key = edition["date"]
        if self.pdf_builder is None:
            return None
        try:
            html = render_pdf_html(ed_date, stories, mcqs, partial_reasons, self.sources, extra=extra)
            pdf = self.pdf_builder(html)
            if not pdf.startswith(b"%PDF"):
                raise ValueError("builder did not return a PDF")
            path = self._pdf_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pdf)
        except Exception as exc:  # PDF problems must never stop delivery
            edition["pdf_error"] = self.s.secrets.redact(f"{type(exc).__name__}: {exc}")[:300]
            self._error(f"PDF could not be created; sending text only: {exc}")
            self._pdf_cache[key] = None
            return None
        edition["pdf"] = {"file": path.name, "bytes": len(pdf), "sha": hashlib.sha256(pdf).hexdigest()[:16]}
        edition.pop("pdf_error", None)
        self._pdf_cache[key] = pdf
        return pdf

    def _pdf_for(self, edition: dict) -> bytes | None:
        """PDF for delivery: from this run, from disk, or rebuilt from state
        (a resumed delivery may run on a fresh machine without the file)."""
        key = edition["date"]
        if key in self._pdf_cache:
            return self._pdf_cache[key]
        if not edition.get("pdf") or not edition.get("story_event_ids"):
            return None
        path = self._pdf_path(key)
        if path.exists():
            self._pdf_cache[key] = path.read_bytes()
            return self._pdf_cache[key]
        events = [self.state["events"].get(eid) for eid in edition["story_event_ids"]]
        if not all(ev and ev.get("analysis") and self._articles_of(ev) for ev in events):
            self._pdf_cache[key] = None
            return None
        stories = [self._story_for_event(ev) for ev in events]
        ed_date = date.fromisoformat(edition.get("edition_date", key[:10]))
        return self._build_pdf(edition, ed_date, stories, edition.get("mcqs", []), edition.get("partial_reasons", []), bool(edition.get("manual")))

    def _send_pdf(self, sub: dict, record: dict, edition: dict, pdf: bytes, now: datetime) -> bool:
        """Send the PDF to one subscriber. Returns False if delivery must stop."""
        ed_date = date.fromisoformat(edition.get("edition_date", edition["date"][:10]))
        suffix = f"-extra{edition['date'].rsplit('-m', 1)[-1]}" if edition.get("manual") else ""
        filename = f"BCS-Briefing-{ed_date.isoformat()}{suffix}.pdf"
        caption = f"📄 আজকের ব্রিফিং (PDF) — {bn_date(ed_date)}"
        try:
            sent = self.telegram.send_document(sub["chat_id"], pdf, filename, caption)
        except TelegramError as exc:
            record["last_error"] = self.s.secrets.redact(str(exc))[:300]
            if exc.recipient_gone:
                deactivate(self.state, sub["chat_id"], "unreachable (blocked bot or left)", now)
                record["status"] = "recipient_gone"
            else:
                self._error(f"PDF delivery to subscriber {sub.get('name', '?')} failed: {exc}")
            self._save()
            return False
        record["pdf_message_id"] = sent.message_id
        self._save()
        return True

    def _deliver(self, edition: dict) -> RunResult:
        """Send the stored chunks to every active subscriber.

        Each subscriber's progress is saved after every confirmed message, so
        an interrupted run resumes exactly where it stopped. One subscriber's
        failure never blocks the others.
        """
        key = edition["date"]
        result = RunResult(
            outcome="failed",
            edition_date=key,
            stories=len(edition.get("story_event_ids", [])),
            mcqs=len(edition.get("mcqs", [])),
            partial=bool(edition.get("partial")),
        )
        if self.telegram is None:
            self._error("Telegram is not configured; edition kept for a later delivery attempt")
            return result
        recipients = active_subscribers(self.state)
        if not recipients:
            self._error("No subscribers yet: share the bot's invite link and have people press Start. Edition kept for later today.")
            result.outcome = "no_subscribers"
            return result

        now = utcnow()
        edition["status"] = "delivering"
        deliveries = edition.setdefault("deliveries", {})
        chunks = sorted(edition["chunks"], key=lambda c: c["index"])
        pdf = self._pdf_for(edition)
        for sub in recipients:
            cid = sub["chat_id"]
            record = deliveries.setdefault(cid, {"sent": {}, "status": "pending", "attempts": 0})
            if record["status"] == "done":
                continue
            record["attempts"] += 1
            for chunk in chunks:
                idx = str(chunk["index"])
                if idx in record["sent"]:
                    continue
                try:
                    sent = self.telegram.send_message(cid, chunk["text"])
                except TelegramError as exc:
                    record["last_error"] = self.s.secrets.redact(str(exc))[:300]
                    if exc.recipient_gone:
                        deactivate(self.state, cid, "unreachable (blocked bot or left)", now)
                        record["status"] = "recipient_gone"
                    else:
                        self._error(f"Delivery to subscriber {sub.get('name', '?')} stopped at message {chunk['index'] + 1}: {exc}")
                    self._save()
                    break
                record["sent"][idx] = sent.message_id
                self._save()  # checkpoint after each confirmed message
            else:
                # All text messages arrived; the PDF comes last.
                if pdf and not record.get("pdf_message_id") and not self._send_pdf(sub, record, edition, pdf, now):
                    continue
                record["status"] = "done"
                record["delivered_at"] = iso(utcnow())
                self._save()

        still_active = {s["chat_id"] for s in active_subscribers(self.state)}
        pending = [cid for cid in still_active if deliveries.get(cid, {}).get("status") != "done"]
        delivered_to = [cid for cid, r in deliveries.items() if r.get("status") == "done"]
        if pending:
            edition["last_delivery_error"] = f"{len(pending)} subscriber(s) not yet delivered"
            self._save()
            return result  # stays "delivering"; the next run resends only what is missing
        if not delivered_to:
            edition["status"] = "ready"
            self._error("Every subscriber was unreachable; edition not delivered")
            self._save()
            result.outcome = "no_subscribers"
            return result

        edition["status"] = "delivered"
        edition["delivered_at"] = iso(utcnow())
        edition["recipients"] = len(delivered_to)
        for chunk in edition["chunks"]:
            chunk.pop("text", None)  # rendered copy lives in the output artifact
        for eid in edition.get("story_event_ids", []):
            if eid in self.state["events"]:
                self.state["events"][eid]["status"] = "included"
                self.state["events"][eid]["included_in"] = key
        self.state["last_successful_edition"] = key
        self._save()
        result.outcome = "delivered"
        return result

    def _fail(self, edition: dict, ed_date: date, reason: str, unprocessed: int) -> RunResult:
        key = edition["date"]
        edition.update(status="failed", failure_reason=reason, unprocessed_candidates=unprocessed)
        result = RunResult(outcome="failed", edition_date=key, partial=True)
        notice = render_failure_notice(ed_date, reason, unprocessed)
        if self.dry_run:
            self._write_output(key, [notice], edition)
            result.outcome = "dry_run"
            return result
        if edition.get("failure_notified"):
            log.info("Failure already notified for %s", key)
            result.outcome = "failure_notified"
            return result
        recipients = active_subscribers(self.state) if self.telegram is not None else []
        if not recipients:
            self._error(f"Edition failed ({reason}) and there is nobody to notify")
            return result
        delivered = 0
        for sub in recipients:
            try:
                self.telegram.send_message(sub["chat_id"], notice)
                delivered += 1
            except TelegramError as exc:
                if exc.recipient_gone:
                    deactivate(self.state, sub["chat_id"], "unreachable (blocked bot or left)", utcnow())
                else:
                    self._error(f"Could not send failure notice to {sub.get('name', '?')}: {exc}")
        if delivered:
            edition["failure_notified"] = True
            result.outcome = "failure_notified"
        return result
