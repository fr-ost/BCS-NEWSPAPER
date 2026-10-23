"""Daily edition pipeline: collect → group → screen → analyse → rank → MCQ →
render → deliver, with a state checkpoint after every stage and story.

Idempotency rules:
- One edition per Asia/Dhaka date. A delivered edition is never re-sent.
- An edition is rendered once and its chunks stored before sending; a retry
  resends only chunks without a recorded Telegram message id.
- Completed analyses are stored and never recomputed.
- Failed stories are retried at most ``max_attempts_per_story`` times;
  quota exhaustion does not count as an attempt.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from .analysis import AnalysisInput, AnalysisRejected, analyse
from .collector import FeedItem, SourceReport, collect, load_fixture_items
from .config import Settings
from .deduplicator import assign_events, merge_groups
from .extractor import FULL, Extraction, choose_analysis_input
from .llm import (
    InvalidResponseError,
    LLMClient,
    LLMError,
    PermanentLLMError,
    QuotaExhaustedError,
    TransientLLMError,
)
from .questions import generate_mcqs
from .render import render_edition, render_failure_notice
from .scoring import rank_stories
from .screening import screen_events
from .sources import Source
from .state import StateStore
from .telegram_delivery import TelegramClient, TelegramError, chunk_sections
from .util import get_tz, iso, normalize_text, parse_iso, sha256, utcnow

log = logging.getLogger(__name__)

DECISION_RANK = {"select": 2, "maybe": 1, "reject": 0}
MAX_SCREENING_ATTEMPTS = 3
MAX_LINKS_PER_STORY = 4
SOURCE_FAILURE_ALERT = 3

ArticleFetcher = Callable[[str], Extraction]


@dataclass
class RunResult:
    outcome: str
    edition_date: str
    stories: int = 0
    mcqs: int = 0
    partial: bool = False
    llm_calls: int = 0
    errors: list[str] = field(default_factory=list)
    output_path: Path | None = None

    @property
    def ok(self) -> bool:
        return self.outcome in {"delivered", "already_delivered", "dry_run", "failure_notified", "no_change"}


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        sources: list[Source],
        store: StateStore,
        llm: LLMClient,
        article_fetcher: ArticleFetcher,
        feed_fetcher=None,
        telegram: TelegramClient | None = None,
        dry_run: bool = False,
        offline_feed_dir: Path | None = None,
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
        self.state: dict = {}
        self.errors: list[str] = []
        self.llm_stop_reason: str | None = None  # "quota" | "permanent: ..." when the model became unusable
        self.screening_incomplete = False

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
    def run(self, now: datetime | None = None, edition_date: date | None = None) -> RunResult:
        now = now or utcnow()
        tz = get_tz(self.s.edition.timezone)
        ed_date = edition_date or now.astimezone(tz).date()
        key = ed_date.isoformat()
        self.state = self.store.load()
        run_record = {"run_id": uuid.uuid4().hex[:10], "started_at": iso(now), "edition_date": key, "dry_run": self.dry_run}
        if self.store.recovered:
            self._error("State had to be recovered or reset; duplicate protection may be weaker for this run")
        result = RunResult(outcome="failed", edition_date=key)
        try:
            result = self._run(now, ed_date, key)
        except Exception as exc:
            self._error(f"Unexpected error: {type(exc).__name__}: {exc}")
            result = RunResult(outcome="failed", edition_date=key)
            raise
        finally:
            result.errors = self.errors
            result.llm_calls = self.llm.calls_made
            run_record.update(
                finished_at=iso(utcnow()),
                outcome=result.outcome,
                stories=result.stories,
                mcqs=result.mcqs,
                partial=result.partial,
                llm_calls=self.llm.calls_made,
                errors=self.errors[-20:],
            )
            self.state.setdefault("runs", []).append(run_record)
            self.store.prune(self.state, now)
            self._save()
        return result

    def _run(self, now: datetime, ed_date: date, key: str) -> RunResult:
        edition = self.state["editions"].get(key)
        if edition and edition["status"] == "delivered":
            log.info("Edition %s already delivered; nothing to do", key)
            return RunResult(outcome="already_delivered", edition_date=key, stories=len(edition.get("story_event_ids", [])))
        if edition and edition["status"] in ("ready", "delivering") and edition.get("chunks"):
            if self.dry_run:
                return RunResult(outcome="dry_run", edition_date=key, stories=len(edition.get("story_event_ids", [])))
            log.info("Resuming delivery of edition %s", key)
            return self._deliver(edition)

        previous = edition or {}
        edition = {
            "date": key,
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

        sections = render_edition(
            ed_date,
            stories,
            mcqs,
            partial_reasons,
            unprocessed if partial_reasons else 0,
            self.sources,
            self.s.edition.mcq_count,
            notices=self._source_notices(),
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
        self._save()

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
            new_items.append(item)
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
            if ev["status"] == "new" and self._fresh(self._event_time(ev), now)
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
        outcome = None
        try:
            outcome = screen_events(self.llm, payload, self.s.screening.batch_size)
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
            ev["updated_at"] = iso(now)

    def _apply_merges(self, merges: list[tuple[str, str]], decided: dict[str, dict]) -> None:
        events = self.state["events"]
        groups = {eid: events[eid]["article_ids"] for pair in merges for eid in pair if eid in events and events[eid]["status"] == "new"}
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
        pending.sort(key=lambda ev: (ev["status"] != "selected", -(ev.get("screening") or {}).get("priority", 0), ev["event_id"]))
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
            try:
                note = analyse(self.llm, inp, art["title"], art["source_name"], art.get("published_at"), self.s.weights)
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
            except LLMError as exc:  # invalid output or exhausted transient retries
                ev["last_error"] = self.s.secrets.redact(f"{type(exc).__name__}: {exc}")[:300]
                if ev["attempts"] >= self.s.analysis.max_attempts_per_story:
                    ev["status"] = "failed"
                self._error(f"Analysis of {ev['event_id']} failed: {exc}")
            else:
                ev["analysis"] = note
                ev["status"] = "analysed"
                ev["last_error"] = None
                ev["analysed_at"] = iso(now)
            ev["updated_at"] = iso(now)
            self._save()  # checkpoint after every story

    # ------------------------------------------------------------------ Stage E
    def _select(self, now: datetime) -> list[dict]:
        stories = []
        for ev in self.state["events"].values():
            if ev["status"] != "analysed" or not ev.get("analysis") or not self._fresh(self._event_time(ev), now):
                continue
            arts = self._articles_of(ev)
            analysed = self.state["articles"].get(ev["analysis"].get("analysed_article_id")) or arts[0]
            links, names = [], []
            for a in [analysed, *[x for x in arts if x is not analysed]]:
                if a["source_name"] not in names and len(links) < MAX_LINKS_PER_STORY:
                    names.append(a["source_name"])
                    links.append((a["source_name"], a["url"]))
            stories.append(
                {
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
            )
        return rank_stories(stories, self.s.edition.min_story_score, self.s.edition.max_stories)

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
        except LLMError as exc:
            self._error(f"MCQ generation failed: {exc}")
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

    def _deliver(self, edition: dict) -> RunResult:
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
        edition["status"] = "delivering"
        for chunk in sorted(edition["chunks"], key=lambda c: c["index"]):
            if chunk["status"] == "sent":
                continue
            chunk["attempts"] = chunk.get("attempts", 0) + 1
            try:
                sent = self.telegram.send_message(chunk["text"])
            except TelegramError as exc:
                edition["last_delivery_error"] = self.s.secrets.redact(str(exc))[:300]
                self._error(f"Delivery of chunk {chunk['index'] + 1} failed: {exc}")
                self._save()
                return result
            chunk.update(status="sent", message_id=sent.message_id, sent_at=iso(utcnow()), plain_text_fallback=sent.fallback_plain)
            self._save()  # checkpoint after each confirmed chunk
        edition["status"] = "delivered"
        edition["delivered_at"] = iso(utcnow())
        edition["message_ids"] = [c["message_id"] for c in edition["chunks"]]
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
        if self.dry_run:
            self._write_output(key, [render_failure_notice(ed_date, reason, unprocessed)], edition)
            result.outcome = "dry_run"
            return result
        if edition.get("failure_notified"):
            log.info("Failure already notified for %s", key)
            result.outcome = "failure_notified"
            return result
        if self.telegram is None:
            self._error(f"Edition failed ({reason}) and Telegram is not configured")
            return result
        try:
            self.telegram.send_message(render_failure_notice(ed_date, reason, unprocessed))
            edition["failure_notified"] = True
            result.outcome = "failure_notified"
        except TelegramError as exc:
            self._error(f"Could not send failure notice: {exc}")
        return result
