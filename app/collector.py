"""Stage A: collect recent items from enabled RSS/Atom feeds."""

from __future__ import annotations

import calendar
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

import feedparser
from bs4 import BeautifulSoup

from .http import FetchError, Fetcher, RobotsDisallowed
from .sources import Source
from .util import canonicalize_url, iso, is_valid_http_url, normalize_text, short_hash

log = logging.getLogger(__name__)

MAX_EXCERPT_CHARS = 1500
MAX_TITLE_CHARS = 400


@dataclass
class FeedItem:
    id: str
    source_id: str
    source_name: str
    title: str
    url: str
    canonical_url: str
    excerpt: str
    published_at: str | None
    retrieved_at: str
    date_status: str  # "ok" | "missing" | "invalid" | "future"
    language: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SourceReport:
    source_id: str
    ok: bool
    item_count: int = 0
    skipped: int = 0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


def html_to_text(value: str) -> str:
    if not value:
        return ""
    if "<" not in value and "&" not in value:
        return " ".join(value.split())
    text = BeautifulSoup(value, "html.parser").get_text(" ")
    return " ".join(text.split())


def article_id_for(canonical_url: str) -> str:
    return "a_" + short_hash(canonical_url)


def _entry_date(entry, now: datetime) -> tuple[datetime | None, str]:
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed = entry.get(key)
        if parsed:
            try:
                dt = datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)
            except (TypeError, ValueError, OverflowError):
                return None, "invalid"
            if dt > now + timedelta(hours=2):
                return None, "future"
            return dt, "ok"
    for key in ("published", "updated"):
        if entry.get(key):
            return None, "invalid"  # present but unparseable
    return None, "missing"


def parse_feed(content: bytes, source: Source, now: datetime, max_items: int) -> tuple[list[FeedItem], SourceReport]:
    """Parse feed bytes into items. Never raises on malformed entries."""
    report = SourceReport(source_id=source.id, ok=True)
    parsed = feedparser.parse(content)
    entries = list(parsed.get("entries") or [])
    if not entries:
        if parsed.get("bozo"):
            exc = parsed.get("bozo_exception")
            report.ok = False
            report.error = f"Malformed feed: {type(exc).__name__ if exc else 'unknown error'}"
        else:
            report.warnings.append("Feed contained no entries")
        return [], report
    if parsed.get("bozo"):
        report.warnings.append("Feed is not well-formed; parsed leniently")

    items: list[FeedItem] = []
    seen: set[str] = set()
    retrieved = iso(now)
    for entry in entries[:max_items]:
        try:
            title = html_to_text(str(entry.get("title") or ""))[:MAX_TITLE_CHARS]
            link = str(entry.get("link") or "").strip()
            if not title or not is_valid_http_url(link):
                report.skipped += 1
                continue
            canonical = canonicalize_url(link)
            if canonical in seen:
                report.skipped += 1
                continue
            seen.add(canonical)
            summary = entry.get("summary") or entry.get("description") or ""
            excerpt = html_to_text(str(summary))[:MAX_EXCERPT_CHARS]
            if normalize_text(excerpt) == normalize_text(title):
                excerpt = ""
            published, date_status = _entry_date(entry, now)
            items.append(
                FeedItem(
                    id=article_id_for(canonical),
                    source_id=source.id,
                    source_name=source.name,
                    title=title,
                    url=link,
                    canonical_url=canonical,
                    excerpt=excerpt,
                    published_at=iso(published),
                    retrieved_at=retrieved,
                    date_status=date_status,
                    language=source.language,
                )
            )
        except Exception as exc:  # one bad entry must not sink the feed
            report.skipped += 1
            report.warnings.append(f"Skipped malformed entry: {type(exc).__name__}")
    report.item_count = len(items)
    return items, report


def collect(sources: list[Source], fetcher: Fetcher, now: datetime, max_items: int, max_bytes: int) -> tuple[list[FeedItem], list[SourceReport]]:
    """Fetch every enabled source. A failing source never stops the others."""
    all_items: list[FeedItem] = []
    reports: list[SourceReport] = []
    for source in sources:
        if not source.collectable:
            continue
        try:
            result = fetcher.get(
                source.feed_url,
                max_bytes=max_bytes,
                timeout=source.timeout_seconds,
                retries=source.max_retries,
                accept="application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.1",
            )
            items, report = parse_feed(result.content, source, now, max_items)
        except RobotsDisallowed as exc:
            items, report = [], SourceReport(source.id, ok=False, error=str(exc))
        except FetchError as exc:
            items, report = [], SourceReport(source.id, ok=False, error=str(exc))
        except Exception as exc:  # defensive: never let one source crash collection
            items, report = [], SourceReport(source.id, ok=False, error=f"Unexpected {type(exc).__name__}")
        log.info("Source %s: ok=%s items=%d skipped=%d error=%s", source.id, report.ok, report.item_count, report.skipped, report.error)
        all_items.extend(items)
        reports.append(report)
    return all_items, reports


def load_fixture_items(sources: list[Source], feed_dir, now: datetime, max_items: int) -> tuple[list[FeedItem], list[SourceReport]]:
    """Offline mode: parse ``<feed_dir>/<source_id>.xml`` instead of fetching."""
    items: list[FeedItem] = []
    reports: list[SourceReport] = []
    for source in sources:
        path = feed_dir / f"{source.id}.xml"
        if not path.exists():
            continue
        parsed, report = parse_feed(path.read_bytes(), source, now, max_items)
        items.extend(parsed)
        reports.append(report)
    return items, reports
