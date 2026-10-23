"""Stage D (fetch part): retrieve and clean article bodies.

Extraction is best-effort. Paywalled, blocked or script-rendered pages are
not worked around; the story falls back to a clearly labelled excerpt-only
analysis or is skipped.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

from .http import FetchError, Fetcher, RobotsDisallowed
from .util import truncate

log = logging.getLogger(__name__)

FULL = "full"
EXCERPT_ONLY = "excerpt_only"

_DROP_TAGS = ("script", "style", "noscript", "iframe", "svg", "form", "nav", "header", "footer", "aside", "button", "template", "figure", "video", "audio", "select", "input")
_NOISE = re.compile(
    r"(^|[\s_-])(related|recommend\w*|share|sharing|social|advert\w*|ads?|sponsor\w*|promo\w*|newsletter|subscribe|"
    r"comments?|breadcrumbs?|tags?|author-bio|sidebar|popular|trending|more-news|read-?more|also-?read|"
    r"cookie\w*|banner|outbrain|taboola|print-only)($|[\s_-])",
    re.I,
)
_BODY_SELECTORS = (
    "[itemprop=articleBody]",
    "article",
    "[class*=article-body]",
    "[class*=story-content]",
    "[class*=story-element]",
    "[class*=article-content]",
    "[class*=content-details]",
    "[class*=post-content]",
    "main",
)
_BOILERPLATE_LINE = re.compile(
    r"^(also read|read more|আরও পড়ুন|আরো পড়ুন|আরও পড়ুন|advertisement|বিজ্ঞাপন|follow us|subscribe|share this)\b",
    re.I,
)


@dataclass
class Extraction:
    status: str  # "ok" | "too_short" | "failed" | "blocked"
    text: str = ""
    reason: str = ""


def _is_noise(tag: Tag) -> bool:
    if tag.name in ("html", "body", "main", "article"):
        return False
    attrs = " ".join(tag.get("class") or []) + " " + str(tag.get("id") or "")
    if not attrs.strip() or not _NOISE.search(attrs):
        return False
    # Never drop a wrapper that contains the article body itself.
    return tag.select_one("[itemprop=articleBody]") is None


def clean_article_html(html: str | bytes, max_chars: int) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for name in _DROP_TAGS:
        for tag in soup.find_all(name):
            tag.decompose()
    for tag in list(soup.find_all(True)):
        if tag.decomposed:
            continue
        if _is_noise(tag):
            tag.decompose()

    best_text = ""
    containers = []
    for selector in _BODY_SELECTORS:
        containers.extend(soup.select(selector))
    if soup.body:
        containers.append(soup.body)
    for container in containers:
        paragraphs = _paragraphs(container)
        text = "\n".join(paragraphs)
        if len(text) > len(best_text):
            best_text = text
    return truncate(best_text, max_chars)


def _paragraphs(container: Tag) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for p in container.find_all(["p", "h2", "h3", "li"]):
        text = " ".join(p.get_text(" ").split())
        if len(text) < 30 or _BOILERPLATE_LINE.match(text):
            continue
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def extract_article(fetcher: Fetcher, url: str, max_bytes: int, min_chars: int, max_chars: int) -> Extraction:
    try:
        result = fetcher.get(url, max_bytes=max_bytes, accept="text/html,application/xhtml+xml;q=0.9,*/*;q=0.1")
    except RobotsDisallowed as exc:
        return Extraction(status="blocked", reason=str(exc))
    except FetchError as exc:
        return Extraction(status="failed", reason=str(exc))
    ctype = result.content_type.lower()
    if ctype and "html" not in ctype and "xml" not in ctype:
        return Extraction(status="failed", reason=f"Unexpected content type {ctype}")
    try:
        text = clean_article_html(result.content, max_chars)
    except Exception as exc:  # malformed HTML must not crash the run
        return Extraction(status="failed", reason=f"Parse error: {type(exc).__name__}")
    if len(text) < min_chars:
        return Extraction(status="too_short", text=text, reason=f"Only {len(text)} chars extracted")
    return Extraction(status="ok", text=text)


def choose_analysis_input(extraction: Extraction, excerpt: str, min_excerpt_chars: int) -> tuple[str, str] | None:
    """Decide what the model may analyse.

    Returns ``(mode, text)`` where mode is FULL or EXCERPT_ONLY, or ``None``
    when there is not enough evidence to analyse the story at all.
    """
    if extraction.status == "ok":
        return FULL, extraction.text
    # A short partial body (e.g. a paywall teaser) is still only an excerpt.
    candidates = [t for t in (extraction.text, excerpt) if t]
    best = max(candidates, key=len, default="")
    if len(best) >= min_excerpt_chars:
        return EXCERPT_ONLY, best
    return None
