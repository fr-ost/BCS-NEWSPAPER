"""Related static general knowledge ("প্রাসঙ্গিক সাধারণ জ্ঞান") for sent stories.

This is NOT taken from the article. BCS candidates need each news item tied
to the textbook facts BPSC actually asks (which amendment introduced a system,
when an organisation was founded, where it is headquartered, what the criteria
of a UN process are). The facts come from the model's general knowledge, so:

1. One batched call proposes at most ``MAX_PER_STORY`` short, durable facts
   per story, each with a question and a short answer (Bengali + English).
2. A second, *blind* call answers every question without seeing the claimed
   answer. Code compares the two: numbers must match exactly, names must
   agree. Anything that disagrees, or that the checker is unsure of, is
   dropped. An empty section is fine; a wrong fact is not.
3. Survivors are shown under their own label ("not from the article; check
   before relying on it") and never mixed with article-derived facts.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

from .deduplicator import distinctive_numbers
from .grounding import clean_model_text
from .llm import InvalidResponseError, LLMClient, LLMRequest
from .prompts import GK_SYSTEM, GK_VERIFY_SYSTEM, gk_prompt, gk_verify_prompt
from .util import extract_numbers, normalize_text

log = logging.getLogger(__name__)

MAX_PER_STORY = 3
MAX_ITEMS = 24
KINDS = ("year", "number", "name", "place", "institution", "law", "other")
# Hedged statements are not facts worth memorising.
HEDGES = ("সম্ভবত", "সম্ভাব্য", "ধারণা করা", "মনে করা হয়", "আনুমানিক", "প্রায়", "বলে জানা")
_HEDGES_EN = re.compile(r"\b(probably|approximately|around|about|roughly|believed|reportedly)\b", re.I)
_LATIN_WORD = re.compile(r"[a-z0-9]+")
_NAME_STOPWORDS = {"the", "of", "and", "a", "an", "in", "on", "for", "to", "at", "by", "de", "la"}


@dataclass
class GkOutcome:
    items: dict[str, list[dict]] = field(default_factory=dict)  # event_id -> verified items
    proposed: int = 0
    rejected: list[str] = field(default_factory=list)


def _story_payload(stories: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    shown, smap = [], {}
    for i, story in enumerate(stories, start=1):
        sid = f"s{i}"
        note = story["analysis"]
        smap[sid] = story
        shown.append(
            {
                "id": sid,
                "headline": note["headline_bn"],
                "syllabus": note.get("syllabus_label") or note.get("syllabus_topic") or "",
                "summary": note.get("what_happened", ""),
                "facts": [f["fact"] for f in note.get("key_facts", [])][:5],
            }
        )
    return shown, smap


def _contains(haystack: str, needle: str) -> bool:
    return bool(needle) and normalize_text(needle) in normalize_text(haystack)


def validate_gk(raw: object, smap: dict[str, dict]) -> tuple[list[dict], list[str]]:
    """Shape and content rules for proposed items (before the blind check)."""
    if isinstance(raw, list):
        raw = {"items": raw}
    if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
        raise InvalidResponseError("Static GK response lacks an 'items' list")
    kept, reasons, seen = [], [], set()
    per_story: dict[str, int] = {}
    for item in raw["items"]:
        if len(kept) >= MAX_ITEMS:
            break
        if not isinstance(item, dict) or item.get("story") not in smap:
            reasons.append("unknown story")
            continue
        sid = item["story"]
        statement = clean_model_text(item.get("statement"), 240)
        question = clean_model_text(item.get("question"), 220)
        answer = clean_model_text(item.get("answer"), 80)
        answer_en = clean_model_text(item.get("answer_en"), 80)
        kind = item.get("kind") if item.get("kind") in KINDS else "other"
        if not (statement and question and answer and answer_en):
            reasons.append("incomplete item")
            continue
        if any(h in statement for h in HEDGES) or _HEDGES_EN.search(statement):
            reasons.append("hedged statement")
            continue
        if not (_contains(statement, answer) or (extract_numbers(answer) and extract_numbers(answer) <= extract_numbers(statement))):
            reasons.append("statement does not state the answer")
            continue
        if _contains(question, answer) and len(normalize_text(answer)) >= 3:
            reasons.append("question gives the answer away")
            continue
        note = smap[sid]["analysis"]
        article_figures = distinctive_numbers([f["fact"] for f in note.get("key_facts", [])] + [note.get("what_happened", "")])
        if distinctive_numbers([statement]) & article_figures:
            reasons.append("restates the article, not background")
            continue
        key = normalize_text(statement)
        if key in seen or per_story.get(sid, 0) >= MAX_PER_STORY:
            reasons.append("duplicate or too many for the story")
            continue
        seen.add(key)
        per_story[sid] = per_story.get(sid, 0) + 1
        kept.append({"story": sid, "statement": statement, "question": question, "answer": answer, "answer_en": answer_en, "kind": kind})
    return kept, reasons


def _latin_tokens(text: str) -> set[str]:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return {t for t in _LATIN_WORD.findall(text) if t not in _NAME_STOPWORDS}


def answers_agree(claimed_en: str, claimed_bn: str, independent: str) -> bool:
    """Do two short answers state the same fact? Numbers must match exactly
    (as a set, Bengali digits normalised); names must share most words."""
    claimed_numbers = extract_numbers(claimed_en) | extract_numbers(claimed_bn)
    got_numbers = extract_numbers(independent)
    if claimed_numbers or got_numbers:
        return claimed_numbers == got_numbers
    a, b = _latin_tokens(claimed_en), _latin_tokens(independent)
    if not a or not b:
        return normalize_text(claimed_bn) == normalize_text(independent)
    return a <= b or b <= a or len(a & b) / len(a | b) >= 0.5


def verify_gk(raw: object, items: list[dict]) -> tuple[list[dict], list[str]]:
    """Keep items whose blind answer agrees and is marked sure."""
    if isinstance(raw, list):
        raw = {"answers": raw}
    if not isinstance(raw, dict) or not isinstance(raw.get("answers"), list):
        raise InvalidResponseError("Static GK check lacks an 'answers' list")
    answers = {}
    for a in raw["answers"]:
        if isinstance(a, dict) and isinstance(a.get("id"), str):
            answers[a["id"]] = a
    kept, reasons = [], []
    for i, item in enumerate(items, start=1):
        got = answers.get(f"g{i}")
        if got is None or got.get("sure") is not True:
            reasons.append("checker unsure")
            continue
        independent = clean_model_text(got.get("answer_en"), 80)
        if not answers_agree(item["answer_en"], item["answer"], independent):
            reasons.append(f"checker disagrees ({item['answer_en']!r} vs {independent!r})")
            continue
        kept.append(item)
    return kept, reasons


def enrich_static_gk(client: LLMClient, stories: list[dict]) -> GkOutcome:
    """Propose and blind-check related static GK for ``stories``.
    Raises LLMError subclasses like any model call; callers treat the
    section as optional."""
    out = GkOutcome()
    if not stories:
        return out
    shown, smap = _story_payload(stories)
    raw = client.generate_json(
        LLMRequest(task="static_gk", system=GK_SYSTEM, prompt=gk_prompt(shown, MAX_PER_STORY), payload={"stories": shown})
    )
    proposed, reasons = validate_gk(raw, smap)
    out.proposed = len(proposed)
    out.rejected.extend(reasons)
    if not proposed:
        return out
    questions = [{"id": f"g{i}", "question": item["question"]} for i, item in enumerate(proposed, start=1)]
    checked = client.generate_json(
        LLMRequest(task="static_gk_check", system=GK_VERIFY_SYSTEM, prompt=gk_verify_prompt(questions), payload={"questions": questions})
    )
    verified, reasons = verify_gk(checked, proposed)
    out.rejected.extend(reasons)
    for item in verified:
        story = smap[item["story"]]
        out.items.setdefault(story["event_id"], []).append({k: item[k] for k in ("statement", "question", "answer", "answer_en", "kind")})
    if out.rejected:
        log.info("Static GK: kept %d of %d proposed; dropped: %s", len(verified), out.proposed, "; ".join(out.rejected[:8]))
    return out
