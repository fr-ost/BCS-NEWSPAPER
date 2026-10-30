"""Stages D-F: fetch article text, analyse it, validate and ground the note."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import SUBJECTS
from .grounding import clean_model_text, filter_unsupported_sentences, verify_fact
from .llm import InvalidResponseError, LLMClient, LLMRequest
from .prompts import ANALYSIS_SYSTEM, analysis_prompt
from .scoring import ScoreError, validate_criteria, weighted_score
from .util import extract_numbers, parse_iso

log = logging.getLogger(__name__)

MAX_FACTS = 6
MAX_CLAIMS = 2
MAX_PRELIMS = 3
MAX_WRITTEN_POINTS = 3
FACT_KINDS = {"number", "name", "date", "place", "institution", "rank", "law_policy", "other"}


class AnalysisRejected(Exception):
    """The model judged the story irrelevant or the evidence insufficient."""


@dataclass
class AnalysisInput:
    mode: str  # "full" | "excerpt_only"
    text: str
    article_id: str


def allowed_numbers_for(source_text: str, headline: str, published_at: str | None) -> set[str]:
    allowed = extract_numbers(source_text) | extract_numbers(headline)
    dt = parse_iso(published_at)
    if dt:
        allowed |= {str(dt.year), str(dt.day), f"{dt.day:02d}", str(dt.month)}
    return allowed


def validate_analysis(
    raw: object,
    source_text: str,
    headline: str,
    published_at: str | None,
    weights: dict[str, float],
    profile=None,
    min_syllabus_relevance: float = 0,
    followup: dict | None = None,
) -> dict:
    """Validate and ground a model analysis. Returns a note dict.

    Raises InvalidResponseError for malformed output and AnalysisRejected when
    the story should not be included.
    """
    if not isinstance(raw, dict):
        raise InvalidResponseError("Analysis response is not an object")
    if raw.get("relevant") is False:
        raise AnalysisRejected(clean_model_text(raw.get("rejection_reason"), 300) or "Model judged it not BCS-relevant")
    if raw.get("evidence_sufficient") is False:
        raise AnalysisRejected(clean_model_text(raw.get("rejection_reason"), 300) or "Insufficient evidence")
    try:
        criteria = validate_criteria(raw.get("scores"))
    except ScoreError as exc:
        raise InvalidResponseError(f"Invalid scores: {exc}") from exc
    topic = None
    if profile is not None:
        topic = raw.get("syllabus_topic") if raw.get("syllabus_topic") in profile.topics else None
        if topic is None:
            raise AnalysisRejected("No BCS syllabus topic fits this item")
    if criteria["syllabus_relevance"] < min_syllabus_relevance:
        raise AnalysisRejected(f"Syllabus relevance {criteria['syllabus_relevance']:g} below {min_syllabus_relevance:g}")

    allowed = allowed_numbers_for(source_text, headline, published_at)
    dropped = 0

    facts = []
    raw_facts = raw.get("key_facts")
    if not isinstance(raw_facts, list):
        raise InvalidResponseError("key_facts must be a list")
    for item in raw_facts[: MAX_FACTS * 2]:
        if not isinstance(item, dict):
            dropped += 1
            continue
        fact = clean_model_text(item.get("fact"), 300)
        evidence = clean_model_text(item.get("evidence"), 500)
        if verify_fact(fact, evidence, source_text):
            kind = item.get("kind") if item.get("kind") in FACT_KINDS else "other"
            # Only the first few flagged facts make the "memorise for prelims" list.
            prelims = item.get("prelims") is True and sum(f["prelims"] for f in facts) < MAX_PRELIMS
            facts.append({"fact": fact, "evidence": evidence, "kind": kind, "prelims": prelims})
        else:
            dropped += 1
        if len(facts) >= MAX_FACTS:
            break

    claims = []
    for item in (raw.get("attributed_claims") or [])[: MAX_CLAIMS * 2]:
        if not isinstance(item, dict):
            continue
        claim = clean_model_text(item.get("claim"), 300)
        who = clean_model_text(item.get("attributed_to"), 100)
        if who and claim.startswith(who):  # "X: X said ..." reads badly; the speaker is shown separately
            claim = claim[len(who):].lstrip(" :,-—").strip()
        evidence = clean_model_text(item.get("evidence"), 500)
        if who and verify_fact(claim, evidence, source_text):
            claims.append({"claim": claim, "attributed_to": who, "evidence": evidence})
        else:
            dropped += 1
        if len(claims) >= MAX_CLAIMS:
            break

    sections = {}
    for key, limit in (("what_happened", 700), ("why_it_matters", 500), ("whats_new", 300), ("background_context", 400)):
        text, n = filter_unsupported_sentences(clean_model_text(raw.get(key), limit), allowed)
        sections[key] = text
        dropped += n
    if not followup:
        sections["whats_new"] = ""
    points = []
    raw_points = raw.get("written_points") if isinstance(raw.get("written_points"), list) else []
    for point in raw_points[: MAX_WRITTEN_POINTS * 2]:
        text, n = filter_unsupported_sentences(clean_model_text(point, 260), allowed)
        dropped += n
        if text and len(points) < MAX_WRITTEN_POINTS:
            points.append(text)

    if not facts:
        raise AnalysisRejected("No key fact could be verified against the source text")
    if not sections["what_happened"]:
        raise AnalysisRejected("Summary could not be grounded in the source text")

    categories = [c for c in raw.get("categories") or [] if isinstance(c, str) and c in SUBJECTS]
    if topic is not None:
        categories.insert(0, profile.topics[topic].subject)  # the topic's subject is authoritative
    categories = list(dict.fromkeys(categories))[:2]
    # Derived, not asked: memorisable facts make it prelims material; a real
    # written-exam angle (by the calibrated score) makes it written material.
    has_prelims = any(f["prelims"] for f in facts)
    written = criteria["written_potential"] >= 7 and bool(points)
    focus = "both" if has_prelims and written else "written" if written else "prelims"
    institutions = [clean_model_text(x, 120) for x in (raw.get("institutions") or []) if isinstance(x, str)][:8]

    return {
        "headline_bn": clean_model_text(raw.get("headline_bn"), 200) or headline,
        "categories": categories,
        "syllabus_topic": topic,
        "likely_question": clean_model_text(raw.get("likely_question"), 300),
        "exam_focus": focus,
        "key_facts": facts,
        "attributed_claims": claims,
        "written_points": points,
        "institutions": [i for i in institutions if i],
        "criteria": criteria,
        "score": weighted_score(criteria, weights),
        "unsupported_dropped": dropped,
        **sections,
    }


def analyse(
    client: LLMClient,
    inp: AnalysisInput,
    headline: str,
    source_name: str,
    published_at: str | None,
    weights: dict[str, float],
    profile=None,
    min_syllabus_relevance: float = 0,
    followup: dict | None = None,
) -> dict:
    """``followup``: an earlier, already-sent story this one follows up
    (date, headline, new_development from screening)."""
    request = LLMRequest(
        task="analysis",
        system=ANALYSIS_SYSTEM,
        prompt=analysis_prompt(headline, source_name, published_at, inp.mode, inp.text, profile, followup),
        payload={"headline": headline, "source": source_name, "mode": inp.mode, "text": inp.text, "published": published_at},
    )
    raw = client.generate_json(request)
    note = validate_analysis(raw, inp.text, headline, published_at, weights, profile, min_syllabus_relevance, followup)
    note["mode"] = inp.mode
    note["analysed_article_id"] = inp.article_id
    if followup:
        note["followup_of"] = {k: followup.get(k) for k in ("event_id", "date", "headline")}
    return note
