"""Stage G: generate and strictly validate MCQs from verified facts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import SUBJECTS
from .grounding import clean_model_text
from .llm import InvalidResponseError, LLMClient, LLMRequest
from .prompts import MCQ_SYSTEM, mcq_prompt
from .util import extract_numbers, normalize_text

log = logging.getLogger(__name__)

MAX_FACTS_FOR_MCQ = 18


@dataclass
class McqOutcome:
    questions: list[dict] = field(default_factory=list)
    rejected: int = 0
    reasons: list[str] = field(default_factory=list)


def build_fact_pool(stories: list[dict]) -> list[dict]:
    """Verified facts from selected stories, prelims facts first."""
    pool: list[dict] = []
    for prelims_first in (True, False):
        for story in stories:
            note = story["analysis"]
            for fact in note["key_facts"]:
                if fact.get("prelims", False) != prelims_first:
                    continue
                pool.append(
                    {
                        "fact_id": f"f{len(pool) + 1}",
                        "story": note["headline_bn"],
                        "fact": fact["fact"],
                        "evidence": fact["evidence"],
                        "subject": (note["categories"] or ["bangladesh_affairs"])[0],
                        "event_id": story["event_id"],
                        "url": story["primary_url"],
                        "source_name": story["primary_source"],
                    }
                )
                if len(pool) >= MAX_FACTS_FOR_MCQ:
                    return pool
    return pool


def validate_mcqs(raw: object, facts: list[dict], count: int) -> McqOutcome:
    if isinstance(raw, list):
        raw = {"questions": raw}
    if not isinstance(raw, dict) or not isinstance(raw.get("questions"), list):
        raise InvalidResponseError("MCQ response lacks a 'questions' list")
    by_id = {f["fact_id"]: f for f in facts}
    out = McqOutcome()
    used_facts: set[str] = set()
    seen_questions: set[str] = set()

    def reject(reason: str) -> None:
        out.rejected += 1
        out.reasons.append(reason)

    for item in raw["questions"]:
        if len(out.questions) >= count:
            break
        if not isinstance(item, dict):
            reject("not an object")
            continue
        question = clean_model_text(item.get("question"), 300)
        options = item.get("options")
        idx = item.get("answer_index")
        fact = by_id.get(item.get("fact_id")) if isinstance(item.get("fact_id"), str) else None
        if not question:
            reject("empty question")
            continue
        if not isinstance(options, list) or len(options) != 4 or not all(isinstance(o, str) for o in options):
            reject("options must be exactly four strings")
            continue
        options = [clean_model_text(o, 150) for o in options]
        if any(not o for o in options) or len({normalize_text(o) for o in options}) != 4:
            reject("options empty or duplicated")
            continue
        if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx <= 3:
            reject("answer_index out of range")
            continue
        if fact is None:
            reject("unknown fact_id")
            continue
        if fact["fact_id"] in used_facts:
            reject("fact already used")
            continue
        key = normalize_text(question)
        if key in seen_questions:
            reject("duplicate question")
            continue
        supported_numbers = extract_numbers(fact["fact"]) | extract_numbers(fact["evidence"])
        if not extract_numbers(options[idx]) <= supported_numbers:
            reject("correct answer contains numbers not in the supporting fact")
            continue
        subject = item.get("subject") if item.get("subject") in SUBJECTS else fact["subject"]
        used_facts.add(fact["fact_id"])
        seen_questions.add(key)
        out.questions.append(
            {
                "question": question,
                "options": options,
                "answer_index": idx,
                "explanation": clean_model_text(item.get("explanation"), 300),
                "subject": subject,
                "fact_id": fact["fact_id"],
                "source_event_id": fact["event_id"],
                "source_url": fact["url"],
                "source_name": fact["source_name"],
                "source_story": fact["story"],
            }
        )
    return out


def generate_mcqs(client: LLMClient, stories: list[dict], count: int) -> McqOutcome:
    facts = build_fact_pool(stories)
    if not facts or count <= 0:
        return McqOutcome(reasons=["No verified facts available for questions"])
    model_facts = [{k: f[k] for k in ("fact_id", "story", "fact", "evidence", "subject")} for f in facts]
    request = LLMRequest(
        task="mcq",
        system=MCQ_SYSTEM,
        prompt=mcq_prompt(model_facts, count),
        payload={"facts": model_facts, "count": count},
    )
    raw = client.generate_json(request)
    outcome = validate_mcqs(raw, facts, count)
    if outcome.rejected:
        log.warning("Rejected %d MCQs: %s", outcome.rejected, "; ".join(outcome.reasons[:5]))
    return outcome
