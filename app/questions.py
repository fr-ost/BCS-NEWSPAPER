"""Stage G: generate and strictly validate MCQs from verified facts."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from .config import SUBJECTS
from .grounding import clean_model_text
from .llm import InvalidResponseError, LLMClient, LLMRequest
from .prompts import MCQ_SYSTEM, mcq_prompt
from .util import extract_numbers, normalize_digits, normalize_text

log = logging.getLogger(__name__)

MAX_FACTS_FOR_MCQ = 30
ARTICLE_FACTS_PER_STORY = 3
GK_FACTS_PER_STORY = 2
# Ask for more candidates than needed; code keeps the best valid ones.
CANDIDATE_FACTOR = 3
# Questions from verified static GK (not the article) per edition.
MAX_GK_QUESTIONS = 2
# Bare-figure questions from the news (an amount, a count) per edition: BPSC
# asks some, but a quiz of three "how much" questions tests nothing durable.
MAX_NEWS_FIGURE_QUESTIONS = 1

# Units/kinds that every option must share with the correct answer.
_UNITS = {
    "percent": re.compile(r"শতাংশ|%|percent", re.I),
    "crore": re.compile(r"কোটি|crore", re.I),
    "lakh": re.compile(r"লাখ|লক্ষ|lakh", re.I),
    "thousand": re.compile(r"হাজার|thousand", re.I),
    "billion": re.compile(r"বিলিয়ন|billion", re.I),
    "million": re.compile(r"মিলিয়ন|million", re.I),
    "dollar": re.compile(r"ডলার|\$|dollar|usd", re.I),
    "taka": re.compile(r"টাকা|৳|\btk\b|taka", re.I),
    "ordinal": re.compile(r"\d\s*(?:তম|ম|য়|র্থ|th|st|nd|rd)\b|তম\b", re.I),
    "day": re.compile(r"দিন|\bdays?\b", re.I),
    "km": re.compile(r"কিলোমিটার|\bkm\b", re.I),
    "mw": re.compile(r"মেগাওয়াট|\bmw\b", re.I),
}
_PURE_NUMBER = re.compile(r"[\d\s.,%৳$+\-–]+")
# The question must stand alone; it may not lean on "the article".
_BANNED_QUESTION = re.compile(
    r"প্রতিবেদন\s*অনুযায়ী|প্রতিবেদনটি|এই\s*প্রতিবেদন|উক্ত\s*প্রতিবেদন|প্রতিবেদনে\s*উল্লিখিত|খবর\s*অনুযায়ী|এই\s*খবর|খবরে\s*বলা"
    r"|খবরে\s*উল্লিখিত|সংবাদ\s*অনুযায়ী|নিবন্ধ|according to the (?:report|article|news)|this (?:report|article)",
    re.I,
)
_BANNED_OPTION = re.compile(r"সবগুলো|সবকটি|কোনোটিই\s*নয়|কোনটিই\s*নয়|ওপরের সব|উপরের সব|all of the above|none of the above", re.I)


@dataclass
class McqOutcome:
    questions: list[dict] = field(default_factory=list)
    rejected: int = 0
    reasons: list[str] = field(default_factory=list)


def build_fact_pool(stories: list[dict]) -> list[dict]:
    """Facts the questions may use: every story's must-remember (prelims)
    article facts first, then its verified static GK, then its other facts;
    interleaved across stories so no single story dominates."""
    per_story: list[list[dict]] = []
    for story in stories:
        note = story["analysis"]
        prelims = [f for f in note["key_facts"] if f.get("prelims")]
        others = [f for f in note["key_facts"] if not f.get("prelims")]
        article = (prelims + others)[:ARTICLE_FACTS_PER_STORY]
        gk = (note.get("static_gk") or [])[:GK_FACTS_PER_STORY]
        # Alternate the story's best article facts with its verified GK, so the
        # pool (capped at MAX_FACTS_FOR_MCQ) always offers both kinds.
        firsts = [(f["fact"], f["evidence"], "article") for f in article if f.get("prelims")]
        firsts += [(f["fact"], f["evidence"], "article") for f in article if not f.get("prelims")]
        background = [(g["statement"], g["statement"], "gk") for g in gk]
        ordered = []
        for i in range(max(len(firsts), len(background))):
            ordered += firsts[i : i + 1] + background[i : i + 1]
        per_story.append([{"story_ref": story, "fact": f, "evidence": e, "basis": b} for f, e, b in ordered])
    pool: list[dict] = []
    depth = 0
    while len(pool) < MAX_FACTS_FOR_MCQ and any(depth < len(items) for items in per_story):
        for items in per_story:
            if depth < len(items) and len(pool) < MAX_FACTS_FOR_MCQ:
                entry = items[depth]
                story, note = entry["story_ref"], entry["story_ref"]["analysis"]
                pool.append(
                    {
                        "fact_id": f"f{len(pool) + 1}",
                        "story": note["headline_bn"],
                        "fact": entry["fact"],
                        "evidence": entry["evidence"],
                        "basis": entry["basis"],
                        "subject": (note["categories"] or ["bangladesh_affairs"])[0],
                        "event_id": story["event_id"],
                        "url": story["primary_url"],
                        "source_name": story["primary_source"],
                    }
                )
        depth += 1
    return pool


def _units(text: str) -> set[str]:
    return {name for name, pat in _UNITS.items() if pat.search(text)}


def _has_digit(text: str) -> bool:
    return bool(re.search(r"\d", normalize_digits(text)))


def option_problem(options: list[str], idx: int) -> str | None:
    """Options must be the same kind of thing as the correct answer."""
    correct = options[idx]
    others = [o for j, o in enumerate(options) if j != idx]
    if _has_digit(correct):
        if not all(_has_digit(o) for o in others):
            return "numeric answer but non-numeric options"
        units = _units(correct)
        if any(_units(o) != units for o in others):
            return "options mix units"
    elif any(_PURE_NUMBER.fullmatch(normalize_digits(o).strip()) for o in others):
        return "non-numeric answer but a bare number among the options"
    if any(_BANNED_OPTION.search(o) for o in options):
        return "'all/none of the above' option"
    longest_other = max(len(o) for o in others)
    if len(correct) > 25 and len(correct) > 2 * longest_other:
        return "correct option much longer than the others"
    return None


def _check_candidate(item: object, by_id: dict[str, dict]) -> tuple[dict | None, str | None]:
    """One candidate on its own: structure, wording, options, grounding."""
    if not isinstance(item, dict):
        return None, "not an object"
    question = clean_model_text(item.get("question"), 300)
    options = item.get("options")
    idx = item.get("answer_index")
    fact = by_id.get(item.get("fact_id")) if isinstance(item.get("fact_id"), str) else None
    if not question:
        return None, "empty question"
    if not isinstance(options, list) or len(options) != 4 or not all(isinstance(o, str) for o in options):
        return None, "options must be exactly four strings"
    options = [clean_model_text(o, 150) for o in options]
    if any(not o for o in options) or len({normalize_text(o) for o in options}) != 4:
        return None, "options empty or duplicated"
    if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx <= 3:
        return None, "answer_index out of range"
    if fact is None:
        return None, "unknown fact_id"
    if _BANNED_QUESTION.search(question):
        return None, "question leans on 'the report/news' instead of standing alone"
    key = normalize_text(question)
    answer_key = normalize_text(options[idx])
    if len(answer_key) >= 3 and not _PURE_NUMBER.fullmatch(answer_key) and answer_key in key:
        return None, "question gives the answer away"
    problem = option_problem(options, idx)
    if problem:
        return None, problem
    supported_numbers = extract_numbers(fact["fact"]) | extract_numbers(fact["evidence"])
    if not extract_numbers(options[idx]) <= supported_numbers:
        return None, "correct answer contains numbers not in the supporting fact"
    basis = fact.get("basis", "article")
    return {
        "question": question,
        "options": options,
        "answer_index": idx,
        "explanation": clean_model_text(item.get("explanation"), 300),
        "subject": item.get("subject") if item.get("subject") in SUBJECTS else fact["subject"],
        "fact_id": fact["fact_id"],
        "basis": basis,
        "source_event_id": fact["event_id"],
        "source_url": fact["url"],
        "source_name": fact["source_name"],
        "source_story": fact["story"],
        "_key": key,
        "_figure": basis == "article" and _has_digit(options[idx]) and "ordinal" not in _units(options[idx]),
    }, None


def validate_mcqs(raw: object, facts: list[dict], count: int) -> McqOutcome:
    """Check every candidate, then pick up to ``count`` in the model's order:
    one fact and one question each, at most one per story (unless there are
    fewer stories than questions), at most MAX_GK_QUESTIONS from static GK and,
    while other good candidates exist, at most MAX_NEWS_FIGURE_QUESTIONS that
    ask for a bare figure from the news."""
    if isinstance(raw, list):
        raw = {"questions": raw}
    if not isinstance(raw, dict) or not isinstance(raw.get("questions"), list):
        raise InvalidResponseError("MCQ response lacks a 'questions' list")
    by_id = {f["fact_id"]: f for f in facts}
    out = McqOutcome()
    valid = []
    for item in raw["questions"]:
        candidate, reason = _check_candidate(item, by_id)
        if candidate is None:
            out.rejected += 1
            out.reasons.append(reason)
        else:
            valid.append(candidate)
    story_cap = 1 if len({f["event_id"] for f in facts}) >= count else 2
    chosen: list[dict] = []
    skipped: dict[int, str] = {}

    def pick(limit_figures: bool) -> None:
        for i, c in enumerate(valid):
            if len(chosen) >= count:
                return
            if any(c is x for x in chosen):
                continue
            if any(x["fact_id"] == c["fact_id"] for x in chosen):
                skipped[i] = "fact already used"
            elif any(x["_key"] == c["_key"] for x in chosen):
                skipped[i] = "duplicate question"
            elif sum(x["source_event_id"] == c["source_event_id"] for x in chosen) >= story_cap:
                skipped[i] = "story already has a question"
            elif c["basis"] == "gk" and sum(x["basis"] == "gk" for x in chosen) >= MAX_GK_QUESTIONS:
                skipped[i] = "enough background-knowledge questions"
            elif limit_figures and c["_figure"] and sum(x["_figure"] for x in chosen) >= MAX_NEWS_FIGURE_QUESTIONS:
                skipped[i] = "enough 'how much' questions from the news"
            else:
                chosen.append(c)
                skipped.pop(i, None)

    # BPSC prelims is mostly durable GK: one verified-GK question goes in first
    # whenever the model wrote a valid one, wherever it ranked it.
    first_gk = next((c for c in valid if c["basis"] == "gk"), None)
    if first_gk is not None and count > 0:
        chosen.append(first_gk)
    pick(limit_figures=True)
    pick(limit_figures=False)  # a thin day: rather a second figure question than none
    out.rejected += len(skipped)
    out.reasons.extend(skipped.values())
    order = {id(c): i for i, c in enumerate(valid)}
    chosen.sort(key=lambda c: order[id(c)])
    out.questions = [{k: v for k, v in c.items() if not k.startswith("_")} for c in chosen]
    return out


def generate_mcqs(client: LLMClient, stories: list[dict], count: int) -> McqOutcome:
    facts = build_fact_pool(stories)
    if not facts or count <= 0:
        return McqOutcome(reasons=["No verified facts available for questions"])
    model_facts = [{k: f[k] for k in ("fact_id", "story", "fact", "evidence", "basis", "subject")} for f in facts]
    wanted = count * CANDIDATE_FACTOR
    request = LLMRequest(
        task="mcq",
        system=MCQ_SYSTEM,
        prompt=mcq_prompt(model_facts, wanted, count),
        payload={"facts": model_facts, "count": wanted},
    )
    raw = client.generate_json(request)
    outcome = validate_mcqs(raw, facts, count)
    if outcome.rejected:
        log.warning("Rejected %d MCQ candidates: %s", outcome.rejected, "; ".join(outcome.reasons[:6]))
    return outcome
