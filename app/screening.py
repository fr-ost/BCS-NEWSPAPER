"""Stage C: lightweight relevance screening of candidate events.

Candidates are given short ids (c1, c2, ...) that map to our own event ids.
Anything the model returns that does not match a candidate is ignored, so a
hallucinated or injected id can never select an unrelated story.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import SUBJECTS
from .grounding import clean_model_text
from .llm import InvalidResponseError, LLMClient, LLMRequest
from .prompts import SCREENING_SYSTEM, screening_prompt

log = logging.getLogger(__name__)

DECISIONS = {"select", "maybe", "reject"}
EXAM_FOCUS = {"prelims", "written", "both", "none"}
MAX_EXCERPT_FOR_SCREENING = 500


@dataclass
class ScreeningOutcome:
    decisions: dict[str, dict] = field(default_factory=dict)  # event_id -> decision record
    merges: list[tuple[str, str]] = field(default_factory=list)
    invalid_items: int = 0
    unscreened: list[str] = field(default_factory=list)  # event ids without a valid result


def build_candidates(events: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """``events`` items need event_id, title, excerpt, source_name, published_at."""
    candidates, id_map = [], {}
    for i, ev in enumerate(events, start=1):
        cid = f"c{i}"
        id_map[cid] = ev["event_id"]
        candidates.append(
            {
                "id": cid,
                "title": ev["title"],
                "excerpt": (ev.get("excerpt") or "")[:MAX_EXCERPT_FOR_SCREENING],
                "source": ev.get("source_name", ""),
                "published": ev.get("published_at") or "unknown",
            }
        )
    return candidates, id_map


def build_recent(recent: list[dict]) -> tuple[list[dict], dict[str, dict]]:
    """Short ids (r1, r2, ...) for stories already sent; see ``validate_screening``."""
    shown, rmap = [], {}
    for i, r in enumerate(recent, start=1):
        rid = f"r{i}"
        rmap[rid] = r
        shown.append({"id": rid, "sent": r.get("date", ""), "headline": r.get("headline", ""), "title": r.get("title", "")})
    return shown, rmap


def validate_screening(
    raw: object,
    id_map: dict[str, str],
    profile=None,
    maybe_min_priority: float = 7,
    recent_map: dict[str, dict] | None = None,
) -> ScreeningOutcome:
    """Validate model output. With a syllabus ``profile``, code enforces the
    selection policy: rejected story types, missing/unknown syllabus topics and
    low-priority "maybe" items are rejected whatever the model decided.
    With ``recent_map`` (already-sent stories by short id), a candidate the
    model marks as a repeat of one of them is rejected unless it names a new
    development; a genuine follow-up records what it follows up."""
    if isinstance(raw, list):
        raw = {"results": raw}
    if not isinstance(raw, dict) or not isinstance(raw.get("results"), list):
        raise InvalidResponseError("Screening response lacks a 'results' list")
    out = ScreeningOutcome()
    seen: set[str] = set()
    for item in raw["results"]:
        if not isinstance(item, dict):
            out.invalid_items += 1
            continue
        cid = item.get("id")
        if not isinstance(cid, str) or cid not in id_map or cid in seen:
            out.invalid_items += 1
            continue
        decision = item.get("decision")
        if decision not in DECISIONS:
            out.invalid_items += 1
            continue
        seen.add(cid)
        categories = [c for c in item.get("categories") or [] if isinstance(c, str) and c in SUBJECTS]
        priority = item.get("priority")
        if isinstance(priority, bool) or not isinstance(priority, (int, float)):
            priority = 5
        focus = item.get("exam_focus") if item.get("exam_focus") in EXAM_FOCUS else "none"
        priority = max(0.0, min(10.0, float(priority)))
        record = {
            "decision": decision,
            "categories": list(dict.fromkeys(categories))[:2],
            "rationale": clean_model_text(item.get("rationale"), 300),
            "exam_focus": focus,
            "priority": priority,
        }
        if profile is not None:
            story_type = item.get("story_type") if item.get("story_type") in profile.story_types else "other_low_value"
            topic = item.get("topic") if item.get("topic") in profile.topics else None
            record.update(story_type=story_type, topic=topic, exam_hook=clean_model_text(item.get("exam_hook"), 200))
            forced = None
            repeat = (recent_map or {}).get(item.get("repeat_of")) if isinstance(item.get("repeat_of"), str) else None
            new_dev = clean_model_text(item.get("new_development"), 200)
            if repeat is not None and new_dev:
                record.update(followup_of={k: repeat.get(k) for k in ("event_id", "date", "headline")}, new_development=new_dev)
            if repeat is not None and not new_dev:
                forced = f"repeat of a story sent on {repeat.get('date', '?')}"
            elif story_type in profile.rejected_types:
                forced = f"story type {story_type}"
            elif topic is None:
                forced = "no syllabus topic"
            elif decision == "maybe" and priority < maybe_min_priority:
                forced = f"weak maybe (priority {priority:g})"
            if forced and decision != "reject":
                record["decision"] = "reject"
                record["forced_reject"] = forced
        out.decisions[id_map[cid]] = record
        for other in item.get("same_event_as") or []:
            if isinstance(other, str) and other in id_map and other != cid:
                out.merges.append((id_map[cid], id_map[other]))
    out.unscreened = [eid for cid, eid in id_map.items() if cid not in seen]
    return out


def screen_events(
    client: LLMClient,
    events: list[dict],
    batch_size: int,
    profile=None,
    maybe_min_priority: float = 7,
    recent: list[dict] | None = None,
) -> ScreeningOutcome:
    """Screen events in batches. Quota errors propagate after earlier batches
    are merged into the returned outcome via the ``partial`` attribute.

    ``recent``: stories already sent (dicts with event_id, date, headline,
    title) so the model can flag repeats; see ``validate_screening``."""
    total = ScreeningOutcome()
    shown_recent, recent_map = build_recent(recent or [])
    for start in range(0, len(events), batch_size):
        batch = events[start : start + batch_size]
        candidates, id_map = build_candidates(batch)
        request = LLMRequest(
            task="screening",
            system=SCREENING_SYSTEM,
            prompt=screening_prompt(candidates, profile, shown_recent),
            payload={"candidates": candidates, "recent": shown_recent},
            max_output_tokens=8192,
        )
        try:
            raw = client.generate_json(request)
            outcome = validate_screening(raw, id_map, profile, maybe_min_priority, recent_map)
        except Exception as exc:
            exc.partial = total  # type: ignore[attr-defined]
            raise
        total.decisions.update(outcome.decisions)
        total.merges.extend(outcome.merges)
        total.invalid_items += outcome.invalid_items
        total.unscreened.extend(outcome.unscreened)
        if outcome.invalid_items:
            log.warning("Screening ignored %d invalid/unknown items", outcome.invalid_items)
    return total
