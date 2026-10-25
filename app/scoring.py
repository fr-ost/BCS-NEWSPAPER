"""Stage E: configurable relevance scoring rubric.

The weights are starting heuristics, not validated predictions of exam
questions. The score supports selection; it never replaces evidence checks.
"""

from __future__ import annotations

from .config import SCORE_CRITERIA


class ScoreError(ValueError):
    pass


def validate_criteria(raw: object) -> dict[str, float]:
    """Validate model-provided 0-10 sub-scores. Raises ScoreError if malformed."""
    if not isinstance(raw, dict):
        raise ScoreError("scores must be an object")
    out: dict[str, float] = {}
    for key in SCORE_CRITERIA:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ScoreError(f"score {key} missing or not a number")
        if not 0 <= value <= 10:
            raise ScoreError(f"score {key}={value} outside 0-10")
        out[key] = float(value)
    return out


def weighted_score(criteria: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted total on a 0-100 scale, rounded to one decimal."""
    total = sum(weights[k] * criteria[k] for k in SCORE_CRITERIA)
    return round(total * 10, 1)


def rank_stories(stories: list[dict], min_score: float, max_stories: int, max_per_topic: int | None = None) -> list[dict]:
    """Pick the best stories. Never pads: only those meeting ``min_score``.

    ``max_per_topic`` keeps the edition diverse: once a syllabus topic has
    that many stories, lower-scoring stories on the same topic are skipped.
    """
    eligible = [s for s in stories if s.get("score", 0) >= min_score]
    eligible.sort(key=lambda s: (-s["score"], s.get("published_at") or "", s["event_id"]))
    picked: list[dict] = []
    per_topic: dict[str, int] = {}
    for story in eligible:
        topic = (story.get("analysis") or {}).get("syllabus_topic")
        if max_per_topic and topic:
            if per_topic.get(topic, 0) >= max_per_topic:
                continue
            per_topic[topic] = per_topic.get(topic, 0) + 1
        picked.append(story)
        if len(picked) >= max_stories:
            break
    return picked
