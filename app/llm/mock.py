"""Offline providers.

``MockProvider`` produces deterministic, clearly labelled placeholder output
from the request payload so the whole pipeline can run without network or
quota (``--offline`` dry runs). It is not an AI and its text is marked
"[মক]" so it can never be mistaken for real analysis.

``ScriptedProvider`` replays queued responses/exceptions for unit tests.
"""

from __future__ import annotations

import json
import re

from .base import LLMProvider, LLMRequest

_SENTENCE = re.compile(r"(?<=[.!?।])\s+")


_TOPIC_KEYWORDS = (
    ("remittance", "bd_macro_data"), ("রেমিট্যান্স", "bd_macro_data"),
    ("labour", "bd_parliament_law"), ("শ্রম", "bd_parliament_law"),
    ("cyclone", "env_disaster"), ("ঘূর্ণিঝড়", "env_disaster"),
    ("nepal", "fp_bilateral"), ("নেপাল", "fp_bilateral"),
)
_FALLBACK_TOPICS = ("bd_infrastructure", "int_conflicts", "sci_health", "rep_indices", "env_climate")


def _mock_topic(text: str) -> str:
    """Deterministic syllabus topic for offline runs (keyword, else hash)."""
    low = (text or "").lower()
    for kw, topic in _TOPIC_KEYWORDS:
        if kw in low:
            return topic
    return _FALLBACK_TOPICS[sum(map(ord, low)) % len(_FALLBACK_TOPICS)]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text or "") if len(s.strip()) > 40]


class MockProvider(LLMProvider):
    name = "mock"

    def generate_raw(self, request: LLMRequest) -> str:
        handler = getattr(self, f"_{request.task}", None)
        if handler is None:
            return json.dumps({"ok": True})
        return json.dumps(handler(request.payload), ensure_ascii=False)

    def _screening(self, payload: dict) -> dict:
        results = []
        for c in payload.get("candidates", []):
            substantial = len(c.get("excerpt", "")) >= 100
            results.append(
                {
                    "id": c["id"],
                    "story_type": "policy_decision" if substantial else "other_low_value",
                    "topic": _mock_topic(c.get("title", "") + " " + c.get("excerpt", "")) if substantial else "none",
                    "exam_hook": "[মক] নমুনা পরীক্ষা-সংযোগ",
                    "decision": "select" if substantial else "reject",
                    "categories": ["bangladesh_affairs"],
                    "rationale": "[মক] দৈর্ঘ্যভিত্তিক নমুনা বাছাই",
                    "exam_focus": "both" if substantial else "none",
                    "priority": 7 if substantial else 2,
                    "same_event_as": [],
                }
            )
        return {"results": results}

    def _analysis(self, payload: dict) -> dict:
        sents = _sentences(payload.get("text", ""))
        with_numbers = [s for s in sents if re.search(r"\d", s)] or sents
        facts = [{"fact": f"[মক] {s}", "evidence": s, "prelims": True} for s in with_numbers[:3]]
        return {
            "relevant": bool(facts),
            "rejection_reason": "" if facts else "[মক] পর্যাপ্ত তথ্য নেই",
            "headline_bn": f"[মক] {payload.get('headline', '')}",
            "syllabus_topic": _mock_topic(payload.get("headline", "") + " " + payload.get("text", "")[:300]),
            "categories": ["bangladesh_affairs"],
            "exam_focus": "both",
            "likely_question": "[মক] এই খবর থেকে একটি নমুনা প্রশ্ন?",
            "what_happened": f"[মক সারাংশ] {sents[0] if sents else ''}",
            "why_it_matters": "[মক] এটি অফলাইন পরীক্ষার নমুনা লেখা, প্রকৃত বিশ্লেষণ নয়।",
            "key_facts": facts,
            "attributed_claims": [],
            "written_angle": "[মক] লিখিত পরীক্ষার নমুনা দৃষ্টিকোণ।",
            "causes_effects": "",
            "background_context": "",
            "institutions": [],
            "scores": {"syllabus_relevance": 7, "significance": 7, "factual_density": 6, "written_potential": 6, "novelty": 6},
            "evidence_sufficient": bool(facts),
        }

    def _mcq(self, payload: dict) -> dict:
        questions = []
        for fact in payload.get("facts", []):
            nums = re.findall(r"\d+(?:\.\d+)?", fact.get("evidence", ""))
            if not nums:
                continue
            correct = nums[0]
            base = float(correct)
            fmt = (lambda x: str(int(x))) if "." not in correct else (lambda x: f"{x:.1f}")
            distractors = [fmt(base + 1), fmt(base * 2 + 3), fmt(base + 10)]
            options = [correct] + [d for d in distractors if d != correct][:3]
            if len(set(options)) != 4:
                continue
            idx = len(questions) % 4
            options[0], options[idx] = options[idx], options[0]
            questions.append(
                {
                    "question": f"[মক] {fact['fact_id']}: প্রতিবেদনে উল্লিখিত সংখ্যাটি কোনটি? ({fact.get('story', '')[:60]})",
                    "options": options,
                    "answer_index": idx,
                    "explanation": f"[মক] সূত্র: {fact.get('evidence', '')[:150]}",
                    "subject": fact.get("subject") or "bangladesh_affairs",
                    "fact_id": fact["fact_id"],
                }
            )
            if len(questions) >= payload.get("count", 3):
                break
        return {"questions": questions}


class ScriptedProvider(LLMProvider):
    """Returns queued responses in order. Items may be str, dict/list or Exception."""

    name = "scripted"

    def __init__(self, responses: list | None = None):
        self.responses = list(responses or [])
        self.requests: list[LLMRequest] = []

    def queue(self, *items) -> None:
        self.responses.extend(items)

    def generate_raw(self, request: LLMRequest) -> str:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"ScriptedProvider has no response queued for task {request.task}")
        item = self.responses.pop(0)
        if callable(item) and not isinstance(item, (str, dict, list)):
            item = item(request)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, (dict, list)):
            return json.dumps(item, ensure_ascii=False)
        return item
