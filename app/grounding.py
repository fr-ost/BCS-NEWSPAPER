"""Deterministic checks that model output is supported by the source text.

These checks are deliberately strict. A fact or sentence that cannot be
tied to the retrieved text is dropped rather than shown: a missing fact is
better than an invented one.
"""

from __future__ import annotations

import re

from .util import extract_numbers, normalize_text, tokens

_URL = re.compile(r"(https?://\S+|www\.\S+|\b[\w-]+\.(?:com|net|org|bd|io|xyz|info)/\S*)", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+")
_QUOTES = str.maketrans({c: " " for c in "\"'“”‘’«»„`"})
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

EVIDENCE_TOKEN_COVERAGE = 0.85
MIN_EVIDENCE_TOKENS = 3


def clean_model_text(value: object, limit: int = 800) -> str:
    """Coerce to a single safe plain-text string: no URLs, no control chars."""
    if not isinstance(value, str):
        return ""
    text = _CONTROL.sub(" ", value)
    text = _URL.sub("", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + "…"
    return text


def _norm_for_match(text: str) -> str:
    return normalize_text(text.translate(_QUOTES))


def evidence_supported(evidence: str, source_text: str) -> bool:
    """True if the quoted evidence appears in the source (allowing tiny drift)."""
    if not evidence or not source_text:
        return False
    ev = _norm_for_match(evidence)
    src = _norm_for_match(source_text)
    if len(ev) < 12:
        return False
    if ev in src:
        return True
    ev_tokens = tokens(evidence)
    if len(ev_tokens) < MIN_EVIDENCE_TOKENS:
        return False
    src_tokens = set(tokens(source_text))
    covered = sum(1 for t in ev_tokens if t in src_tokens)
    return covered / len(ev_tokens) >= EVIDENCE_TOKEN_COVERAGE


def numbers_supported(text: str, allowed: set[str]) -> bool:
    return extract_numbers(text) <= allowed


def filter_unsupported_sentences(text: str, allowed_numbers: set[str]) -> tuple[str, int]:
    """Drop sentences that contain numbers not present in the allowed set."""
    if not text:
        return "", 0
    kept, dropped = [], 0
    for sentence in _SENTENCE_SPLIT.split(text):
        if not sentence.strip():
            continue
        if numbers_supported(sentence, allowed_numbers):
            kept.append(sentence.strip())
        else:
            dropped += 1
    return " ".join(kept), dropped


def verify_fact(fact: str, evidence: str, source_text: str) -> bool:
    """A fact is verified when its evidence quote is in the source and every
    number in the fact also appears in that evidence quote."""
    if not fact or not evidence_supported(evidence, source_text):
        return False
    return numbers_supported(fact, extract_numbers(evidence))
