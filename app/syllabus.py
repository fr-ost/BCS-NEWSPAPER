"""BCS syllabus profile (``config/syllabus.toml``) used to select news."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .config import SUBJECTS, ConfigError


@dataclass(frozen=True)
class Topic:
    id: str
    label: str
    subject: str
    hint: str


@dataclass(frozen=True)
class SyllabusProfile:
    topics: dict[str, Topic]
    accepted_types: tuple[str, ...]
    rejected_types: tuple[str, ...]
    bad_examples: tuple[str, ...]
    good_examples: tuple[str, ...]

    @property
    def story_types(self) -> tuple[str, ...]:
        return self.accepted_types + self.rejected_types

    def catalog_text(self) -> str:
        """Compact topic list for prompts: ``id: label (hint)``."""
        return "\n".join(f"- {t.id}: {t.label} — {t.hint}" for t in self.topics.values())

    def examples_text(self) -> str:
        bad = "\n".join(f"  ✗ {e}" for e in self.bad_examples)
        good = "\n".join(f"  ✓ {e}" for e in self.good_examples)
        return f"Examples to REJECT:\n{bad}\nExamples to SELECT:\n{good}"


def parse_syllabus(data: dict) -> SyllabusProfile:
    raw_topics = data.get("topic") or {}
    if not isinstance(raw_topics, dict) or not raw_topics:
        raise ConfigError("syllabus.toml must define [topic.<id>] tables")
    topics: dict[str, Topic] = {}
    for tid, raw in raw_topics.items():
        try:
            topic = Topic(id=tid, label=str(raw["label"]), subject=str(raw["subject"]), hint=str(raw.get("hint", "")))
        except (KeyError, TypeError) as exc:
            raise ConfigError(f"Invalid syllabus topic {tid!r}: {exc}") from exc
        if topic.subject not in SUBJECTS:
            raise ConfigError(f"Syllabus topic {tid!r} has unknown subject {topic.subject!r}")
        topics[tid] = topic
    policy = data.get("policy") or {}
    accepted = tuple(policy.get("accepted_types") or ())
    rejected = tuple(policy.get("rejected_types") or ())
    if not accepted or not rejected or set(accepted) & set(rejected):
        raise ConfigError("syllabus.toml [policy] needs disjoint, non-empty accepted_types and rejected_types")
    return SyllabusProfile(
        topics=topics,
        accepted_types=accepted,
        rejected_types=rejected,
        bad_examples=tuple(policy.get("bad_examples") or ()),
        good_examples=tuple(policy.get("good_examples") or ()),
    )


def load_syllabus(config_dir: Path) -> SyllabusProfile:
    path = config_dir / "syllabus.toml"
    try:
        return parse_syllabus(tomllib.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        raise ConfigError(f"Missing syllabus profile: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc
