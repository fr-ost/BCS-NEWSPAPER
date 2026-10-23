"""News source registry (``config/sources.toml``)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .config import SUBJECTS, ConfigError
from .util import is_valid_http_url

SOURCE_TYPES = {"newspaper", "news_agency", "official", "international_org"}


@dataclass(frozen=True)
class Source:
    id: str
    name: str
    homepage: str
    feed_url: str
    type: str
    language: str
    subjects: tuple[str, ...]
    enabled: bool
    fetch_articles: bool
    timeout_seconds: float
    max_retries: int
    verification: str = ""
    aliases: tuple[str, ...] = ()

    @property
    def collectable(self) -> bool:
        return self.enabled and bool(self.feed_url)


def parse_sources(data: dict) -> list[Source]:
    entries = data.get("source", [])
    if not isinstance(entries, list):
        raise ConfigError("sources.toml must contain [[source]] tables")
    sources: list[Source] = []
    seen: set[str] = set()
    for raw in entries:
        try:
            src = Source(
                id=str(raw["id"]),
                name=str(raw["name"]),
                homepage=str(raw["homepage"]),
                feed_url=str(raw.get("feed_url", "") or ""),
                type=str(raw.get("type", "newspaper")),
                language=str(raw.get("language", "en")),
                subjects=tuple(raw.get("subjects", [])),
                enabled=bool(raw.get("enabled", False)),
                fetch_articles=bool(raw.get("fetch_articles", True)),
                timeout_seconds=float(raw.get("timeout_seconds", 20)),
                max_retries=max(0, min(int(raw.get("max_retries", 2)), 5)),
                verification=str(raw.get("verification", "")),
                aliases=tuple(str(a) for a in raw.get("aliases", [])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid source entry {raw!r}: {exc}") from exc
        if src.id in seen:
            raise ConfigError(f"Duplicate source id: {src.id}")
        seen.add(src.id)
        if src.type not in SOURCE_TYPES:
            raise ConfigError(f"Source {src.id}: unknown type {src.type!r}")
        bad_subjects = set(src.subjects) - set(SUBJECTS)
        if bad_subjects:
            raise ConfigError(f"Source {src.id}: unknown subjects {sorted(bad_subjects)}")
        if not is_valid_http_url(src.homepage):
            raise ConfigError(f"Source {src.id}: invalid homepage URL")
        if src.feed_url and not is_valid_http_url(src.feed_url):
            raise ConfigError(f"Source {src.id}: invalid feed URL")
        if src.enabled and not src.feed_url:
            raise ConfigError(f"Source {src.id} is enabled but has no verified feed_url")
        if not (1 <= src.timeout_seconds <= 120):
            raise ConfigError(f"Source {src.id}: timeout_seconds must be 1-120")
        sources.append(src)
    return sources


def load_sources(config_dir: Path) -> list[Source]:
    path = config_dir / "sources.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Missing source registry: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc
    return parse_sources(data)
