"""Configuration loading.

Non-secret settings live in ``config/settings.toml``. Secrets are read only
from environment variables and are never logged or written to disk.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"

SUBJECTS: dict[str, str] = {
    "bangladesh_affairs": "বাংলাদেশ বিষয়াবলি",
    "international_affairs": "আন্তর্জাতিক বিষয়াবলি",
    "economy": "অর্থনীতি",
    "environment_climate": "পরিবেশ ও জলবায়ু",
    "science_technology": "বিজ্ঞান ও প্রযুক্তি",
    "foreign_policy_geopolitics": "পররাষ্ট্রনীতি ও ভূ-রাজনীতি",
    "constitution_governance": "সংবিধান ও সুশাসন",
    "awards_reports_indices": "পুরস্কার, প্রতিবেদন ও সূচক",
}

SCORE_CRITERIA = (
    "syllabus_relevance",
    "significance",
    "factual_density",
    "written_potential",
    "novelty",
)


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class EditionConfig:
    timezone: str = "Asia/Dhaka"
    min_target_stories: int = 5
    max_stories: int = 8
    min_story_score: float = 55
    freshness_hours: int = 36
    mcq_count: int = 3


@dataclass(frozen=True)
class CollectionConfig:
    max_items_per_source: int = 40
    max_feed_bytes: int = 3_000_000
    max_article_bytes: int = 3_000_000
    request_timeout_seconds: float = 20
    request_retries: int = 2
    user_agent: str = "BCSNewspaperBot/1.0"
    respect_robots_txt: bool = True


@dataclass(frozen=True)
class ExtractionConfig:
    min_article_chars: int = 600
    max_article_chars: int = 12_000
    min_excerpt_chars_for_analysis: int = 220


@dataclass(frozen=True)
class ScreeningConfig:
    max_candidates: int = 60
    batch_size: int = 60


@dataclass(frozen=True)
class AnalysisConfig:
    max_analyses_per_run: int = 9
    max_attempts_per_story: int = 3


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "gemini"
    gemini_model: str = "gemini-3.8-flash"
    gemini_api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_output_tokens: int = 8192
    request_timeout_seconds: float = 120
    max_calls_per_run: int = 16
    min_seconds_between_calls: float = 6.0
    max_retries: int = 3
    backoff_base_seconds: float = 4.0
    backoff_max_seconds: float = 60.0
    max_retry_after_seconds: float = 90.0


@dataclass(frozen=True)
class TelegramConfig:
    api_base: str = "https://api.telegram.org"
    max_message_chars: int = 3800
    request_timeout_seconds: float = 30
    max_retries: int = 3
    disable_web_page_preview: bool = True
    # Subscribers (you and your friends). Nobody is added without pressing Start.
    max_subscribers: int = 10
    min_seconds_between_messages: float = 1.1


@dataclass(frozen=True)
class StateConfig:
    path: str = "state/state.json"
    retention_days: int = 14
    max_run_history: int = 30


@dataclass(frozen=True)
class Secrets:
    """Credentials from the environment. ``repr`` never reveals values."""

    gemini_api_key: str = field(default="", repr=False)
    telegram_bot_token: str = field(default="", repr=False)
    # Optional. When set, people can only subscribe via t.me/<bot>?start=<code>.
    bot_invite_code: str = field(default="", repr=False)

    def redact(self, text: str) -> str:
        """Remove any secret value that leaked into ``text`` (e.g. an exception)."""
        for value in (self.gemini_api_key, self.telegram_bot_token, self.bot_invite_code):
            if value and len(value) >= 6:
                text = text.replace(value, "***")
        return text


@dataclass(frozen=True)
class Settings:
    edition: EditionConfig
    collection: CollectionConfig
    extraction: ExtractionConfig
    screening: ScreeningConfig
    analysis: AnalysisConfig
    weights: dict[str, float]
    llm: LLMConfig
    telegram: TelegramConfig
    state: StateConfig
    output_dir: Path
    config_dir: Path
    secrets: Secrets = field(default_factory=Secrets, repr=False)

    @property
    def state_path(self) -> Path:
        path = Path(self.state.path)
        return path if path.is_absolute() else PROJECT_ROOT / path


def _section(cls, data: dict, name: str):
    raw = data.get(name, {})
    known = {f for f in cls.__dataclass_fields__}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"Unknown keys in [{name}]: {sorted(unknown)}")
    return cls(**raw)


def validate_weights(weights: dict[str, float]) -> dict[str, float]:
    if set(weights) != set(SCORE_CRITERIA):
        raise ConfigError(
            f"[scoring.weights] must define exactly {list(SCORE_CRITERIA)}; got {sorted(weights)}"
        )
    for key, value in weights.items():
        if not isinstance(value, (int, float)) or value < 0:
            raise ConfigError(f"Weight {key} must be a non-negative number")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(f"Scoring weights must sum to 1.0 (got {total:.3f})")
    return {k: float(v) for k, v in weights.items()}


def load_settings(config_dir: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    config_dir = Path(env.get("BCS_CONFIG_DIR") or config_dir or DEFAULT_CONFIG_DIR)
    path = config_dir / "settings.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Missing settings file: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc

    llm_raw = dict(data.get("llm", {}))
    # Environment overrides for model configuration (kept out of code).
    if env.get("GEMINI_MODEL"):
        llm_raw["gemini_model"] = env["GEMINI_MODEL"]
    if env.get("LLM_PROVIDER"):
        llm_raw["provider"] = env["LLM_PROVIDER"]
    if env.get("OLLAMA_MODEL"):
        llm_raw["ollama_model"] = env["OLLAMA_MODEL"]
    if env.get("OLLAMA_BASE_URL"):
        llm_raw["ollama_base_url"] = env["OLLAMA_BASE_URL"]
    data["llm"] = llm_raw

    state_raw = dict(data.get("state", {}))
    if env.get("BCS_STATE_PATH"):
        state_raw["path"] = env["BCS_STATE_PATH"]
    data["state"] = state_raw

    edition = _section(EditionConfig, data, "edition")
    if not (1 <= edition.max_stories <= 15):
        raise ConfigError("edition.max_stories must be between 1 and 15")
    if edition.mcq_count < 0:
        raise ConfigError("edition.mcq_count must be >= 0")
    invite = env.get("BOT_INVITE_CODE", "").strip()
    if invite and not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", invite):
        raise ConfigError("BOT_INVITE_CODE must be 8-64 characters of A-Z, a-z, 0-9, _ or - (Telegram start-link rules)")

    output_dir = Path(env.get("BCS_OUTPUT_DIR") or data.get("output", {}).get("dir", "output"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    return Settings(
        edition=edition,
        collection=_section(CollectionConfig, data, "collection"),
        extraction=_section(ExtractionConfig, data, "extraction"),
        screening=_section(ScreeningConfig, data, "screening"),
        analysis=_section(AnalysisConfig, data, "analysis"),
        weights=validate_weights(data.get("scoring", {}).get("weights", {})),
        llm=_section(LLMConfig, data, "llm"),
        telegram=_section(TelegramConfig, data, "telegram"),
        state=_section(StateConfig, data, "state"),
        output_dir=output_dir,
        config_dir=config_dir,
        secrets=Secrets(
            gemini_api_key=env.get("GEMINI_API_KEY", "").strip(),
            telegram_bot_token=env.get("TELE_BOT_TOKEN", "").strip(),
            bot_invite_code=env.get("BOT_INVITE_CODE", "").strip(),
        ),
    )
