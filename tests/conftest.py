"""Shared fixtures. Every external API is faked: no test touches the network,
Gemini quota or a real Telegram chat (except tests marked ``integration``)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.__main__ import SAMPLE_DIR, offline_article_fetcher
from app.config import load_settings
from app.llm import build_client
from app.llm.mock import MockProvider
from app.pipeline import Pipeline
from app.sources import load_sources
from app.state import StateStore
from app.telegram_delivery import SendResult, TelegramError

NOW = datetime(2026, 10, 23, 2, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Fail loudly if any test tries a real HTTP request."""
    import requests

    def blocked(*args, **kwargs):
        raise AssertionError("Real network access attempted in a unit test")

    monkeypatch.setattr(requests.Session, "request", blocked)


@pytest.fixture
def settings(tmp_path):
    env = {
        "BCS_STATE_PATH": str(tmp_path / "state" / "state.json"),
        "BCS_OUTPUT_DIR": str(tmp_path / "output"),
        "GEMINI_API_KEY": "test-gemini-key-123456",
        "TELE_BOT_TOKEN": "123456:test-telegram-token",
        "TELE_CHAT_ID": "42",
    }
    return load_settings(env=env)


@pytest.fixture
def sources(settings):
    return load_sources(settings.config_dir)


class FakeTelegram:
    """Records messages; can fail on chosen call numbers (1-based)."""

    def __init__(self, fail_on: set[int] | None = None, permanent: bool = False):
        self.sent: list[str] = []
        self.calls = 0
        self.fail_on = fail_on or set()
        self.permanent = permanent

    def send_message(self, text: str) -> SendResult:
        self.calls += 1
        if self.calls in self.fail_on:
            raise TelegramError("simulated failure", permanent=self.permanent)
        self.sent.append(text)
        return SendResult(message_id=1000 + len(self.sent))


def no_sleep(_seconds):
    return None


@pytest.fixture
def make_pipeline(settings, sources):
    def factory(provider=None, telegram=None, dry_run=False, feed_dir: Path | None = None, store=None):
        provider = provider or MockProvider()
        store = store or StateStore(settings.state_path, settings.state.retention_days, settings.state.max_run_history)
        return Pipeline(
            settings,
            sources,
            store,
            build_client(settings, provider, sleep=no_sleep),
            offline_article_fetcher(settings, SAMPLE_DIR / "articles"),
            telegram=telegram,
            dry_run=dry_run,
            offline_feed_dir=feed_dir or SAMPLE_DIR / "feeds",
        )

    return factory
