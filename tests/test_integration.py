"""Live integration tests. Skipped unless you opt in explicitly:

    RUN_INTEGRATION=1 GEMINI_API_KEY=... TELE_BOT_TOKEN=... TELE_CHAT_ID=... \
        pytest -m integration -p no:randomly

They make real network calls: one Gemini model lookup (no generation),
one Telegram test message, and live fetches of enabled feeds.
"""

import os

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1 to run live tests"),
]


@pytest.fixture(autouse=True)
def _no_network():
    """Override the unit-test network block: these tests are allowed out."""
    yield


@pytest.fixture
def live_settings():
    from app.config import load_settings

    return load_settings()


def test_gemini_model_available(live_settings):
    from app.llm import build_provider

    build_provider(live_settings, "gemini").check_model()


def test_telegram_test_message(live_settings):
    from app.__main__ import make_telegram

    assert make_telegram(live_settings).send_message("🧪 Integration test (not a briefing)").message_id > 0


def test_enabled_feeds_parse(live_settings):
    from app.__main__ import make_fetcher
    from app.collector import collect
    from app.sources import load_sources
    from app.util import utcnow

    sources = [s for s in load_sources(live_settings.config_dir) if s.collectable]
    items, reports = collect(sources, make_fetcher(live_settings), utcnow(), 20, live_settings.collection.max_feed_bytes)
    failing = [r.source_id for r in reports if not r.ok]
    assert not failing, f"Feeds failing: {failing}"
