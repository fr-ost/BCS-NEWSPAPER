from datetime import datetime, timezone

from app.collector import collect, parse_feed
from app.http import FetchError
from app.sources import Source

NOW = datetime(2026, 10, 23, 2, 0, tzinfo=timezone.utc)


def src(id="s1", feed="https://example.com/feed", enabled=True):
    return Source(
        id=id, name=id.upper(), homepage="https://example.com/", feed_url=feed, type="newspaper", language="en",
        subjects=("economy",), enabled=enabled, fetch_articles=True, timeout_seconds=10, max_retries=1,
    )


RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>Good &lt;b&gt;item&lt;/b&gt;</title><link>https://www.example.com/a/1?utm_source=x</link>
<description>&lt;p&gt;Body text&lt;/p&gt;</description><pubDate>Thu, 22 Oct 2026 10:00:00 +0600</pubDate></item>
<item><title>No date</title><link>https://example.com/a/2</link></item>
<item><title>Bad date</title><link>https://example.com/a/3</link><pubDate>not a date</pubDate></item>
<item><title>Future date</title><link>https://example.com/a/4</link><pubDate>Thu, 22 Oct 2099 10:00:00 +0600</pubDate></item>
<item><title>Bad url</title><link>javascript:alert(1)</link></item>
<item><title></title><link>https://example.com/a/5</link></item>
<item><title>Duplicate of first</title><link>http://example.com/a/1/</link></item>
</channel></rss>"""


def test_parse_feed_handles_malformed_items():
    items, report = parse_feed(RSS, src(), NOW, 50)
    titles = [i.title for i in items]
    assert titles == ["Good item", "No date", "Bad date", "Future date"]
    assert report.ok and report.skipped == 3  # bad url, empty title, duplicate canonical URL
    first = items[0]
    assert first.excerpt == "Body text"
    assert first.canonical_url == "https://example.com/a/1"
    assert first.published_at == "2026-10-22T04:00:00+00:00"
    assert first.date_status == "ok"


def test_missing_invalid_and_future_dates_do_not_crash():
    items, _ = parse_feed(RSS, src(), NOW, 50)
    status = {i.title: (i.published_at, i.date_status) for i in items}
    assert status["No date"] == (None, "missing")
    assert status["Bad date"][1] in ("invalid", "missing")
    assert status["Bad date"][0] is None
    assert status["Future date"] == (None, "future")


def test_garbage_feed_reports_error():
    items, report = parse_feed(b"<html><body>not a feed", src(), NOW, 50)
    assert items == []
    assert report.ok is False or report.warnings


def test_one_failing_source_does_not_stop_others():
    class Fetcher:
        def get(self, url, **kwargs):
            if "bad" in url:
                raise FetchError("HTTP 503", permanent=False)
            from types import SimpleNamespace
            return SimpleNamespace(content=RSS, status=200)

    sources = [src("bad", "https://bad.example.com/feed"), src("good"), src("off", enabled=False)]
    items, reports = collect(sources, Fetcher(), NOW, 50, 1_000_000)
    assert {r.source_id: r.ok for r in reports} == {"bad": False, "good": True}
    assert len(items) == 4 and all(i.source_id == "good" for i in items)
