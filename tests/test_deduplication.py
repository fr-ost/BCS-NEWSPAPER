from app.deduplicator import assign_events, is_duplicate, merge_groups
from app.util import canonicalize_url, is_valid_http_url


def art(id, title, url=None, excerpt="", event_id=None):
    return {"id": id, "title": title, "canonical_url": url or f"https://example.com/{id}", "excerpt": excerpt, "event_id": event_id}


def test_canonicalize_url_is_conservative():
    assert canonicalize_url("HTTP://WWW.Example.com//news/a/?utm_source=fb&id=5#top") == "https://example.com/news/a?id=5"
    # Meaningful query parameters survive and are sorted.
    assert canonicalize_url("https://example.com/p?b=2&a=1") == "https://example.com/p?a=1&b=2"


def test_invalid_urls_rejected():
    for bad in (None, "", "ftp://x.com/a", "javascript:alert(1)", "https://", "not a url", "https://exa mple.com/"):
        assert not is_valid_http_url(bad)
    assert is_valid_http_url("https://example.com/a")


def test_duplicate_detection():
    a = art("1", "Cabinet approves draft amendment to Bangladesh Labour Act")
    b = art("2", "Cabinet approves draft amendment to the Bangladesh Labour Act")
    c = art("3", "Remittance inflow rises in September")
    assert is_duplicate(a, b)
    assert not is_duplicate(a, c)
    # Same canonical URL is always a duplicate.
    assert is_duplicate(a, art("4", "Different", url=a["canonical_url"]))
    # Identical substantial excerpt.
    long = "x" * 100
    assert is_duplicate(art("5", "A", excerpt=long), art("6", "B", excerpt=long))


def test_short_titles_not_merged_by_similarity():
    assert not is_duplicate(art("1", "Budget passed"), art("2", "Budget delayed"))


def test_assign_events_groups_new_and_joins_existing():
    existing = [art("old", "Bangladesh Nepal sign hydropower trade memorandum", event_id="ev_existing")]
    new = [
        art("n1", "Bangladesh, Nepal sign hydropower trade memorandum"),
        art("n2", "Cabinet approves draft amendment to Bangladesh Labour Act"),
        art("n3", "Cabinet approves draft amendment to the Bangladesh Labour Act"),
    ]
    mapping = assign_events(new, existing)
    assert mapping["n1"] == "ev_existing"
    assert mapping["n2"] == mapping["n3"] != "ev_existing"
    # Deterministic ids.
    assert assign_events(new, existing) == mapping


def test_merge_groups_validates_ids():
    groups = {"ev_a": ["1"], "ev_b": ["2"], "ev_c": ["3"]}
    merged = merge_groups(groups, [("ev_a", "ev_b"), ("ev_a", "ev_unknown"), ("ev_c", "ev_c")])
    assert merged == {"ev_a": ["1", "2"], "ev_c": ["3"]}
