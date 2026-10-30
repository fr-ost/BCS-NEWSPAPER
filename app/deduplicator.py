"""Stage B: deterministic duplicate detection and event grouping.

Articles are first de-duplicated by canonical URL (their id is derived from
it), then grouped into events by exact title/excerpt hash and title-token
similarity. The screening model may later merge further groups (e.g. a
Bengali and an English report of the same event); those merges are applied
with :func:`merge_groups` after validating every identifier.
"""

from __future__ import annotations

from collections.abc import Iterable

from .util import extract_numbers, jaccard, normalize_text, short_hash, tokens

TITLE_SIMILARITY = 0.6
MIN_TOKENS = 4

# Very common words carry no signal about the event.
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "with", "at", "by", "from", "as", "is", "are",
    "was", "be", "says", "say", "said", "new", "over", "after", "amid", "its", "his", "her", "their",
    "ও", "এবং", "এর", "থেকে", "করে", "হবে", "হয়েছে", "নিয়ে", "জন্য", "বলেছেন", "না", "যে", "এ", "করা",
}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic: the lexicographically smaller root wins.
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for x in self.parent:
            out.setdefault(self.find(x), []).append(x)
        return out


def title_signature(title: str) -> set[str]:
    return {t for t in tokens(title) if t not in _STOPWORDS and len(t) > 1}


def is_duplicate(a: dict, b: dict) -> bool:
    if a["canonical_url"] == b["canonical_url"]:
        return True
    ta, tb = normalize_text(a["title"]), normalize_text(b["title"])
    if ta and ta == tb:
        return True
    ea, eb = normalize_text(a.get("excerpt", "")), normalize_text(b.get("excerpt", ""))
    if len(ea) >= 80 and ea == eb:
        return True
    sa, sb = title_signature(a["title"]), title_signature(b["title"])
    if len(sa) >= MIN_TOKENS and len(sb) >= MIN_TOKENS and jaccard(sa, sb) >= TITLE_SIMILARITY:
        return True
    return False


def event_id_for(article_ids: Iterable[str]) -> str:
    return "ev_" + short_hash(min(article_ids), 12)


def assign_events(new_articles: list[dict], existing_articles: list[dict]) -> dict[str, str]:
    """Map each new article id to an event id.

    A new article joins an existing event when it duplicates any article
    already in that event; otherwise duplicates among the new articles form a
    new event. Existing events are never merged here, so stored analyses and
    delivery records stay attached to stable ids.
    """
    uf = UnionFind()
    for art in new_articles:
        uf.add(art["id"])
    # O(n^2) is fine for the few hundred articles kept in the retention window.
    for i, a in enumerate(new_articles):
        for b in new_articles[i + 1 :]:
            if uf.find(a["id"]) != uf.find(b["id"]) and is_duplicate(a, b):
                uf.union(a["id"], b["id"])
    matched: dict[str, set[str]] = {}
    for art in new_articles:
        for old in existing_articles:
            if old.get("event_id") and is_duplicate(art, old):
                matched.setdefault(uf.find(art["id"]), set()).add(old["event_id"])
    mapping: dict[str, str] = {}
    for root, members in uf.groups().items():
        eid = min(matched[root]) if root in matched else event_id_for(members)
        for aid in members:
            mapping[aid] = eid
    return mapping


def merge_groups(groups: dict[str, list[str]], merges: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Apply validated (event_id, event_id) merge pairs from the screening model."""
    uf = UnionFind()
    for eid in groups:
        uf.add(eid)
    for a, b in merges:
        if a in groups and b in groups and a != b:
            uf.union(a, b)
    merged: dict[str, list[str]] = {}
    for root, eids in uf.groups().items():
        members: list[str] = []
        for eid in eids:
            members.extend(groups[eid])
        merged[min(eids)] = sorted(set(members))
    return merged


def distinctive_numbers(texts: Iterable[str]) -> set[str]:
    """Numbers specific enough to identify a story (figures like 41879 or
    3.89), not small counts, days or years."""
    out = set()
    for text in texts:
        for n in extract_numbers(text or ""):
            if "." in n:
                if len(n.replace(".", "").lstrip("0")) >= 2:
                    out.add(n)
            elif len(n.lstrip("0")) >= 3 and not (len(n) == 4 and 1900 <= int(n) <= 2100):
                out.add(n)
    return out


def same_story(a_numbers: set[str], a_title: str, b_numbers: set[str], b_title: str) -> bool:
    """Deterministic repeat check for two notes on the same syllabus topic."""
    shared = a_numbers & b_numbers
    if len(shared) >= 2:
        return True
    return bool(shared) and jaccard(title_signature(a_title), title_signature(b_title)) >= 0.3
