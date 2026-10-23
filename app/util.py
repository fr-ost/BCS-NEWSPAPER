"""Small shared helpers: time, hashing, URL and text normalisation."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]
    ZoneInfoNotFoundError = Exception  # type: ignore[assignment,misc]

# Bangladesh has no DST, so a fixed offset is a safe fallback when the tz
# database is unavailable (e.g. minimal Windows Python installs).
_DHAKA_FALLBACK = timezone(timedelta(hours=6), "Asia/Dhaka")

_TRACKING_PARAMS = re.compile(r"^(utm_[a-z]+|fbclid|gclid|mc_cid|mc_eid|ref|ref_src|igshid|_ga|cmpid|s_cid)$", re.I)
_BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_WORD_RE = re.compile(r"[\wঀ-৿]+", re.UNICODE)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def get_tz(name: str):
    if ZoneInfo is not None:
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError:
            pass
    if name == "Asia/Dhaka":
        return _DHAKA_FALLBACK
    return timezone.utc


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_iso(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text: str, length: int = 16) -> str:
    return sha256(text)[:length]


def is_valid_http_url(url: str | None) -> bool:
    if not url or not isinstance(url, str) or len(url) > 2048:
        return False
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return False
    return parts.scheme in ("http", "https") and bool(parts.netloc) and " " not in parts.netloc


def canonicalize_url(url: str) -> str:
    """Conservative URL normalisation.

    Lower-cases scheme and host, upgrades http to https, drops fragments,
    ``www.`` prefixes, trailing slashes and well-known tracking parameters.
    Meaningful query parameters (e.g. article ids) are kept and sorted.
    """
    parts = urlsplit(url.strip())
    scheme = "https" if parts.scheme.lower() in ("http", "https") else parts.scheme.lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    netloc = host
    if parts.port and parts.port not in (80, 443):
        netloc = f"{host}:{parts.port}"
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if len(path) > 1:
        path = path.rstrip("/")
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False) if not _TRACKING_PARAMS.match(k)]
    query.sort()
    return urlunsplit((scheme, netloc, path, urlencode(query), ""))


def normalize_digits(text: str) -> str:
    return text.translate(_BENGALI_DIGITS)


def normalize_text(text: str) -> str:
    """Lower-case, NFC-normalise, unify digits and collapse whitespace."""
    text = unicodedata.normalize("NFC", text or "")
    text = normalize_digits(text).lower()
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return _WORD_RE.findall(normalize_text(text))


def extract_numbers(text: str) -> set[str]:
    """Numbers in ``text`` with separators removed (``1,250.5`` -> ``1250.5``)."""
    found = set()
    for raw in _NUMBER_RE.findall(normalize_digits(text or "")):
        # Treat commas as thousands separators; keep decimal points.
        cleaned = raw.replace(",", "")
        cleaned = cleaned.rstrip(".")
        if cleaned:
            found.add(cleaned)
    return found


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # Prefer to cut at a sentence or line boundary.
    for sep in ("\n", "। ", ". "):
        idx = cut.rfind(sep)
        if idx > limit * 0.7:
            return cut[: idx + len(sep)].rstrip()
    return cut.rstrip()
