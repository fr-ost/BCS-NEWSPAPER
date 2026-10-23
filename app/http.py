"""Polite HTTP fetching with timeouts, size caps, bounded retries and robots.txt."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from urllib import robotparser
from urllib.parse import urlsplit

import requests

log = logging.getLogger(__name__)


class FetchError(Exception):
    """A fetch failed. ``permanent`` errors are not worth retrying."""

    def __init__(self, message: str, permanent: bool = False, status: int | None = None):
        super().__init__(message)
        self.permanent = permanent
        self.status = status


class RobotsDisallowed(FetchError):
    def __init__(self, url: str):
        super().__init__(f"robots.txt disallows {url}", permanent=True)


@dataclass
class FetchResult:
    url: str
    final_url: str
    status: int
    content: bytes
    content_type: str


class Fetcher:
    """HTTP client used for feeds and article pages.

    Never impersonates a browser and never retries 401/403/404/451, which are
    access decisions by the publisher.
    """

    def __init__(
        self,
        user_agent: str,
        timeout: float = 20,
        retries: int = 2,
        respect_robots: bool = True,
        session: requests.Session | None = None,
        sleep=time.sleep,
    ):
        self.user_agent = user_agent
        self.timeout = timeout
        self.retries = retries
        self.respect_robots = respect_robots
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Language": "bn,en;q=0.8"})
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}
        self._sleep = sleep

    def allowed_by_robots(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            parser = robotparser.RobotFileParser()
            try:
                resp = self.session.get(f"{origin}/robots.txt", timeout=min(self.timeout, 15))
                if resp.status_code in (401, 403):
                    # Treat a forbidden robots.txt as "disallow all", per RFC 9309 guidance.
                    parser.parse(["User-agent: *", "Disallow: /"])
                elif resp.status_code >= 400:
                    parser.parse([])  # no robots.txt -> allowed
                else:
                    parser.parse(resp.text[:500_000].splitlines())
            except requests.RequestException as exc:
                log.warning("robots.txt unavailable for %s (%s); treating as allowed", origin, type(exc).__name__)
                parser.parse([])
            self._robots[origin] = parser
        parser = self._robots[origin]
        return parser is None or parser.can_fetch(self.user_agent, url)

    def get(
        self,
        url: str,
        max_bytes: int,
        timeout: float | None = None,
        retries: int | None = None,
        accept: str = "*/*",
    ) -> FetchResult:
        if not self.allowed_by_robots(url):
            raise RobotsDisallowed(url)
        attempts = 1 + (self.retries if retries is None else retries)
        last_error: FetchError | None = None
        for attempt in range(attempts):
            try:
                return self._get_once(url, max_bytes, timeout or self.timeout, accept)
            except FetchError as exc:
                last_error = exc
                if exc.permanent or attempt == attempts - 1:
                    break
                delay = min(2 ** attempt * 2, 20) + random.uniform(0, 1)
                log.info("Retrying %s in %.1fs (%s)", url, delay, exc)
                self._sleep(delay)
        assert last_error is not None
        raise last_error

    def _get_once(self, url: str, max_bytes: int, timeout: float, accept: str) -> FetchResult:
        try:
            with self.session.get(url, timeout=timeout, stream=True, headers={"Accept": accept}, allow_redirects=True) as resp:
                status = resp.status_code
                if status in (401, 402, 403, 404, 410, 451):
                    raise FetchError(f"HTTP {status} for {url}", permanent=True, status=status)
                if status == 429 or status >= 500:
                    raise FetchError(f"HTTP {status} for {url}", permanent=False, status=status)
                if status >= 400:
                    raise FetchError(f"HTTP {status} for {url}", permanent=True, status=status)
                declared = resp.headers.get("Content-Length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    raise FetchError(f"Response too large ({declared} bytes) for {url}", permanent=True)
                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_content(chunk_size=65536):
                    total += len(chunk)
                    if total > max_bytes:
                        raise FetchError(f"Response exceeded {max_bytes} bytes for {url}", permanent=True)
                    chunks.append(chunk)
                return FetchResult(
                    url=url,
                    final_url=resp.url,
                    status=status,
                    content=b"".join(chunks),
                    content_type=resp.headers.get("Content-Type", ""),
                )
        except requests.Timeout as exc:
            raise FetchError(f"Timeout fetching {url}") from exc
        except requests.ConnectionError as exc:
            raise FetchError(f"Connection error fetching {url}") from exc
        except requests.RequestException as exc:
            raise FetchError(f"Request failed for {url}: {type(exc).__name__}", permanent=True) from exc
