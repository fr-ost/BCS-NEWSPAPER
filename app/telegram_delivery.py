"""Telegram Bot API delivery: chunking, bounded retries, idempotent resend.

The bot token comes from TELE_BOT_TOKEN. Recipients are the subscribers who
pressed Start (see ``subscribers.py``); there is no hard-coded chat id. The
token is part of the request URL (Telegram's design), so it is never logged
and is redacted from any error text.
"""

from __future__ import annotations

import html
import logging
import re
import time
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

TELEGRAM_HARD_LIMIT = 4096
_TAG = re.compile(r"<[^>]+>")


class TelegramError(Exception):
    def __init__(self, message: str, permanent: bool = False, recipient_gone: bool = False):
        super().__init__(message)
        self.permanent = permanent
        # The chat can no longer receive messages (user blocked the bot,
        # deleted their account, or removed the bot from a group).
        self.recipient_gone = recipient_gone


_GONE_HINTS = ("blocked by the user", "user is deactivated", "chat not found", "bot was kicked", "not a member", "group chat was deleted")


def tg_len(text: str) -> int:
    """Length as Telegram counts it (UTF-16 code units). Tags included, which
    over-estimates the visible length and keeps us safely under the limit."""
    return len(text.encode("utf-16-le")) // 2


def strip_html(text: str) -> str:
    return html.unescape(_TAG.sub("", text))


def _split_long_line(line: str, limit: int) -> list[str]:
    """Split a single over-long line. Falls back to plain text so no tag or
    entity is ever cut in half."""
    plain = strip_html(line)
    pieces, current = [], ""
    for word in plain.split(" "):
        candidate = f"{current} {word}" if current else word
        if tg_len(html.escape(candidate, quote=False)) <= limit:
            current = candidate
            continue
        if current:
            pieces.append(html.escape(current, quote=False))
        # A single enormous "word": hard-cut the plain text.
        while tg_len(html.escape(word, quote=False)) > limit:
            cut = limit // 2
            pieces.append(html.escape(word[:cut], quote=False))
            word = word[cut:]
        current = word
    if current:
        pieces.append(html.escape(current, quote=False))
    return pieces


def chunk_sections(sections: list[str], limit: int) -> list[str]:
    """Pack sections into messages <= ``limit``, splitting at section
    boundaries first, then at line boundaries, then (rarely) inside a line."""
    limit = min(limit, TELEGRAM_HARD_LIMIT)
    chunks: list[str] = []
    current = ""
    sep = "\n\n"

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue
        candidate = f"{current}{sep}{section}" if current else section
        if tg_len(candidate) <= limit:
            current = candidate
            continue
        flush()
        if tg_len(section) <= limit:
            current = section
            continue
        for line in section.split("\n"):
            pieces = [line] if tg_len(line) <= limit else _split_long_line(line, limit)
            for piece in pieces:
                candidate = f"{current}\n{piece}" if current else piece
                if tg_len(candidate) <= limit:
                    current = candidate
                else:
                    flush()
                    current = piece
    flush()
    return chunks


@dataclass
class SendResult:
    message_id: int
    fallback_plain: bool = False


class TelegramClient:
    """Bot API client. Recipients are passed per call; there is no fixed chat."""

    def __init__(
        self,
        token: str,
        api_base: str = "https://api.telegram.org",
        timeout: float = 30,
        max_retries: int = 3,
        disable_preview: bool = True,
        min_interval_per_chat: float = 1.1,
        session: requests.Session | None = None,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        if not token:
            raise TelegramError("TELE_BOT_TOKEN must be set", permanent=True)
        self._token = token
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.disable_preview = disable_preview
        self.min_interval_per_chat = min_interval_per_chat
        self.session = session or requests.Session()
        self._sleep = sleep
        self._clock = clock
        self._last_send: dict[str, float] = {}

    def _redact(self, text: str) -> str:
        return text.replace(self._token, "***") if self._token else text

    def _post(self, method: str, payload: dict) -> dict:
        url = f"{self.api_base}/bot{self._token}/{method}"
        last: TelegramError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                # Note: a timeout may hide a delivered message; retries are bounded.
                last = TelegramError(f"Network error calling Telegram: {type(exc).__name__}")
                if attempt < self.max_retries:
                    self._sleep(min(2 ** attempt * 2, 30))
                continue
            try:
                data = resp.json()
            except ValueError:
                data = {}
            if resp.status_code == 200 and data.get("ok") is True:
                return data
            description = self._redact(str(data.get("description", "")))[:300]
            if resp.status_code == 429:
                retry_after = (data.get("parameters") or {}).get("retry_after", 5)
                last = TelegramError(f"Telegram rate limit: {description}")
                if not isinstance(retry_after, (int, float)) or retry_after > 120 or attempt >= self.max_retries:
                    break
                self._sleep(float(retry_after) + 0.5)
                continue
            if resp.status_code >= 500:
                last = TelegramError(f"Telegram server error {resp.status_code}: {description}")
                if attempt < self.max_retries:
                    self._sleep(min(2 ** attempt * 2, 30))
                continue
            # 400/401/403/404/409: our request, credentials or recipient is wrong. Not retryable.
            gone = resp.status_code == 403 or any(h in description.lower() for h in _GONE_HINTS)
            if resp.status_code == 401:
                gone = False  # bad token, not a bad recipient
            raise TelegramError(f"Telegram rejected request ({resp.status_code}): {description}", permanent=True, recipient_gone=gone)
        assert last is not None
        raise last

    def _space(self, chat_id: str) -> None:
        """Telegram allows roughly one message per second per chat."""
        last = self._last_send.get(chat_id)
        if last is not None:
            wait = self.min_interval_per_chat - (self._clock() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_send[chat_id] = self._clock()

    def send_message(self, chat_id: str | int, text: str) -> SendResult:
        chat_id = str(chat_id)
        self._space(chat_id)
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": self.disable_preview},
        }
        try:
            data = self._post("sendMessage", payload)
            return SendResult(message_id=int(data["result"]["message_id"]))
        except TelegramError as exc:
            # A 400 "can't parse entities" means our HTML was rejected. Resend
            # the same content once as plain text rather than lose it.
            if exc.permanent and not exc.recipient_gone and "parse" in str(exc).lower():
                log.warning("Telegram rejected HTML; resending chunk as plain text")
                payload = {k: v for k, v in payload.items() if k != "parse_mode"}
                payload["text"] = strip_html(text)
                data = self._post("sendMessage", payload)
                return SendResult(message_id=int(data["result"]["message_id"]), fallback_plain=True)
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise TelegramError("Telegram response missing message_id") from exc

    def get_updates(self, offset: int | None) -> list[dict]:
        """Fetch pending updates (kept by Telegram for 24 hours).

        Passing ``offset`` = last processed update_id + 1 confirms everything
        before it, so Telegram stops returning those updates.
        """
        payload: dict = {"timeout": 0, "limit": 100, "allowed_updates": ["message", "my_chat_member"]}
        if offset is not None:
            payload["offset"] = offset
        try:
            data = self._post("getUpdates", payload)
        except TelegramError as exc:
            if "409" in str(exc):
                raise TelegramError("A webhook is set for this bot, so getUpdates is disabled. Remove it with deleteWebhook.", permanent=True) from exc
            raise
        result = data.get("result")
        return result if isinstance(result, list) else []

    def get_me(self) -> dict:
        return self._post("getMe", {}).get("result") or {}
