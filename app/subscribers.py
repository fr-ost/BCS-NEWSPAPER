"""Subscriber management without a server.

People subscribe by pressing Start in the bot (ideally through an invite
link ``https://t.me/<bot>?start=<code>``). There is no always-on process:
each run reads pending messages with ``getUpdates`` (Telegram keeps them for
24 hours), updates the subscriber list in state, and replies.

Commands: ``/start [code]`` subscribe, ``/stop`` unsubscribe, ``/help``.
A user who blocks the bot is removed automatically.
"""

from __future__ import annotations

import hmac
import logging
from dataclasses import dataclass, field
from datetime import datetime

from .telegram_delivery import TelegramClient, TelegramError
from .util import iso

log = logging.getLogger(__name__)

MAX_UPDATE_PAGES = 5
NAME_LIMIT = 40

MSG_WELCOME = (
    "✅ <b>বিসিএস দৈনিক ব্রিফিং-এ আপনাকে যুক্ত করা হয়েছে।</b>\n"
    "প্রতিদিন সকাল প্রায় ৮টায় (বাংলাদেশ সময়) ব্রিফিং আসবে।\n"
    "বন্ধ করতে /stop লিখুন।"
)
MSG_ALREADY = "আপনি আগে থেকেই যুক্ত আছেন। বন্ধ করতে /stop লিখুন।"
MSG_BAD_CODE = "❌ আমন্ত্রণ লিংক/কোডটি সঠিক নয়। যিনি আপনাকে আমন্ত্রণ জানিয়েছেন, তার কাছ থেকে সঠিক লিংক নিন।"
MSG_FULL = "দুঃখিত, সদস্যসংখ্যা পূর্ণ হয়ে গেছে। বটের মালিকের সঙ্গে যোগাযোগ করুন।"
MSG_STOPPED = "আপনাকে তালিকা থেকে বাদ দেওয়া হয়েছে। আবার যুক্ত হতে আমন্ত্রণ লিংকে ক্লিক করুন বা /start লিখুন।"
MSG_HELP = (
    "এই বট প্রতিদিন সকালে বিসিএস কারেন্ট অ্যাফেয়ার্স ব্রিফিং পাঠায়।\n"
    "/start — যুক্ত হোন (আমন্ত্রণ লিংক ব্যবহার করুন)\n"
    "/stop — বন্ধ করুন\n"
    "নোট: বট দিনে একবার বার্তা দেখে, তাই উত্তর আসতে কয়েক ঘণ্টা লাগতে পারে।"
)


@dataclass
class SyncSummary:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    rejected: int = 0
    updates: int = 0
    reply_failures: int = 0


def ensure_structures(state: dict) -> None:
    state.setdefault("subscribers", {})
    state.setdefault("telegram", {"last_update_id": None})


def active_subscribers(state: dict) -> list[dict]:
    subs = [s for s in state.get("subscribers", {}).values() if s.get("status") == "active"]
    return sorted(subs, key=lambda s: (s.get("joined_at") or "", s["chat_id"]))


def deactivate(state: dict, chat_id: str, reason: str, now: datetime) -> bool:
    sub = state.get("subscribers", {}).get(str(chat_id))
    if not sub or sub.get("status") != "active":
        return False
    sub.update(status="inactive", left_at=iso(now), left_reason=reason)
    return True


def _chat_name(chat: dict) -> str:
    name = chat.get("title") or " ".join(x for x in (chat.get("first_name"), chat.get("last_name")) if x) or chat.get("username") or "?"
    return str(name)[:NAME_LIMIT]


def _parse_command(text: str) -> tuple[str, str]:
    """``'/start@MyBot abc'`` -> ``('/start', 'abc')``."""
    parts = text.strip().split(maxsplit=1)
    if not parts or not parts[0].startswith("/"):
        return "", ""
    command = parts[0].split("@", 1)[0].lower()
    return command, (parts[1].strip() if len(parts) > 1 else "")


def _code_ok(given: str, expected: str) -> bool:
    if not expected:
        return True  # open subscription (capped by max_subscribers)
    return hmac.compare_digest(given.encode(), expected.encode())


def process_updates(state: dict, updates: list[dict], invite_code: str, max_subscribers: int, now: datetime) -> tuple[SyncSummary, dict[str, str]]:
    """Apply updates to state. Returns the summary and one reply per chat."""
    ensure_structures(state)
    subs = state["subscribers"]
    summary = SyncSummary()
    replies: dict[str, str] = {}
    last_id = state["telegram"].get("last_update_id")

    valid = [u for u in updates if isinstance(u, dict) and isinstance(u.get("update_id"), int) and not isinstance(u["update_id"], bool)]
    for update in sorted(valid, key=lambda u: u["update_id"]):
        uid = update["update_id"]
        if last_id is not None and uid <= last_id:
            continue
        last_id = uid
        summary.updates += 1

        member = update.get("my_chat_member")
        if isinstance(member, dict):
            chat = member.get("chat") or {}
            status = (member.get("new_chat_member") or {}).get("status")
            if status in ("kicked", "left") and deactivate(state, str(chat.get("id")), f"bot {status}", now):
                summary.removed.append(_chat_name(chat))
            continue

        msg = update.get("message")
        if not isinstance(msg, dict):
            continue
        chat = msg.get("chat") or {}
        if "id" not in chat:
            continue
        chat_id = str(chat["id"])
        command, arg = _parse_command(str(msg.get("text") or ""))
        current = subs.get(chat_id)
        is_active = bool(current and current.get("status") == "active")

        if command == "/start":
            if is_active:
                replies[chat_id] = MSG_ALREADY
            elif not _code_ok(arg, invite_code):
                summary.rejected += 1
                replies[chat_id] = MSG_BAD_CODE
            elif len(active_subscribers(state)) >= max_subscribers:
                summary.rejected += 1
                replies[chat_id] = MSG_FULL
            else:
                subs[chat_id] = {
                    "chat_id": chat_id,
                    "name": _chat_name(chat),
                    "type": str(chat.get("type", "private")),
                    "status": "active",
                    "joined_at": iso(now),
                    "left_at": None,
                    "left_reason": None,
                }
                summary.added.append(_chat_name(chat))
                replies[chat_id] = MSG_WELCOME
        elif command == "/stop":
            if deactivate(state, chat_id, "/stop", now):
                summary.removed.append(_chat_name(chat))
            replies[chat_id] = MSG_STOPPED
        elif command == "/help" or (command == "" and chat.get("type") == "private"):
            replies.setdefault(chat_id, MSG_HELP)
        # Other commands, and ordinary chatter in groups, are ignored.

    state["telegram"]["last_update_id"] = last_id
    return summary, replies


def sync_subscribers(client: TelegramClient, state: dict, invite_code: str, max_subscribers: int, now: datetime) -> SyncSummary:
    """Read pending updates, update subscribers, send replies (best effort)."""
    ensure_structures(state)
    total = SyncSummary()
    all_replies: dict[str, str] = {}
    for _ in range(MAX_UPDATE_PAGES):
        last = state["telegram"].get("last_update_id")
        updates = client.get_updates(None if last is None else last + 1)
        summary, replies = process_updates(state, updates, invite_code, max_subscribers, now)
        total.added += summary.added
        total.removed += summary.removed
        total.rejected += summary.rejected
        total.updates += summary.updates
        all_replies.update(replies)
        if len(updates) < 100:
            break
    for chat_id, text in all_replies.items():
        try:
            client.send_message(chat_id, text)
        except TelegramError as exc:
            total.reply_failures += 1
            if exc.recipient_gone:
                deactivate(state, chat_id, "unreachable", now)
            log.warning("Could not reply to a subscriber: %s", exc)
    if total.updates:
        log.info("Subscribers: +%d -%d, %d rejected, %d active", len(total.added), len(total.removed), total.rejected, len(active_subscribers(state)))
    return total
