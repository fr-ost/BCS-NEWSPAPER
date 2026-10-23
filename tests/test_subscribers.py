from datetime import datetime, timezone

import pytest

from app.state import empty_state
from app.subscribers import (
    MSG_ALREADY,
    MSG_BAD_CODE,
    MSG_FULL,
    MSG_HELP,
    MSG_STOPPED,
    MSG_WELCOME,
    active_subscribers,
    process_updates,
    sync_subscribers,
)
from app.telegram_delivery import TelegramError

from .conftest import FakeTelegram, start_update

NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)
CODE = "friends-2026_abc"


def run(updates, code=CODE, cap=10, state=None):
    state = state or empty_state()
    summary, replies = process_updates(state, updates, code, cap, NOW)
    return state, summary, replies


def test_invite_code_required_when_configured():
    state, summary, replies = run([start_update(1, 11), start_update(2, 12, f"/start wrong-code"), start_update(3, 13, f"/start {CODE}")])
    assert [s["chat_id"] for s in active_subscribers(state)] == ["13"]
    assert replies == {"11": MSG_BAD_CODE, "12": MSG_BAD_CODE, "13": MSG_WELCOME}
    assert summary.rejected == 2


def test_open_subscription_without_code():
    state, _, replies = run([start_update(1, 11)], code="")
    assert [s["chat_id"] for s in active_subscribers(state)] == ["11"] and replies["11"] == MSG_WELCOME


def test_group_command_with_bot_suffix():
    state, _, _ = run([start_update(1, -100500, f"/start@test_bcs_bot {CODE}", chat_type="supergroup")])
    sub = active_subscribers(state)[0]
    assert sub["chat_id"] == "-100500" and sub["type"] == "supergroup"


def test_stop_restart_and_duplicates():
    state, _, replies = run([start_update(1, 11, f"/start {CODE}"), start_update(2, 11, f"/start {CODE}")])
    assert replies["11"] == MSG_ALREADY and len(active_subscribers(state)) == 1
    state, summary, replies = run([start_update(3, 11, "/stop")], state=state)
    assert active_subscribers(state) == [] and replies["11"] == MSG_STOPPED and summary.removed
    state, _, _ = run([start_update(4, 11, f"/start {CODE}")], state=state)
    assert len(active_subscribers(state)) == 1


def test_cap_on_subscribers():
    updates = [start_update(i, 100 + i, f"/start {CODE}") for i in range(1, 5)]
    state, summary, replies = run(updates, cap=3)
    assert len(active_subscribers(state)) == 3 and replies["104"] == MSG_FULL and summary.rejected == 1


def test_blocking_the_bot_unsubscribes():
    state, _, _ = run([start_update(1, 11, f"/start {CODE}")])
    kicked = {"update_id": 2, "my_chat_member": {"chat": {"id": 11, "type": "private"}, "new_chat_member": {"status": "kicked"}}}
    state, summary, _ = run([kicked], state=state)
    assert active_subscribers(state) == [] and state["subscribers"]["11"]["left_reason"] == "bot kicked"


def test_old_updates_are_never_reprocessed():
    state, _, _ = run([start_update(5, 11, f"/start {CODE}")])
    assert state["telegram"]["last_update_id"] == 5
    state, summary, replies = run([start_update(5, 11, "/stop"), start_update(4, 12, f"/start {CODE}")], state=state)
    assert summary.updates == 0 and replies == {} and len(active_subscribers(state)) == 1


def test_chatter_gets_help_in_private_and_is_ignored_in_groups():
    _, _, replies = run([start_update(1, 11, "hello"), start_update(2, -5, "hello", chat_type="group"), start_update(3, 12, "/help")])
    assert replies == {"11": MSG_HELP, "12": MSG_HELP}


def test_malformed_updates_are_skipped():
    state, summary, _ = run([{"update_id": 1}, {"update_id": "x"}, {"update_id": 2, "message": {"text": "/start"}}, {"update_id": 3, "message": "junk"}])
    assert active_subscribers(state) == [] and state["telegram"]["last_update_id"] == 3


def test_sync_replies_and_handles_unreachable_reply():
    tg = FakeTelegram(updates=[start_update(1, 11, f"/start {CODE}"), start_update(2, 12, f"/start {CODE}")])
    tg.gone = {"12"}

    def send(chat_id, text):
        if str(chat_id) == "12":
            raise TelegramError("Forbidden: bot was blocked by the user", permanent=True, recipient_gone=True)
        return FakeTelegram.send_message(tg, chat_id, text)

    tg.send_message = send
    state = empty_state()
    summary = sync_subscribers(tg, state, CODE, 10, NOW)
    assert [s["chat_id"] for s in active_subscribers(state)] == ["11"]
    assert summary.reply_failures == 1 and tg.replies == [("11", MSG_WELCOME)]


def test_invite_code_format_validated(tmp_path):
    from app.config import ConfigError, load_settings

    with pytest.raises(ConfigError):
        load_settings(env={"BOT_INVITE_CODE": "has spaces!"})
    assert load_settings(env={"BOT_INVITE_CODE": CODE}).secrets.bot_invite_code == CODE
