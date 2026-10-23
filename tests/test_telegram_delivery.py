import re

import pytest

from app.render import esc, link
from app.telegram_delivery import TELEGRAM_HARD_LIMIT, TelegramClient, TelegramError, chunk_sections, strip_html, tg_len


def balanced(chunk: str) -> bool:
    stack = []
    for closing, name in re.findall(r"<(/?)([a-z-]+)[^>]*>", chunk):
        if closing:
            if not stack or stack.pop() != name:
                return False
        else:
            stack.append(name)
    return not stack


def test_html_escaping():
    assert esc('<script>alert("x")</script> & co') == '&lt;script&gt;alert("x")&lt;/script&gt; &amp; co'
    assert link('https://e.com/a?x=1&y="2"', "<b>name</b>") == '<a href="https://e.com/a?x=1&amp;y=&quot;2&quot;">&lt;b&gt;name&lt;/b&gt;</a>'
    assert link("javascript:alert(1)", "label") == "label"  # never a link


def test_chunks_respect_limit_and_section_boundaries():
    sections = [f"<b>Section {i}</b>\n" + "\n".join(f"• line {j} " + "অ" * 40 for j in range(10)) for i in range(20)]
    chunks = chunk_sections(sections, 1500)
    assert len(chunks) > 1
    assert all(tg_len(c) <= 1500 for c in chunks)
    assert all(balanced(c) for c in chunks)
    # Sections that fit are never split: each starts a chunk or follows a blank line.
    joined = "\n\n".join(chunks)
    for s in sections:
        assert s in joined


def test_oversized_section_split_by_lines_and_long_line_by_words():
    giant_line = "<b>x</b> " + " ".join(["শব্দ"] * 3000) + " &amp; end"
    chunks = chunk_sections(["header", giant_line, "<i>tail</i>"], 1000)
    assert all(tg_len(c) <= 1000 for c in chunks)
    assert all(balanced(c) for c in chunks)
    assert "&amp;" in "".join(chunks)  # entities stay intact
    assert chunk_sections([], 1000) == []
    assert all(tg_len(c) <= TELEGRAM_HARD_LIMIT for c in chunk_sections(["a" * 10000], 99999))


def test_emoji_counted_as_two_units():
    assert tg_len("📰") == 2


class Resp:
    def __init__(self, status, data):
        self.status_code, self._data = status, data

    def json(self):
        return self._data


class Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, json, timeout):
        self.calls.append((url, json))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


OK = Resp(200, {"ok": True, "result": {"message_id": 77}})


def tg(responses, retries=3):
    sleeps = []
    s = Session(responses)
    return TelegramClient("123:SECRET", session=s, sleep=sleeps.append, max_retries=retries, min_interval_per_chat=0), s, sleeps


def test_send_success_and_payload():
    c, s, _ = tg([OK])
    assert c.send_message("42", "<b>hi</b>").message_id == 77
    url, payload = s.calls[0]
    assert payload["parse_mode"] == "HTML" and payload["chat_id"] == "42"


def test_429_respects_retry_after_with_bounded_retries():
    c, s, sleeps = tg([Resp(429, {"ok": False, "description": "Too Many Requests", "parameters": {"retry_after": 3}}), OK])
    assert c.send_message("42", "x").message_id == 77 and sleeps == [3.5]
    c, s, _ = tg([Resp(500, {"ok": False})] * 10, retries=2)
    with pytest.raises(TelegramError):
        c.send_message("42", "x")
    assert len(s.calls) == 3


def test_api_error_is_permanent_and_redacted():
    c, s, _ = tg([Resp(401, {"ok": False, "description": "Unauthorized for bot123:SECRET"})])
    with pytest.raises(TelegramError) as exc:
        c.send_message("42", "x")
    assert exc.value.permanent and not exc.value.recipient_gone
    assert "SECRET" not in str(exc.value) and len(s.calls) == 1


@pytest.mark.parametrize("status,description", [(403, "Forbidden: bot was blocked by the user"), (400, "Bad Request: chat not found"), (403, "Forbidden: user is deactivated")])
def test_unreachable_recipient_detected(status, description):
    c, _, _ = tg([Resp(status, {"ok": False, "description": description})])
    with pytest.raises(TelegramError) as exc:
        c.send_message("42", "x")
    assert exc.value.recipient_gone


def test_messages_to_same_chat_are_spaced():
    sleeps = []
    clock = iter([0.0, 0.2, 0.2, 5.0])  # set, check+set, set (new chat)
    c = TelegramClient("123:SECRET", session=Session([OK, OK, OK]), sleep=sleeps.append, clock=lambda: next(clock), min_interval_per_chat=1.1)
    c.send_message("1", "a")
    c.send_message("1", "b")  # 0.2s later -> waits ~0.9s
    c.send_message("2", "c")  # different chat -> no wait
    assert len(sleeps) == 1 and abs(sleeps[0] - 0.9) < 1e-9


def test_get_updates_payload_and_webhook_conflict():
    c, s, _ = tg([Resp(200, {"ok": True, "result": [{"update_id": 5}]})])
    assert c.get_updates(5) == [{"update_id": 5}]
    assert s.calls[0][0].endswith("/getUpdates") and s.calls[0][1]["offset"] == 5
    c, _, _ = tg([Resp(409, {"ok": False, "description": "Conflict: can't use getUpdates method while webhook is active"})])
    with pytest.raises(TelegramError, match="webhook"):
        c.get_updates(None)


def test_ok_false_with_200_is_not_success():
    c, s, _ = tg([Resp(200, {"ok": False, "description": "weird"})])
    with pytest.raises(TelegramError):
        c.send_message("42", "x")


def test_html_parse_error_falls_back_to_plain_text():
    c, s, _ = tg([Resp(400, {"ok": False, "description": "Bad Request: can't parse entities"}), OK])
    res = c.send_message("42", "<b>bold &amp; text</b>")
    assert res.fallback_plain
    assert "parse_mode" not in s.calls[1][1] and s.calls[1][1]["text"] == "bold & text"


def test_missing_credentials():
    with pytest.raises(TelegramError):
        TelegramClient("")


def test_strip_html():
    assert strip_html("<a href='x'>a &lt; b</a>") == "a < b"
