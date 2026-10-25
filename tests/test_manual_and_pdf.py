"""Manual (extra) editions and the PDF attachment. All external calls mocked."""

import json
from datetime import date

import pytest

from app.pdf import PdfError, find_browser, html_to_pdf
from app.render import render_pdf_html
from app.telegram_delivery import TelegramClient

from .conftest import NOW, FakeTelegram, start_update
from .test_pipeline import Programmable

FAKE_PDF = b"%PDF-1.7 fake edition"


def pdf_builder(html: str) -> bytes:
    assert "<html" in html
    return FAKE_PDF


def load_state(settings):
    return json.loads(settings.state_path.read_text(encoding="utf-8"))


# ------------------------------------------------------------ manual runs


def test_manual_run_after_daily_edition_sends_an_extra_edition(make_pipeline, settings):
    tg1 = FakeTelegram()
    assert make_pipeline(telegram=tg1).run(now=NOW).outcome == "delivered"
    daily_text = "\n".join(tg1.sent)

    tg2 = FakeTelegram()
    r = make_pipeline(provider=Programmable(), telegram=tg2).run(now=NOW, manual=True)
    assert r.outcome == "delivered" and r.edition_date == "2026-10-23-m1"
    text = "\n".join(tg2.sent)
    assert "অতিরিক্ত আপডেট" in text and "নতুন কোনো প্রাসঙ্গিক" in text
    assert "Cabinet approves" in daily_text and "Cabinet approves" not in text  # nothing repeated

    tg3 = FakeTelegram()
    assert make_pipeline(telegram=tg3).run(now=NOW, manual=True).edition_date == "2026-10-23-m2"
    assert tg3.sent  # unlimited manual runs
    # The scheduled run is still limited to once a day.
    tg4 = FakeTelegram()
    assert make_pipeline(telegram=tg4).run(now=NOW).outcome == "already_delivered" and tg4.sent == []


def test_manual_run_first_does_not_block_scheduled_run(make_pipeline, settings):
    tg = FakeTelegram()
    r1 = make_pipeline(telegram=tg).run(now=NOW, manual=True)
    assert r1.outcome == "delivered" and r1.stories == 3
    tg2 = FakeTelegram()
    r2 = make_pipeline(telegram=tg2).run(now=NOW)
    assert r2.outcome == "delivered" and r2.edition_date == "2026-10-23"
    assert r2.stories == 0 and tg2.sent == []  # nothing new: no message to everyone
    tg3 = FakeTelegram()
    assert make_pipeline(telegram=tg3).run(now=NOW).outcome == "already_delivered" and tg3.sent == []  # backup cron is a no-op
    state = load_state(settings)
    assert state["editions"]["2026-10-23-m1"]["manual"] is True
    assert state["editions"]["2026-10-23"]["manual"] is False


def test_manual_run_resumes_unfinished_edition_instead_of_stacking(make_pipeline, settings):
    tg1 = FakeTelegram(fail_on={2})
    assert make_pipeline(telegram=tg1).run(now=NOW).outcome == "failed"
    tg2 = FakeTelegram()
    r = make_pipeline(provider=Programmable(), telegram=tg2).run(now=NOW, manual=True)
    assert r.outcome == "delivered" and r.edition_date == "2026-10-23"
    assert not any(k.endswith("-m1") for k in load_state(settings)["editions"])


# ------------------------------------------------------------ PDF delivery


def test_pdf_sent_to_each_subscriber_after_all_messages(make_pipeline, settings):
    tg = FakeTelegram(subscribers=2)
    r = make_pipeline(telegram=tg, pdf_builder=pdf_builder).run(now=NOW)
    assert r.outcome == "delivered"
    assert [(c, f, b) for c, f, b, _ in tg.documents] == [
        ("101", "BCS-Briefing-2026-10-23.pdf", FAKE_PDF),
        ("102", "BCS-Briefing-2026-10-23.pdf", FAKE_PDF),
    ]
    assert "PDF" in tg.documents[0][3]
    edition = load_state(settings)["editions"]["2026-10-23"]
    assert edition["pdf"]["bytes"] == len(FAKE_PDF)
    assert all(rec["pdf_message_id"] for rec in edition["deliveries"].values())
    assert (settings.output_dir / "edition-2026-10-23.pdf").read_bytes() == FAKE_PDF


def test_manual_edition_pdf_name(make_pipeline):
    tg = FakeTelegram()
    make_pipeline(telegram=tg, pdf_builder=pdf_builder).run(now=NOW, manual=True)
    assert tg.documents[0][1] == "BCS-Briefing-2026-10-23-extra1.pdf"


def test_pdf_failure_never_blocks_text_edition(make_pipeline, settings):
    def broken(html):
        raise PdfError("no fonts")

    tg = FakeTelegram()
    r = make_pipeline(telegram=tg, pdf_builder=broken).run(now=NOW)
    assert r.outcome == "delivered" and tg.sent and tg.documents == []
    assert "no fonts" in load_state(settings)["editions"]["2026-10-23"]["pdf_error"]


def test_failed_pdf_send_is_retried_alone_next_run(make_pipeline, settings):
    from app.telegram_delivery import TelegramError

    tg1 = FakeTelegram()

    def flaky_document(*args, **kwargs):
        raise TelegramError("Telegram server error 502")

    tg1.send_document = flaky_document
    assert make_pipeline(telegram=tg1, pdf_builder=pdf_builder).run(now=NOW).outcome == "failed"
    n_text = len(tg1.sent)
    assert n_text >= 1
    record = load_state(settings)["editions"]["2026-10-23"]["deliveries"]["101"]
    assert record["status"] == "pending" and len(record["sent"]) == n_text and "pdf_message_id" not in record

    # Next run on a fresh machine: the PDF file is gone and is rebuilt from saved state.
    (settings.output_dir / "edition-2026-10-23.pdf").unlink()
    calls = []

    def builder(html):
        calls.append(html)
        return FAKE_PDF

    tg2 = FakeTelegram()
    assert make_pipeline(telegram=tg2, pdf_builder=builder).run(now=NOW).outcome == "delivered"
    assert tg2.sent == [] and len(tg2.documents) == 1 and len(calls) == 1  # only the PDF, no repeated text


def test_no_pdf_for_empty_edition(make_pipeline):
    tg = FakeTelegram()
    make_pipeline(provider=Programmable(reject_all=True), telegram=tg, pdf_builder=pdf_builder).run(now=NOW)
    assert tg.sent and tg.documents == []


def test_blocked_user_on_pdf_is_deactivated(make_pipeline, settings):
    tg = FakeTelegram(subscribers=2)
    original = tg.send_document

    def send_document(chat_id, *a, **k):
        if str(chat_id) == "102":
            tg.gone.add("102")
        return original(chat_id, *a, **k)

    tg.send_document = send_document
    r = make_pipeline(telegram=tg, pdf_builder=pdf_builder).run(now=NOW)
    assert r.outcome == "delivered"
    assert load_state(settings)["subscribers"]["102"]["status"] == "inactive"


# ------------------------------------------------------------ rendering and tooling


def test_pdf_html_is_escaped_and_locked_down(make_pipeline, settings, sources):
    story = {
        "event_id": "e1",
        "analysis": {
            "headline_bn": "<script>alert(1)</script> শিরোনাম",
            "categories": ["economy"],
            "exam_focus": "both",
            "score": 70,
            "mode": "excerpt_only",
            "what_happened": "ঘটনা & <b>bold</b>",
            "why_it_matters": "",
            "key_facts": [{"fact": "তথ্য ১", "evidence": "x", "prelims": True}],
            "attributed_claims": [],
            "institutions": [],
        },
        "published_at": None,
        "date_status": "missing",
        "source_names": ["X"],
        "links": [("X", "https://example.com/a"), ("Bad", "javascript:alert(1)")],
        "primary_url": "https://example.com/a",
        "primary_source": "X",
    }
    mcq = {"question": "প্রশ্ন?", "options": ["ক", "খ", "গ", "ঘ"], "answer_index": 2, "explanation": "কারণ",
           "subject": "economy", "source_url": "https://example.com/a", "source_name": "X"}
    html = render_pdf_html(date(2026, 10, 23), [story], [mcq], ["আংশিক"], sources)
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "Content-Security-Policy" in html and "default-src 'none'" in html
    assert 'href="javascript:' not in html
    assert "শুধু সংক্ষিপ্তাংশ" in html and "আংশিক সংস্করণ" in html
    assert html.index("প্রশ্ন ১.") < html.index('class="answers"')  # answers after questions, on a new page


def test_find_browser_respects_chrome_path(tmp_path):
    fake = tmp_path / "chrome"
    fake.write_text("")
    assert find_browser({"CHROME_PATH": str(fake)}) == str(fake)
    assert find_browser({"CHROME_PATH": str(tmp_path / "missing")}) is None


def test_html_to_pdf_reports_browser_failure():
    with pytest.raises(PdfError):
        html_to_pdf("<html></html>", "/bin/false", timeout=10)
    with pytest.raises(PdfError):
        html_to_pdf("<html></html>", "/nonexistent/chrome", timeout=10)


def test_send_document_is_multipart():
    seen = {}

    class Resp:
        status_code = 200

        def json(self):
            return {"ok": True, "result": {"message_id": 9}}

    class Session:
        def post(self, url, timeout, json=None, data=None, files=None):
            seen.update(url=url, data=data, files=files, json=json)
            return Resp()

    c = TelegramClient("123:SECRET", session=Session(), sleep=lambda s: None, min_interval_per_chat=0)
    assert c.send_document("42", FAKE_PDF, "a.pdf", "cap").message_id == 9
    assert seen["url"].endswith("/sendDocument") and seen["json"] is None
    assert seen["data"] == {"chat_id": "42", "caption": "cap"}
    assert seen["files"]["document"] == ("a.pdf", FAKE_PDF, "application/pdf")


@pytest.mark.skipif(find_browser() is None, reason="no Chrome/Chromium installed")
def test_real_browser_renders_pdf(sources):
    html = render_pdf_html(date(2026, 10, 23), [], [], [], sources)
    assert html_to_pdf(html, find_browser()).startswith(b"%PDF")
