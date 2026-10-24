import json
import logging

from app.__main__ import RedactingFilter, main
from app.config import load_settings


def env(monkeypatch, tmp_path, **extra):
    monkeypatch.setenv("BCS_STATE_PATH", str(tmp_path / "state" / "state.json"))
    monkeypatch.setenv("BCS_OUTPUT_DIR", str(tmp_path / "out"))
    for k in ("GEMINI_API_KEY", "TELE_BOT_TOKEN", "BOT_INVITE_CODE", "LLM_PROVIDER", "GEMINI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in extra.items():
        monkeypatch.setenv(k, v)


def test_offline_dry_run_needs_no_credentials_and_leaves_state_untouched(monkeypatch, tmp_path):
    env(monkeypatch, tmp_path)
    assert main(["run", "--offline", "--no-pdf"]) == 0
    result = json.loads((tmp_path / "out" / "run-result.json").read_text())
    assert result["outcome"] == "dry_run" and result["stories"] == 3 and result["mcqs"] == 3
    assert not (tmp_path / "state" / "state.json").exists()  # production state never written
    assert (tmp_path / "out" / "edition-2026-10-23-dry-run.txt").exists()


def test_live_run_without_telegram_credentials_fails_cleanly(monkeypatch, tmp_path):
    env(monkeypatch, tmp_path, GEMINI_API_KEY="k")
    assert main(["run"]) == 3


def test_notify_failure_skips_when_already_reported(monkeypatch, tmp_path, capsys):
    env(monkeypatch, tmp_path)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "run-result.json").write_text(json.dumps({"failure_notified": True}))
    assert main(["notify-failure"]) == 0
    assert "not sending" in capsys.readouterr().out


def test_logs_redact_secrets(monkeypatch, tmp_path):
    env(monkeypatch, tmp_path, GEMINI_API_KEY="AIzaSECRETKEY123", TELE_BOT_TOKEN="999:TOKENVALUE")
    settings = load_settings()
    record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed with %s and %s", ("AIzaSECRETKEY123", "bot999:TOKENVALUE/x"), None)
    RedactingFilter(settings).filter(record)
    msg = record.getMessage()
    assert "SECRETKEY" not in msg and "TOKENVALUE" not in msg
    assert "AIzaSECRETKEY123" not in repr(settings) and "TOKENVALUE" not in repr(settings)
