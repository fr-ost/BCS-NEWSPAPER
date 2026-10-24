"""Turn the edition HTML into a PDF with headless Chrome/Chromium.

GitHub's Ubuntu runners ship Google Chrome, and the workflow installs the
Noto fonts (which include Bengali). Locally any Chrome/Chromium works; set
CHROME_PATH to pick one. If no browser is found the edition is still sent
as text, just without the PDF.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

CANDIDATES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "/opt/google/chrome/chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)
MAX_PDF_BYTES = 20 * 1024 * 1024  # Telegram bots may send up to 50 MB


class PdfError(Exception):
    pass


def find_browser(env: dict | None = None) -> str | None:
    env = os.environ if env is None else env
    if env.get("CHROME_PATH"):
        return env["CHROME_PATH"] if Path(env["CHROME_PATH"]).exists() else None
    for name in CANDIDATES:
        found = shutil.which(name) or (name if Path(name).is_file() else None)
        if found:
            return found
    for path in sorted(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome")):  # Playwright installs
        return str(path)
    return None


def html_to_pdf(html: str, browser: str, timeout: float = 90) -> bytes:
    with tempfile.TemporaryDirectory(prefix="bcs-pdf-") as tmp:
        src = Path(tmp) / "edition.html"
        out = Path(tmp) / "edition.pdf"
        src.write_text(html, encoding="utf-8")
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",  # runners/containers lack the user namespaces Chrome's sandbox needs
            "--no-first-run",
            "--disable-extensions",
            # Scripts are blocked by the page's Content-Security-Policy instead of
            # --blink-settings=scriptEnabled=false, which stops headless printing.
            "--no-pdf-header-footer",
            "--print-to-pdf-no-header",
            f"--user-data-dir={tmp}/profile",
            f"--print-to-pdf={out}",
            src.as_uri(),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PdfError(f"Browser failed to run: {type(exc).__name__}") from exc
        if not out.exists() or out.stat().st_size == 0:
            raise PdfError(f"Browser produced no PDF (exit {proc.returncode}): {proc.stderr.decode(errors='replace')[-300:]}")
        data = out.read_bytes()
    if len(data) > MAX_PDF_BYTES:
        raise PdfError(f"PDF too large ({len(data)} bytes)")
    return data


def make_pdf_builder(env: dict | None = None):
    """Return an HTML->PDF callable, or None when no browser is available."""
    browser = find_browser(env)
    if not browser:
        log.warning("No Chrome/Chromium found; editions will be sent without a PDF (set CHROME_PATH)")
        return None
    log.info("PDF renderer: %s", browser)
    return lambda html: html_to_pdf(html, browser)
