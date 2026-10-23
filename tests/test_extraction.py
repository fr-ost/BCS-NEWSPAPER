from app.extractor import EXCERPT_ONLY, FULL, Extraction, choose_analysis_input, clean_article_html, extract_article
from app.http import FetchError, FetchResult, RobotsDisallowed

PAGE = """<html><head><script>evil()</script><style>p{}</style></head><body>
<header><nav>Home | World | Business | Sports menu items here</nav></header>
<div class="ad-container"><p>Advertisement: buy our product right now please</p></div>
<article><div itemprop="articleBody">
<p>The central bank raised the policy rate to 10 percent on Monday, the third hike this year.</p>
<p>Officials said inflation stood at 9.2 percent in September and remained the main concern.</p>
<div class="related-stories"><p>Related: Ten things you need to know about the economy today</p></div>
<p>Also read: another story that should be removed from the body</p>
</div></article>
<aside><p>Most read stories of the week on our website</p></aside>
<footer><p>Copyright 2026 Example News. All rights reserved worldwide.</p></footer>
</body></html>"""


def test_clean_html_removes_boilerplate():
    text = clean_article_html(PAGE, 10_000)
    assert "policy rate to 10 percent" in text
    assert "9.2 percent" in text
    for noise in ("evil", "Advertisement", "Related", "Also read", "Most read", "Copyright", "menu items"):
        assert noise not in text


def test_text_length_limit_enforced():
    long_page = "<article>" + "".join(f"<p>Sentence number {i} with enough words to count.</p>" for i in range(500)) + "</article>"
    assert len(clean_article_html(long_page, 1000)) <= 1000


class StubFetcher:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def get(self, url, **kwargs):
        if self.error:
            raise self.error
        return self.result


def test_extraction_failure_is_reported_not_raised():
    ex = extract_article(StubFetcher(error=FetchError("HTTP 403", permanent=True)), "https://e.com/a", 10**6, 100, 5000)
    assert ex.status == "failed" and "403" in ex.reason
    ex = extract_article(StubFetcher(error=RobotsDisallowed("https://e.com/a")), "https://e.com/a", 10**6, 100, 5000)
    assert ex.status == "blocked"
    pdf = FetchResult("u", "u", 200, b"%PDF", "application/pdf")
    assert extract_article(StubFetcher(result=pdf), "https://e.com/a", 10**6, 100, 5000).status == "failed"


def test_short_page_is_too_short():
    page = FetchResult("u", "u", 200, b"<article><p>Only a short teaser paragraph behind a paywall.</p></article>", "text/html")
    ex = extract_article(StubFetcher(result=page), "https://e.com/a", 10**6, 600, 5000)
    assert ex.status == "too_short"


def test_excerpt_only_classification():
    excerpt = "x " * 150
    assert choose_analysis_input(Extraction(status="ok", text="full text"), excerpt, 200) == (FULL, "full text")
    mode, text = choose_analysis_input(Extraction(status="failed"), excerpt, 200)
    assert mode == EXCERPT_ONLY and text == excerpt
    # A short paywall teaser never counts as the full article.
    mode, _ = choose_analysis_input(Extraction(status="too_short", text="y " * 200), excerpt, 200)
    assert mode == EXCERPT_ONLY
    # Not enough evidence at all -> skip.
    assert choose_analysis_input(Extraction(status="failed"), "tiny", 200) is None
