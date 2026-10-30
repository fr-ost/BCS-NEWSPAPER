"""Render an edition as Telegram-safe HTML sections.

Invariant: every rendered *line* is self-contained, balanced HTML. All text
from articles or the model is escaped; only URLs from our own feed records
are ever turned into links. This lets the chunker split safely at line
boundaries.
"""

from __future__ import annotations

import html
from datetime import date
from pathlib import Path

from .config import SUBJECTS
from .sources import Source
from .util import is_valid_http_url, parse_iso, tokens

BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
BN_MONTHS = ["জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন", "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর"]
OPTION_LABELS = ["ক", "খ", "গ", "ঘ"]
FOCUS_LABELS = {"prelims": "প্রিলি", "written": "লিখিত", "both": "প্রিলি + লিখিত"}
WORDS_PER_MINUTE = 150  # conservative for mixed Bengali/English reading on a phone
GK_NOTE = "প্রতিবেদনের বাইরের তথ্য; AI দুইবার আলাদাভাবে মিলিয়ে দেখেছে, তবু মূল সূত্রে যাচাই করুন"
FOOTER = (
    "AI-সহায়িত নোট। 'প্রতিবেদন থেকে' অংশগুলো মূল লেখার উদ্ধৃতির সঙ্গে মিলিয়ে রাখা হয়েছে; "
    "'AI বিশ্লেষণ' ও 'পটভূমি' ব্যাখ্যামূলক; 'প্রাসঙ্গিক সাধারণ জ্ঞান' প্রতিবেদনের বাইরের। "
    "গুরুত্বপূর্ণ তথ্য মূল সূত্রে যাচাই করে নিন।"
)


def _mcq_source(q: dict) -> str:
    """Where an MCQ's answer comes from: the article, or checked background GK."""
    if q.get("basis") == "gk":
        return f"📚 সাধারণ জ্ঞান (প্রতিবেদনের বাইরে) · প্রসঙ্গ: {link(q['source_url'], q['source_name'])}"
    return f"সূত্র: {link(q['source_url'], q['source_name'])}"


def _is_date(value) -> bool:
    try:
        date.fromisoformat(value)
    except (TypeError, ValueError):
        return False
    return True


def esc(text: str) -> str:
    return html.escape(text or "", quote=False)


def link(url: str, label: str) -> str:
    if not is_valid_http_url(url):
        return esc(label)
    return f'<a href="{html.escape(url, quote=True)}">{esc(label)}</a>'


def bn_num(value: int | float | str) -> str:
    return str(value).translate(BN_DIGITS)


def bn_date(d: date) -> str:
    return f"{bn_num(d.day)} {BN_MONTHS[d.month - 1]} {bn_num(d.year)}"


def _story_section(n: int, story: dict, official_refs: list[tuple[str, str]]) -> list[str]:
    note = story["analysis"]
    lines = [f"<b>{bn_num(n)}. {esc(note['headline_bn'])}</b>"]
    cats = " · ".join(SUBJECTS[c] for c in note["categories"] if c in SUBJECTS) or "সাধারণ"
    focus = FOCUS_LABELS.get(note.get("exam_focus"), "প্রিলি")
    lines.append(f"🏷 {esc(cats)} | {focus} | স্কোর {bn_num(int(round(note['score'])))}/১০০")
    published = parse_iso(story.get("published_at"))
    when = bn_date(published.date()) if published and story.get("date_status") == "ok" else "প্রকাশের তারিখ যাচাই করা যায়নি"
    lines.append(f"📰 {esc(', '.join(story['source_names']))} · {when}")
    if note.get("mode") == "excerpt_only":
        lines.append("⚠️ <i>শুধু সংক্ষিপ্তাংশ (excerpt) থেকে তৈরি — পূর্ণ প্রতিবেদন পাওয়া যায়নি।</i>")
    lines.append("")
    if note.get("syllabus_label"):
        lines.append(f"🎯 সিলেবাস: {esc(note['syllabus_label'])}")
    lines.append(f"<b>কী ঘটেছে:</b> {esc(note['what_happened'])}")
    if note.get("whats_new"):
        earlier = (note.get("followup_of") or {}).get("date")
        since = f" <i>(আগের খবর: {bn_date(date.fromisoformat(earlier))})</i>" if _is_date(earlier) else ""
        lines.append(f"🆕 <b>নতুন কী:</b> {esc(note['whats_new'])}{since}")

    prelims = [f for f in note["key_facts"] if f.get("prelims")]
    others = [f for f in note["key_facts"] if not f.get("prelims")]
    if prelims:
        lines.append("<b>📌 প্রিলির জন্য মনে রাখুন</b> <i>(প্রতিবেদন থেকে)</i>")
        lines.extend(f"• {esc(f['fact'])}" for f in prelims)
    if others:
        lines.append("<b>আরও তথ্য</b> <i>(প্রতিবেদন থেকে)</i>")
        lines.extend(f"• {esc(f['fact'])}" for f in others)
    if note.get("attributed_claims"):
        lines.append("<b>উদ্ধৃত দাবি</b> <i>(যাচাইকৃত তথ্য নয়)</i>")
        lines.extend(f"• {esc(c['attributed_to'])}: {esc(c['claim'])}" for c in note["attributed_claims"])
    if note.get("why_it_matters"):
        lines.append(f"<b>বিসিএস-এ কেন গুরুত্বপূর্ণ:</b> {esc(note['why_it_matters'])}")
    if note.get("written_points"):
        lines.append("<b>✍️ লিখিত উত্তরের পয়েন্ট</b> <i>(AI বিশ্লেষণ)</i>")
        lines.extend(f"• {esc(p)}" for p in note["written_points"])
    else:  # notes written before written_points existed
        if note.get("written_angle"):
            lines.append(f"<b>✍️ লিখিত পরীক্ষার দৃষ্টিকোণ</b> <i>(AI বিশ্লেষণ)</i>: {esc(note['written_angle'])}")
        if note.get("causes_effects"):
            lines.append(f"<b>কারণ ও প্রভাব</b> <i>(AI বিশ্লেষণ)</i>: {esc(note['causes_effects'])}")
    if note.get("static_gk"):
        lines.append(f"<b>📚 প্রাসঙ্গিক সাধারণ জ্ঞান</b> <i>({GK_NOTE})</i>")
        lines.extend(f"• {esc(g['statement'])}" for g in note["static_gk"])
    elif note.get("background_context"):
        lines.append(f"<b>পটভূমি</b> <i>(প্রতিবেদনের বাইরের সাধারণ প্রেক্ষাপট)</i>: {esc(note['background_context'])}")
    if note.get("likely_question"):
        lines.append(f"<b>📝 সম্ভাব্য লিখিত প্রশ্ন:</b> {esc(note['likely_question'])}")
    links = " | ".join(link(u, name) for name, u in story["links"])
    lines.append(f"🔗 {links}")
    if official_refs:
        refs = " | ".join(link(u, name) for name, u in official_refs)
        lines.append(f"🏛 দাপ্তরিক সূত্রে যাচাই: {refs}")
    return lines


def official_references(note: dict, sources: list[Source]) -> list[tuple[str, str]]:
    """Official-source homepages for institutions the article itself mentions."""
    mentioned = " ".join(note.get("institutions", [])).lower()
    refs = []
    for src in sources:
        if src.type not in ("official", "news_agency", "international_org"):
            continue
        names = [src.name.split(" (")[0].split(" —")[0], *src.aliases]
        if any(n and n.lower() in mentioned for n in names):
            refs.append((src.name, src.homepage))
    return refs


def estimate_minutes(texts: list[str]) -> int:
    words = sum(len(tokens(t)) for t in texts)
    return max(1, round(words / WORDS_PER_MINUTE))


def render_edition(
    edition_date: date,
    stories: list[dict],
    mcqs: list[dict],
    partial_reasons: list[str],
    unprocessed: int,
    sources: list[Source],
    mcq_target: int,
    notices: list[str] | None = None,
    extra: bool = False,
    since_last: bool = False,
) -> list[str]:
    """Return logical sections (each a string of newline-joined lines).

    ``extra`` marks an additional (manually triggered) edition, which only
    carries stories not sent in an earlier edition. ``since_last`` words an
    empty edition as "nothing new since the last briefing" (an earlier
    edition already went out today).
    """
    story_sections = [
        "\n".join(_story_section(i, s, official_references(s["analysis"], sources))) for i, s in enumerate(stories, start=1)
    ]
    mcq_sections: list[str] = []
    if mcqs:
        q_lines = [f"<b>🧠 আজকের অনুশীলনী ({bn_num(len(mcqs))}টি MCQ)</b>", "<i>উত্তর দেখার আগে নিজে চেষ্টা করুন।</i>", ""]
        for i, q in enumerate(mcqs, start=1):
            q_lines.append(f"<b>প্রশ্ন {bn_num(i)}.</b> {esc(q['question'])}")
            q_lines.extend(f"   ({OPTION_LABELS[j]}) {esc(opt)}" for j, opt in enumerate(q["options"]))
            q_lines.append("")
        mcq_sections.append("\n".join(q_lines).rstrip())
        a_lines = ["━━━━━━━━━━━━", "<b>✅ উত্তর ও ব্যাখ্যা</b> <i>(ট্যাপ করে দেখুন)</i>", ""]
        for i, q in enumerate(mcqs, start=1):
            ans = f"({OPTION_LABELS[q['answer_index']]}) {q['options'][q['answer_index']]}"
            a_lines.append(f"<b>{bn_num(i)}.</b> <tg-spoiler>{esc(ans)}</tg-spoiler>")
            if q.get("explanation"):
                a_lines.append(f"   <tg-spoiler>{esc(q['explanation'])}</tg-spoiler>")
            subject = SUBJECTS.get(q["subject"], "")
            a_lines.append(f"   🏷 {esc(subject)} · {_mcq_source(q)}")
        mcq_sections.append("\n".join(a_lines))

    minutes = estimate_minutes(story_sections + mcq_sections)
    title = "🔄 <b>বিসিএস ব্রিফিং: অতিরিক্ত আপডেট</b>" if extra else "📰 <b>বিসিএস দৈনিক ব্রিফিং</b>"
    header = [f"{title} — {bn_date(edition_date)}"]
    if stories:
        header.append(f"📚 {bn_num(len(stories))}টি খবর · ⏱ পড়তে প্রায় {bn_num(minutes)} মিনিট")
        header.append("")
        header.append("<b>আজকের শিরোনাম</b>")
        header.extend(f"{bn_num(i)}. {esc(s['analysis']['headline_bn'])}" for i, s in enumerate(stories, start=1))
    if partial_reasons:
        header.append("⚠️ <b>আংশিক সংস্করণ</b>")
        header.extend(f"   • {esc(r)}" for r in partial_reasons)
        if unprocessed:
            header.append(f"   • বিশ্লেষণ বাকি আছে এমন প্রার্থী খবর: {bn_num(unprocessed)}টি (পরে আবার চেষ্টা করা হবে)")
    if not stories:
        header.append("")
        if extra or since_last:
            header.append("সর্বশেষ ব্রিফিংয়ের পর বিসিএস-এর জন্য নতুন কোনো প্রাসঙ্গিক ও যাচাইযোগ্য খবর পাওয়া যায়নি। আগে পাঠানো খবর আবার পাঠানো হয়নি।")
        else:
            header.append("আজ বিসিএস-এর জন্য যথেষ্ট প্রাসঙ্গিক ও যাচাইযোগ্য কোনো খবর পাওয়া যায়নি। অপ্রাসঙ্গিক খবর দিয়ে ব্রিফিং পূরণ করা হয়নি।")
    elif len(mcqs) < mcq_target:
        header.append(f"ℹ️ যাচাইকৃত তথ্য থেকে {bn_num(len(mcqs))}টি নির্ভরযোগ্য MCQ তৈরি করা গেছে (লক্ষ্য {bn_num(mcq_target)}টি)।")

    footer = f"<i>{esc(FOOTER)}</i>"
    sections = ["\n".join(header)] + story_sections + mcq_sections
    tail = [footer] if stories else []
    tail.extend(f"⚙️ <i>{esc(n)}</i>" for n in notices or [])
    if tail:
        sections.append("\n".join(tail))
    return sections


def render_failure_notice(edition_date: date, reason: str, unprocessed: int) -> str:
    lines = [
        f"⚠️ <b>বিসিএস ব্রিফিং তৈরি করা যায়নি</b> — {bn_date(edition_date)}",
        f"কারণ: {esc(reason)}",
    ]
    if unprocessed:
        lines.append(f"অপেক্ষমাণ প্রার্থী খবর: {bn_num(unprocessed)}টি — পরের রানে নিয়ন্ত্রিতভাবে আবার চেষ্টা করা হবে।")
    lines.append("কোনো খবর বা বিশ্লেষণ বানিয়ে পাঠানো হয়নি।")
    return "\n".join(lines)


# --------------------------------------------------------------------- PDF

LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "logo.png"

PDF_CSS = """
@page { size: A4; margin: 14mm 13mm 16mm 13mm; }
* { box-sizing: border-box; }
body { margin: 0; color: #1d2b26; font-size: 10.5pt; line-height: 1.55;
  font-family: "Noto Sans Bengali", "Noto Serif Bengali", "Hind Siliguri", "Lohit Bengali", "Kalpurush", "SolaimanLipi", "Noto Sans", sans-serif; }
a { color: #00694a; text-decoration: none; }
.masthead { display: flex; align-items: center; gap: 12px; background: #00563f; color: #fff;
  border-radius: 10px; padding: 12px 16px; border-bottom: 4px solid #f42a41; }
.masthead img { width: 52px; height: 52px; border-radius: 50%; }
.masthead h1 { margin: 0; font-size: 19pt; line-height: 1.25; }
.masthead .meta { font-size: 10pt; opacity: .9; }
.notice { margin: 10px 0 0; padding: 8px 12px; border-radius: 8px; background: #fff4f4; border-left: 4px solid #f42a41; font-size: 9.5pt; }
.notice.info { background: #eef6f2; border-left-color: #00875f; }
.story { margin: 14px 0 0; padding: 10px 14px 12px; border: 1px solid #d5e2dc; border-radius: 10px; break-inside: avoid; page-break-inside: avoid; }
.story h2 { margin: 0 0 4px; font-size: 13pt; line-height: 1.35; color: #00563f; }
.tags { font-size: 8.8pt; color: #4d6159; margin-bottom: 6px; }
.chip { display: inline-block; padding: 0 7px; margin-right: 4px; border-radius: 9px; background: #e3efe9; color: #00563f; }
.chip.score { background: #fde8ea; color: #b3172a; }
.excerpt { font-size: 9pt; color: #9a3412; margin: 2px 0 6px; }
h3 { margin: 8px 0 2px; font-size: 10.5pt; color: #00563f; }
h3 small { font-weight: normal; color: #6b7c75; font-size: 8.5pt; }
p { margin: 2px 0; }
ul { margin: 2px 0 2px 18px; padding: 0; }
li { margin: 1px 0; }
.remember { background: #fff8e6; border-left: 4px solid #f2b705; border-radius: 6px; padding: 2px 10px 4px; margin: 6px 0; }
.gk { background: #eef3fb; border-left: 4px solid #3b6fb6; border-radius: 6px; padding: 2px 10px 4px; margin: 6px 0; }
.revision { break-before: page; page-break-before: always; }
.revision h3 { margin-top: 10px; }
.sources { margin-top: 6px; font-size: 8.8pt; color: #4d6159; }
.section-title { margin: 18px 0 6px; font-size: 14pt; color: #00563f; border-bottom: 2px solid #d5e2dc; padding-bottom: 3px; }
.mcq { margin: 8px 0; break-inside: avoid; }
.mcq .opts { display: grid; grid-template-columns: 1fr 1fr; gap: 2px 18px; margin: 3px 0 0 14px; }
.answers { break-before: page; page-break-before: always; }
.answer { margin: 6px 0; break-inside: avoid; }
.footer { margin-top: 18px; font-size: 8.5pt; color: #6b7c75; border-top: 1px solid #d5e2dc; padding-top: 6px; }
"""


def _logo_data_uri() -> str:
    try:
        import base64

        return "data:image/png;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode()
    except OSError:
        return ""


def _pdf_story(n: int, story: dict, sources: list[Source]) -> str:
    note = story["analysis"]
    cats = "".join(f'<span class="chip">{esc(SUBJECTS[c])}</span>' for c in note["categories"] if c in SUBJECTS)
    focus = FOCUS_LABELS.get(note.get("exam_focus"), "প্রিলি")
    published = parse_iso(story.get("published_at"))
    when = bn_date(published.date()) if published and story.get("date_status") == "ok" else "প্রকাশের তারিখ যাচাই করা যায়নি"
    parts = [
        '<section class="story">',
        f"<h2>{bn_num(n)}. {esc(note['headline_bn'])}</h2>",
        f'<div class="tags">{cats}<span class="chip">{focus}</span><span class="chip score">স্কোর {bn_num(int(round(note["score"])))}/১০০</span>'
        f" &nbsp;📰 {esc(', '.join(story['source_names']))} · {when}</div>",
    ]
    if note.get("mode") == "excerpt_only":
        parts.append('<div class="excerpt">⚠️ শুধু সংক্ষিপ্তাংশ (excerpt) থেকে তৈরি — পূর্ণ প্রতিবেদন পাওয়া যায়নি।</div>')
    if note.get("syllabus_label"):
        parts.append(f'<div class="tags">🎯 সিলেবাস: {esc(note["syllabus_label"])}</div>')
    parts.append(f"<h3>কী ঘটেছে</h3><p>{esc(note['what_happened'])}</p>")
    if note.get("whats_new"):
        earlier = (note.get("followup_of") or {}).get("date")
        since = f" <small>(আগের খবর: {bn_date(date.fromisoformat(earlier))})</small>" if _is_date(earlier) else ""
        parts.append(f"<h3>🆕 নতুন কী{since}</h3><p>{esc(note['whats_new'])}</p>")
    prelims = [f for f in note["key_facts"] if f.get("prelims")]
    others = [f for f in note["key_facts"] if not f.get("prelims")]
    if prelims:
        parts.append('<div class="remember"><h3>📌 প্রিলির জন্য মনে রাখুন <small>(প্রতিবেদন থেকে)</small></h3><ul>'
                     + "".join(f"<li>{esc(f['fact'])}</li>" for f in prelims) + "</ul></div>")
    if others:
        parts.append("<h3>আরও তথ্য <small>(প্রতিবেদন থেকে)</small></h3><ul>" + "".join(f"<li>{esc(f['fact'])}</li>" for f in others) + "</ul>")
    if note.get("attributed_claims"):
        parts.append("<h3>উদ্ধৃত দাবি <small>(যাচাইকৃত তথ্য নয়)</small></h3><ul>"
                     + "".join(f"<li>{esc(c['attributed_to'])}: {esc(c['claim'])}</li>" for c in note["attributed_claims"]) + "</ul>")
    if note.get("why_it_matters"):
        parts.append(f"<h3>বিসিএস-এ কেন গুরুত্বপূর্ণ</h3><p>{esc(note['why_it_matters'])}</p>")
    if note.get("written_points"):
        parts.append("<h3>✍️ লিখিত উত্তরের পয়েন্ট <small>(AI বিশ্লেষণ)</small></h3><ul>"
                     + "".join(f"<li>{esc(p)}</li>" for p in note["written_points"]) + "</ul>")
    else:  # notes written before written_points existed
        if note.get("written_angle"):
            parts.append(f"<h3>✍️ লিখিত পরীক্ষার দৃষ্টিকোণ <small>(AI বিশ্লেষণ)</small></h3><p>{esc(note['written_angle'])}</p>")
        if note.get("causes_effects"):
            parts.append(f"<h3>কারণ ও প্রভাব <small>(AI বিশ্লেষণ)</small></h3><p>{esc(note['causes_effects'])}</p>")
    if note.get("static_gk"):
        parts.append(f'<div class="gk"><h3>📚 প্রাসঙ্গিক সাধারণ জ্ঞান <small>({GK_NOTE})</small></h3><ul>'
                     + "".join(f"<li>{esc(g['statement'])}</li>" for g in note["static_gk"]) + "</ul></div>")
    elif note.get("background_context"):
        parts.append(f"<h3>পটভূমি <small>(প্রতিবেদনের বাইরের সাধারণ প্রেক্ষাপট)</small></h3><p>{esc(note['background_context'])}</p>")
    if note.get("likely_question"):
        parts.append(f"<h3>📝 সম্ভাব্য লিখিত প্রশ্ন</h3><p>{esc(note['likely_question'])}</p>")
    links = " | ".join(link(u, name) for name, u in story["links"])
    refs = official_references(note, sources)
    official = (" &nbsp;🏛 দাপ্তরিক সূত্র: " + " | ".join(link(u, name) for name, u in refs)) if refs else ""
    parts.append(f'<div class="sources">🔗 {links}{official}</div></section>')
    return "\n".join(parts)


def _pdf_revision(stories: list[dict]) -> str:
    """Last page: every must-remember fact and verified GK item in one place."""
    blocks = []
    for i, story in enumerate(stories, start=1):
        note = story["analysis"]
        facts = [f["fact"] for f in note["key_facts"] if f.get("prelims")]
        gk = [g["statement"] for g in note.get("static_gk") or []]
        if not facts and not gk:
            continue
        items = "".join(f"<li>{esc(x)}</li>" for x in facts) + "".join(f"<li>📚 {esc(x)}</li>" for x in gk)
        blocks.append(f"<h3>{bn_num(i)}. {esc(note['headline_bn'])}</h3><ul>{items}</ul>")
    if not blocks:
        return ""
    return ('<div class="revision"><h2 class="section-title">📋 এক নজরে রিভিশন</h2>'
            '<div class="notice info">প্রিলির জন্য মনে রাখার তথ্য (প্রতিবেদন থেকে) ও 📚 প্রাসঙ্গিক সাধারণ জ্ঞান (প্রতিবেদনের বাইরে)।</div>'
            + "".join(blocks) + "</div>")


def render_pdf_html(
    edition_date: date,
    stories: list[dict],
    mcqs: list[dict],
    partial_reasons: list[str],
    sources: list[Source],
    extra: bool = False,
) -> str:
    """A self-contained, printable HTML page for the PDF version.

    All article/model text is escaped and the page allows no scripts or
    external loads (Content-Security-Policy), so content can't run code.
    """
    title = "বিসিএস ব্রিফিং: অতিরিক্ত আপডেট" if extra else "বিসিএস দৈনিক ব্রিফিং"
    logo = _logo_data_uri()
    body = [
        '<header class="masthead">',
        f'<img src="{logo}" alt="">' if logo else "",
        f'<div><h1>{title}</h1><div class="meta">{bn_date(edition_date)} · {bn_num(len(stories))}টি খবর'
        + (f" · {bn_num(len(mcqs))}টি MCQ" if mcqs else "") + "</div></div></header>",
    ]
    if partial_reasons:
        body.append('<div class="notice"><b>⚠️ আংশিক সংস্করণ:</b> ' + " ".join(esc(r) for r in partial_reasons) + "</div>")
    body.extend(_pdf_story(i, s, sources) for i, s in enumerate(stories, start=1))
    if mcqs:
        body.append('<h2 class="section-title">🧠 অনুশীলনী (MCQ)</h2>'
                    '<div class="notice info">উত্তর শেষ পৃষ্ঠায় — আগে নিজে চেষ্টা করুন।</div>')
        for i, q in enumerate(mcqs, start=1):
            opts = "".join(f"<div>({OPTION_LABELS[j]}) {esc(o)}</div>" for j, o in enumerate(q["options"]))
            body.append(f'<div class="mcq"><b>প্রশ্ন {bn_num(i)}.</b> {esc(q["question"])}<div class="opts">{opts}</div></div>')
        body.append('<div class="answers"><h2 class="section-title">✅ উত্তর ও ব্যাখ্যা</h2>')
        for i, q in enumerate(mcqs, start=1):
            ans = f"({OPTION_LABELS[q['answer_index']]}) {q['options'][q['answer_index']]}"
            expl = f"<br>{esc(q['explanation'])}" if q.get("explanation") else ""
            subject = SUBJECTS.get(q["subject"], "")
            body.append(f'<div class="answer"><b>{bn_num(i)}. {esc(ans)}</b>{expl}'
                        f'<div class="sources">🏷 {esc(subject)} · {_mcq_source(q)}</div></div>')
        body.append("</div>")
    body.append(_pdf_revision(stories))
    body.append(f'<div class="footer">{esc(FOOTER)}</div>')
    csp = "default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:"
    return (
        '<!doctype html><html lang="bn"><head><meta charset="utf-8">'
        f'<meta http-equiv="Content-Security-Policy" content="{csp}">'
        f"<title>{title} — {bn_date(edition_date)}</title><style>{PDF_CSS}</style></head><body>"
        + "\n".join(body) + "</body></html>"
    )
