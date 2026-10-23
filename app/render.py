"""Render an edition as Telegram-safe HTML sections.

Invariant: every rendered *line* is self-contained, balanced HTML. All text
from articles or the model is escaped; only URLs from our own feed records
are ever turned into links. This lets the chunker split safely at line
boundaries.
"""

from __future__ import annotations

import html
from datetime import date

from .config import SUBJECTS
from .sources import Source
from .util import is_valid_http_url, parse_iso, tokens

BN_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
BN_MONTHS = ["জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন", "জুলাই", "আগস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর"]
OPTION_LABELS = ["ক", "খ", "গ", "ঘ"]
FOCUS_LABELS = {"prelims": "প্রিলি", "written": "লিখিত", "both": "প্রিলি + লিখিত"}
WORDS_PER_MINUTE = 150  # conservative for mixed Bengali/English reading on a phone


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
    lines.append(f"<b>কী ঘটেছে:</b> {esc(note['what_happened'])}")
    if note.get("why_it_matters"):
        lines.append(f"<b>বিসিএস-এ কেন গুরুত্বপূর্ণ:</b> {esc(note['why_it_matters'])}")

    prelims = [f for f in note["key_facts"] if f.get("prelims")]
    others = [f for f in note["key_facts"] if not f.get("prelims")]
    if others:
        lines.append("<b>গুরুত্বপূর্ণ তথ্য (প্রতিবেদন থেকে):</b>")
        lines.extend(f"• {esc(f['fact'])}" for f in others)
    if prelims:
        lines.append("<b>📌 প্রিলির জন্য মনে রাখুন (প্রতিবেদন থেকে):</b>")
        lines.extend(f"• {esc(f['fact'])}" for f in prelims)
    if note.get("attributed_claims"):
        lines.append("<b>উদ্ধৃত দাবি (যাচাইকৃত তথ্য নয়):</b>")
        lines.extend(f"• {esc(c['attributed_to'])}: {esc(c['claim'])}" for c in note["attributed_claims"])
    if note.get("written_angle"):
        lines.append(f"<b>✍️ লিখিত পরীক্ষার দৃষ্টিকোণ</b> <i>(AI বিশ্লেষণ)</i>: {esc(note['written_angle'])}")
    if note.get("causes_effects"):
        lines.append(f"<b>কারণ ও প্রভাব</b> <i>(AI বিশ্লেষণ)</i>: {esc(note['causes_effects'])}")
    if note.get("background_context"):
        lines.append(f"<b>পটভূমি</b> <i>(প্রতিবেদনের বাইরের সাধারণ প্রেক্ষাপট)</i>: {esc(note['background_context'])}")
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
) -> list[str]:
    """Return logical sections (each a string of newline-joined lines)."""
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
            a_lines.append(f"   🏷 {esc(subject)} · সূত্র: {link(q['source_url'], q['source_name'])}")
        mcq_sections.append("\n".join(a_lines))

    minutes = estimate_minutes(story_sections + mcq_sections)
    header = [f"📰 <b>বিসিএস দৈনিক ব্রিফিং</b> — {bn_date(edition_date)}"]
    if stories:
        header.append(f"📚 {bn_num(len(stories))}টি খবর · ⏱ পড়তে প্রায় {bn_num(minutes)} মিনিট")
    if partial_reasons:
        header.append("⚠️ <b>আংশিক সংস্করণ</b>")
        header.extend(f"   • {esc(r)}" for r in partial_reasons)
        if unprocessed:
            header.append(f"   • বিশ্লেষণ বাকি আছে এমন প্রার্থী খবর: {bn_num(unprocessed)}টি (পরে আবার চেষ্টা করা হবে)")
    if not stories:
        header.append("")
        header.append("আজ বিসিএস-এর জন্য যথেষ্ট প্রাসঙ্গিক ও যাচাইযোগ্য কোনো খবর পাওয়া যায়নি। অপ্রাসঙ্গিক খবর দিয়ে ব্রিফিং পূরণ করা হয়নি।")
    elif len(mcqs) < mcq_target:
        header.append(f"ℹ️ যাচাইকৃত তথ্য থেকে {bn_num(len(mcqs))}টি নির্ভরযোগ্য MCQ তৈরি করা গেছে (লক্ষ্য {bn_num(mcq_target)}টি)।")

    footer = (
        "<i>AI-সহায়িত নোট। 'প্রতিবেদন থেকে' অংশগুলো মূল লেখার উদ্ধৃতির সঙ্গে মিলিয়ে রাখা হয়েছে; "
        "'AI বিশ্লেষণ' ও 'পটভূমি' অংশ ব্যাখ্যামূলক। গুরুত্বপূর্ণ তথ্য মূল সূত্রে যাচাই করে নিন।</i>"
    )
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
