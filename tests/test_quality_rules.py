"""Quality rules: repeats of sent stories, note structure, static GK checks, MCQ options."""

import json

import pytest

from app.analysis import validate_analysis
from app.deduplicator import distinctive_numbers, same_story
from app.enrichment import answers_agree, enrich_static_gk, validate_gk, verify_gk
from app.llm import build_client
from app.llm.mock import ScriptedProvider
from app.questions import build_fact_pool, option_problem, validate_mcqs
from app.screening import build_recent, validate_screening
from app.syllabus import load_syllabus

from .conftest import NOW, FakeTelegram, no_sleep

WEIGHTS = {"syllabus_relevance": 0.30, "significance": 0.25, "factual_density": 0.20, "written_potential": 0.15, "novelty": 0.10}
SOURCE = (
    "Bangladesh Bank raised the policy rate to 10 percent on Monday, citing inflation of 9.2 percent in September. "
    "The governor also said the central bank would publish a new monetary policy statement in January."
)


@pytest.fixture
def profile(settings):
    return load_syllabus(settings.config_dir)


# ------------------------------------------------------------------ repeats
def screen_item(cid, **kw):
    item = {"id": cid, "decision": "select", "story_type": "economic_data", "topic": "bd_budget_fiscal", "priority": 8,
            "categories": ["economy"], "exam_hook": "h", "rationale": "r"}
    item.update(kw)
    return item


def test_repeat_of_a_sent_story_is_rejected_unless_something_is_new(profile):
    shown, rmap = build_recent([{"event_id": "ev_old", "date": "2026-10-29", "headline": "এনবিআরের রাজস্ব ঘাটতি", "title": "NBR shortfall"}])
    assert shown == [{"id": "r1", "sent": "2026-10-29", "headline": "এনবিআরের রাজস্ব ঘাটতি", "title": "NBR shortfall"}]
    raw = {"results": [
        screen_item("c1", repeat_of="r1", new_development=""),                                  # same story again
        screen_item("c2", repeat_of="r1", new_development="সরকার রাজস্ব লক্ষ্য কমানোর সিদ্ধান্ত নিয়েছে"),  # real follow-up
        screen_item("c3", repeat_of="r9", new_development=""),                                  # unknown id: ignored
    ]}
    out = validate_screening(raw, {"c1": "e1", "c2": "e2", "c3": "e3"}, profile, recent_map=rmap)
    assert out.decisions["e1"]["decision"] == "reject" and "repeat of a story sent on 2026-10-29" in out.decisions["e1"]["forced_reject"]
    assert out.decisions["e2"]["decision"] == "select"
    assert out.decisions["e2"]["followup_of"] == {"event_id": "ev_old", "date": "2026-10-29", "headline": "এনবিআরের রাজস্ব ঘাটতি"}
    assert out.decisions["e3"]["decision"] == "select" and "followup_of" not in out.decisions["e3"]


def test_distinctive_numbers_and_same_story():
    a = distinctive_numbers(["প্রথম প্রান্তিকে ঘাটতি ৪১,৮৭৯ কোটি টাকা", "আদায় ৮২,৪৬৭ কোটি টাকা, ২০২৬ সালে ৩ মাসে", "হার 3.89 শতাংশ"])
    assert a == {"41879", "82467", "3.89"}  # not years, days or small counts
    b = distinctive_numbers(["রাজস্ব ঘাটতি 41,879 কোটি", "আদায় 82,467 কোটি"])
    assert same_story(a, "এনবিআরের রাজস্ব ঘাটতি", b, "তিন মাসে রাজস্ব ঘাটতি")
    assert not same_story(a, "এনবিআরের রাজস্ব ঘাটতি", {"41879"}, "সম্পূর্ণ ভিন্ন শিরোনাম এখানে")
    assert not same_story(set(), "x", set(), "x")


def test_pipeline_skips_a_story_already_sent_on_an_earlier_day(make_pipeline, settings):
    tg = FakeTelegram()
    make_pipeline(telegram=tg).run(now=NOW)
    state = json.loads(settings.state_path.read_text())
    sent = [e for e in state["events"].values() if e["status"] == "included"]
    # The next day the same report arrives again as a new event (other outlet, other wording).
    clone = json.loads(json.dumps(sent[0]))
    clone.update(event_id="ev_clone", status="analysed", included_in=None)
    clone["analysis"]["headline_bn"] = sent[0]["analysis"]["headline_bn"] + " (অন্য সূত্রে)"
    state["events"]["ev_clone"] = clone
    settings.state_path.write_text(json.dumps(state))
    pipe = make_pipeline(telegram=FakeTelegram())
    pipe.state = pipe.store.load()
    from datetime import timedelta

    recent = pipe._recent_sent(NOW + timedelta(days=1))
    assert {r["event_id"] for r in recent} == {e["event_id"] for e in sent}
    assert pipe._repeat_of_sent(pipe.state["events"]["ev_clone"], recent)["event_id"] == sent[0]["event_id"]
    # An accepted follow-up is never blocked by the safety net.
    pipe.state["events"]["ev_clone"]["screening"]["followup_of"] = {"event_id": sent[0]["event_id"]}
    assert pipe._repeat_of_sent(pipe.state["events"]["ev_clone"], recent) is None


def test_screening_prompt_lists_recent_stories(make_pipeline, settings):
    from datetime import timedelta

    from .test_pipeline import Programmable

    make_pipeline(telegram=FakeTelegram()).run(now=NOW)
    prov = ScriptedProvider()
    captured = []

    def screening(request):
        captured.append(request)
        return {"results": []}

    prov.queue(screening, screening)
    pipe = make_pipeline(provider=prov, telegram=FakeTelegram())
    pipe.state = pipe.store.load()
    for ev in pipe.state["events"].values():
        if ev["status"] != "included":
            ev["status"] = "new"
    pipe._screen(NOW + timedelta(hours=1))
    assert captured and "Recently sent" in captured[0].prompt and captured[0].payload["recent"]
    assert Programmable  # imported for parity with other suites


# ------------------------------------------------------------------ notes
def note_raw(**kw):
    raw = {
        "relevant": True, "evidence_sufficient": True, "headline_bn": "নীতি সুদহার বেড়ে ১০ শতাংশ",
        "syllabus_topic": "bd_banking_monetary", "categories": ["economy"], "likely_question": "মুদ্রানীতির লক্ষ্য আলোচনা করুন।",
        "what_happened": "বাংলাদেশ ব্যাংক নীতি সুদহার 10 percent করেছে।", "why_it_matters": "মুদ্রানীতি ও মূল্যস্ফীতি।",
        "key_facts": [
            {"fact": "নীতি সুদহার 10 percent", "evidence": "raised the policy rate to 10 percent", "kind": "number", "prelims": True},
            {"fact": "সেপ্টেম্বরে মূল্যস্ফীতি 9.2 percent", "evidence": "citing inflation of 9.2 percent in September", "kind": "number", "prelims": True},
            {"fact": "জানুয়ারিতে নতুন মুদ্রানীতি বিবৃতি", "evidence": "publish a new monetary policy statement in January", "kind": "date", "prelims": True},
            {"fact": "বাংলাদেশ ব্যাংক সিদ্ধান্ত নিয়েছে সোমবার", "evidence": "Bangladesh Bank raised the policy rate", "kind": "weird", "prelims": True},
        ],
        "attributed_claims": [], "written_points": ["মূল্যস্ফীতি নিয়ন্ত্রণে সুদহার বাড়ানো হয়।", "বিনিয়োগ কমতে পারে 50 percent।", "বিনিয়োগে প্রভাব পড়ে।"],
        "whats_new": "নতুন তথ্য।", "background_context": "", "institutions": [],
        "scores": {"syllabus_relevance": 9, "significance": 8, "factual_density": 8, "written_potential": 8, "novelty": 8},
    }
    raw.update(kw)
    return raw


def test_note_caps_prelims_derives_focus_and_grounds_written_points(profile):
    note = validate_analysis(note_raw(), SOURCE, "h", None, WEIGHTS, profile, 7)
    assert [f["prelims"] for f in note["key_facts"]] == [True, True, True, False]  # at most three
    assert note["key_facts"][3]["kind"] == "other"
    assert note["written_points"] == ["মূল্যস্ফীতি নিয়ন্ত্রণে সুদহার বাড়ানো হয়।", "বিনিয়োগে প্রভাব পড়ে।"]  # unsupported 50 dropped
    assert note["exam_focus"] == "both"
    assert note["whats_new"] == ""  # only for follow-ups
    low_written = note_raw(scores={"syllabus_relevance": 9, "significance": 8, "factual_density": 8, "written_potential": 5, "novelty": 8})
    assert validate_analysis(low_written, SOURCE, "h", None, WEIGHTS, profile, 7)["exam_focus"] == "prelims"
    follow = validate_analysis(note_raw(), SOURCE, "h", None, WEIGHTS, profile, 7, followup={"date": "2026-10-29"})
    assert follow["whats_new"] == "নতুন তথ্য।"


# ------------------------------------------------------------------ static GK
def story(eid="e1", headline="তত্ত্বাবধায়ক সরকারের অধীনে নির্বাচন", facts=("রাজস্ব ঘাটতি ৪১,৮৭৯ কোটি টাকা",)):
    return {"event_id": eid, "primary_url": "https://example.com/a", "primary_source": "X",
            "analysis": {"headline_bn": headline, "what_happened": "সারাংশ", "categories": ["constitution_governance"],
                         "key_facts": [{"fact": f, "evidence": f, "prelims": True} for f in facts]}}


def gk(**kw):
    item = {"story": "s1", "statement": "তত্ত্বাবধায়ক সরকার ব্যবস্থা ১৯৯৬ সালে ত্রয়োদশ সংশোধনীর মাধ্যমে যুক্ত হয়।",
            "question": "কোন সালে তত্ত্বাবধায়ক সরকার ব্যবস্থা সংবিধানে যুক্ত হয়?", "answer": "১৯৯৬", "answer_en": "1996", "kind": "year"}
    item.update(kw)
    return item


def test_gk_shape_rules():
    smap = {"s1": story()}
    good, _ = validate_gk({"items": [gk()]}, smap)
    assert len(good) == 1
    bad = [
        gk(story="s9"),                                                  # unknown story
        gk(statement="তত্ত্বাবধায়ক সরকার সম্ভবত ১৯৯৬ সালে যুক্ত হয়।"),          # hedged
        gk(answer="২০১১", answer_en="2011"),                             # statement doesn't state the answer
        gk(question="১৯৯৬ সালে কী যুক্ত হয়?"),                             # question leaks the answer
        gk(statement="রাজস্ব ঘাটতি ৪১,৮৭৯ কোটি টাকা, ১৯৯৬ সালের পর সর্বোচ্চ।"),  # restates the article's figures
        gk(answer_en=""),                                                 # incomplete
    ]
    kept, reasons = validate_gk({"items": bad}, smap)
    assert kept == [] and len(reasons) == len(bad)
    many, _ = validate_gk({"items": [gk(statement=f"বিবৃতি {i}: ১৯৯৬ সালে যুক্ত হয়।") for i in range(5)]}, smap)
    assert len(many) == 3  # per-story cap


@pytest.mark.parametrize("claimed_en, claimed_bn, got, agree", [
    ("1996", "১৯৯৬", "1996", True),
    ("13th", "ত্রয়োদশ", "13th Amendment", True),
    ("1996", "১৯৯৬", "1991", False),
    ("1950", "১৯৫০", "Founded 1950 in Geneva", True),
    ("1950", "১৯৫০", "1950 or 1951", False),  # not one clear answer
    ("Geneva", "জেনেভা", "Geneva, Switzerland", True),
    ("Geneva", "জেনেভা", "New York", False),
    ("Rosatom", "রোসাটম", "rosatom", True),
    ("International Criminal Court", "আইসিসি", "International Court of Justice", False),
    ("World Health Organization", "ডব্লিউএইচও", "World Meteorological Organization", False),
    ("United Nations", "জাতিসংঘ", "United Nations Development Programme", False),
    ("World Meteorological Organization", "ডব্লিউএমও", "world meteorological organization", True),
])
def test_blind_answers_must_agree(claimed_en, claimed_bn, got, agree):
    assert answers_agree(claimed_en, claimed_bn, got) is agree


def test_verify_keeps_only_sure_agreeing_items():
    items = [gk(), gk(answer_en="Geneva", answer="জেনেভা"), gk(answer_en="2011", answer="২০১১")]
    raw = {"answers": [{"id": "g1", "answer_en": "1996", "sure": True}, {"id": "g2", "answer_en": "Geneva", "sure": False},
                       {"id": "g3", "answer_en": "2009", "sure": True}]}
    kept, reasons = verify_gk(raw, items)
    assert kept == [items[0]] and len(reasons) == 2


def test_enrich_runs_two_calls_and_maps_items_to_events(settings):
    prov = ScriptedProvider([{"items": [gk()]}, {"answers": [{"id": "g1", "answer_en": "1996", "sure": True}]}])
    out = enrich_static_gk(build_client(settings, prov, sleep=no_sleep), [story()])
    assert [r.task for r in prov.requests] == ["static_gk", "static_gk_check"]
    assert "১৯৯৬" not in prov.requests[1].prompt  # the checker never sees the claimed answer
    assert out.items == {"e1": [{k: gk()[k] for k in ("statement", "question", "answer", "answer_en", "kind")}]}


# ------------------------------------------------------------------ MCQs
FACTS = [
    {"fact_id": "f1", "story": "a", "fact": "নীতি সুদহার ১০ শতাংশ", "evidence": "policy rate to 10 percent", "basis": "article",
     "subject": "economy", "event_id": "e1", "url": "https://example.com/1", "source_name": "X"},
    {"fact_id": "f2", "story": "b", "fact": "তত্ত্বাবধায়ক সরকার ১৯৯৬ সালে যুক্ত হয়", "evidence": "তত্ত্বাবধায়ক সরকার ১৯৯৬ সালে যুক্ত হয়",
     "basis": "gk", "subject": "constitution_governance", "event_id": "e2", "url": "https://example.com/2", "source_name": "Y"},
    {"fact_id": "f3", "story": "b", "fact": "ত্রয়োদশ সংশোধনী", "evidence": "ত্রয়োদশ সংশোধনী", "basis": "gk",
     "subject": "constitution_governance", "event_id": "e2", "url": "https://example.com/2", "source_name": "Y"},
    {"fact_id": "f4", "story": "c", "fact": "মেট্রোরেল লাইন ১", "evidence": "Metro Line 1", "basis": "gk",
     "subject": "bangladesh_affairs", "event_id": "e3", "url": "https://example.com/3", "source_name": "Z"},
]


def mcq(fact_id="f1", options=("১০ শতাংশ", "৮ শতাংশ", "৯ শতাংশ", "১২ শতাংশ"), idx=0, question="বাংলাদেশ ব্যাংকের নতুন নীতি সুদহার কত?"):
    return {"question": question, "options": list(options), "answer_index": idx, "explanation": "e", "subject": "economy", "fact_id": fact_id}


@pytest.mark.parametrize("options, idx, problem", [
    (["১০ শতাংশ", "৮ শতাংশ", "৯ শতাংশ", "১২ শতাংশ"], 0, None),
    (["১০ শতাংশ", "৫৮,৫০০ কোটি টাকা", "৯ শতাংশ", "১২ শতাংশ"], 0, "options mix units"),
    (["১৯৯৬", "১৯৯১", "জাতিসংঘ", "২০১১"], 0, "numeric answer but non-numeric options"),
    (["জেনেভা", "নিউইয়র্ক", "প্যারিস", "১৯৫০"], 0, "non-numeric answer but a bare number among the options"),
    (["জেনেভা", "নিউইয়র্ক", "প্যারিস", "ওপরের সবগুলো"], 0, "'all/none of the above' option"),
    (["সুইজারল্যান্ডের জেনেভা শহরে অবস্থিত সদর দপ্তর", "রোম", "ভিয়েনা", "বার্ন"], 0, "correct option much longer than the others"),
    (["১৩তম", "১২তম", "১৪তম", "১৫তম"], 0, None),
])
def test_option_rules(options, idx, problem):
    assert option_problem(options, idx) == problem


def test_mcq_selection_rules():
    raw = {"questions": [
        mcq(question="প্রতিবেদন অনুযায়ী নীতি সুদহার কত?"),                                    # leans on the article
        mcq(question="নীতি সুদহার ১০ শতাংশ করা হয়েছে, নতুন হার কোনটি?"),                      # gives the answer away
        mcq(question="বাংলাদেশ ব্যাংক সম্প্রতি নীতি সুদহার বাড়িয়ে কত শতাংশ করেছে?"),
        mcq("f2", ("১৯৯৬", "১৯৯১", "২০০১", "২০১১"), 0, "তত্ত্বাবধায়ক সরকার ব্যবস্থা কোন সালে সংবিধানে যুক্ত হয়?"),
        mcq("f3", ("ত্রয়োদশ", "দ্বাদশ", "চতুর্দশ", "পঞ্চদশ"), 0, "কোন সংশোধনীতে তত্ত্বাবধায়ক সরকার যুক্ত হয়?"),  # same story again
        mcq("f4", ("মেট্রোরেল লাইন ১", "লাইন ৫", "লাইন ৬", "লাইন ২"), 0, "কোন মেট্রোরেল লাইন মেট্রোরেল লাইন ১ নামে পরিচিত?"),  # leaks
    ]}
    out = validate_mcqs(raw, FACTS, 3)
    assert [q["fact_id"] for q in out.questions] == ["f1", "f2"]
    assert out.questions[1]["basis"] == "gk"
    assert "question leans on 'the report/news' instead of standing alone" in out.reasons
    assert "story already has a question" in out.reasons and "question gives the answer away" in out.reasons


def test_fact_pool_interleaves_stories_and_includes_verified_gk():
    s1 = story("e1", "এক", ("তথ্য এক ১০০০", "তথ্য দুই ২০০০"))
    s2 = story("e2", "দুই", ("তথ্য তিন ৩০০০",))
    s2["analysis"]["static_gk"] = [{"statement": "জিকে বিবৃতি ১৯৭২", "question": "?", "answer": "১৯৭২", "answer_en": "1972", "kind": "year"}]
    pool = build_fact_pool([s1, s2])
    assert [(p["event_id"], p["basis"]) for p in pool] == [("e1", "article"), ("e2", "article"), ("e1", "article"), ("e2", "gk")]
    s2["analysis"]["key_facts"].append({"fact": "তথ্য চার ৪০০০", "evidence": "x", "prelims": True})
    pool = build_fact_pool([s2])
    assert [p["basis"] for p in pool] == ["article", "gk", "article"]  # GK is not pushed past the cap


def test_news_figure_questions_are_limited_while_alternatives_exist():
    def fact(fid, eid, text, basis="article"):
        return {"fact_id": fid, "story": eid, "fact": text, "evidence": text, "basis": basis, "subject": "economy",
                "event_id": eid, "url": "https://example.com/" + eid, "source_name": "X"}

    facts = [fact("f1", "e1", "রাজস্ব ঘাটতি ৪১,৮৭৯ কোটি টাকা"), fact("f2", "e2", "রপ্তানি ৫.২ বিলিয়ন ডলার"),
             fact("f3", "e3", "ডব্লিউএমও প্রতিষ্ঠিত ১৯৫০ সালে", "gk"), fact("f4", "e4", "সদর দপ্তর জেনেভা")]
    raw = [
        mcq("f1", ("৪১,৮৭৯ কোটি টাকা", "৩৯,৮৭৯ কোটি টাকা", "৪৩,৮৭৯ কোটি টাকা", "৪৫,৮৭৯ কোটি টাকা"), 0, "প্রথম প্রান্তিকে এনবিআরের রাজস্ব ঘাটতি কত?"),
        mcq("f2", ("৫.২ বিলিয়ন ডলার", "৪.২ বিলিয়ন ডলার", "৬.২ বিলিয়ন ডলার", "৩.২ বিলিয়ন ডলার"), 0, "সেপ্টেম্বরে রপ্তানি আয় কত?"),
        mcq("f3", ("১৯৫০", "১৯৪৫", "১৯৬০", "১৯৭২"), 0, "বিশ্ব আবহাওয়া সংস্থা কোন সালে প্রতিষ্ঠিত হয়?"),
        mcq("f4", ("জেনেভা", "নিউইয়র্ক", "রোম", "ভিয়েনা"), 0, "বিশ্ব আবহাওয়া সংস্থার সদর দপ্তর কোথায়?"),
    ]
    out = validate_mcqs({"questions": raw}, facts, 3)
    assert [q["fact_id"] for q in out.questions] == ["f1", "f3", "f4"]
    assert "enough 'how much' questions from the news" in out.reasons
    # A thin day with only figures still gets its questions.
    thin = validate_mcqs({"questions": raw[:2]}, facts, 3)
    assert [q["fact_id"] for q in thin.questions] == ["f1", "f2"]


def test_one_verified_gk_question_is_always_kept_when_valid():
    def fact(fid, eid, text, basis="article"):
        return {"fact_id": fid, "story": eid, "fact": text, "evidence": text, "basis": basis, "subject": "economy",
                "event_id": eid, "url": "https://example.com/" + eid, "source_name": "X"}

    facts = [fact("f1", "e1", "ছয় বছর মেয়াদ"), fact("f2", "e2", "সদর দপ্তর ঢাকা"), fact("f3", "e3", "বন্দর চট্টগ্রাম"),
             fact("f4", "e4", "ডব্লিউএমও প্রতিষ্ঠিত ১৯৫০ সালে", "gk")]
    raw = [
        mcq("f1", ("ছয় বছর", "পাঁচ বছর", "সাত বছর", "আট বছর"), 0, "পরিচালকদের টানা মেয়াদ কত বছর?"),
        mcq("f2", ("ঢাকা", "খুলনা", "রাজশাহী", "সিলেট"), 0, "সংস্থাটির সদর দপ্তর কোথায়?"),
        mcq("f3", ("চট্টগ্রাম", "মোংলা", "পায়রা", "মাতারবাড়ী"), 0, "দেশের প্রধান সমুদ্রবন্দর কোনটি?"),
        mcq("f4", ("১৯৫০", "১৯৪৫", "১৯৬০", "১৯৭২"), 0, "বিশ্ব আবহাওয়া সংস্থা কোন সালে প্রতিষ্ঠিত হয়?"),
    ]
    out = validate_mcqs({"questions": raw}, facts, 3)
    assert [q["fact_id"] for q in out.questions] == ["f1", "f2", "f4"]  # model order kept, GK guaranteed


def test_answer_position_is_not_always_the_first_option():
    from app.questions import arrange_options

    numeric = arrange_options({"question": "q?", "options": ["২০.৭ বিলিয়ন ডলার", "১৭.৭ বিলিয়ন ডলার", "২৪.৭ বিলিয়ন ডলার", "২২.৭ বিলিয়ন ডলার"], "answer_index": 0})
    assert numeric["options"] == ["১৭.৭ বিলিয়ন ডলার", "২০.৭ বিলিয়ন ডলার", "২২.৭ বিলিয়ন ডলার", "২৪.৭ বিলিয়ন ডলার"]
    assert numeric["answer_index"] == 1
    positions = set()
    for i in range(12):
        q = arrange_options({"question": f"প্রশ্ন {i}?", "options": ["কিনশাসা", "ব্রাজাভিল", "কাম্পালা", "লুসাকা"], "answer_index": 0})
        assert q["options"][q["answer_index"]] == "কিনশাসা" and sorted(q["options"]) == sorted(["কিনশাসা", "ব্রাজাভিল", "কাম্পালা", "লুসাকা"])
        positions.add(q["answer_index"])
    assert len(positions) >= 3  # spread over the options
    again = arrange_options({"question": "প্রশ্ন 0?", "options": ["কিনশাসা", "ব্রাজাভিল", "কাম্পালা", "লুসাকা"], "answer_index": 0})
    assert again["options"] == arrange_options({"question": "প্রশ্ন 0?", "options": ["কিনশাসা", "ব্রাজাভিল", "কাম্পালা", "লুসাকা"], "answer_index": 0})["options"]


def test_bengali_number_words_match_digits():
    from app.util import extract_numbers

    assert extract_numbers("৪১ হাজার ৮৭৯ কোটি টাকা") == extract_numbers("Tk 41,879 crore") == {"41879"}
    assert extract_numbers("১ লাখ ৩৪ হাজার ৩৯৬") == {"134396"} and extract_numbers("৩৩৯ দশমিক ১") == {"339.1"}
    # The NBR shortfall written both ways is now recognised as the same story.
    a = distinctive_numbers(["রাজস্ব ঘাটতি ৪১ হাজার ৮৭৯ কোটি টাকা", "লক্ষ্য ১ লাখ ৩৪ হাজার ৩৯৬ কোটি"])
    b = distinctive_numbers(["ঘাটতি ৪১,৮৭৯ কোটি টাকা", "লক্ষ্য ১,৩৪,৩৯৬ কোটি টাকা"])
    assert {"41879", "134396"} <= a & b


def test_malformed_model_values_never_crash(profile, settings):
    """Lists/dicts where strings are expected must be rejected, not raise TypeError."""
    smap = {"s1": story()}
    kept, _ = validate_gk({"items": [gk(story=["s1", "s2"]), gk(story={"id": "s1"}), gk(kind=["year"])]}, smap)
    assert [k["kind"] for k in kept] == ["other"]
    note = validate_analysis(note_raw(key_facts=[{"fact": "নীতি সুদহার 10 percent", "evidence": "raised the policy rate to 10 percent", "kind": ["number"], "prelims": True}]),
                             SOURCE, "h", None, WEIGHTS, profile, 7)
    assert note["key_facts"][0]["kind"] == "other"
    from app.analysis import AnalysisRejected

    with pytest.raises(AnalysisRejected):
        validate_analysis(note_raw(syllabus_topic=["bd_banking_monetary"]), SOURCE, "h", None, WEIGHTS, profile, 7)
    out = validate_screening({"results": [screen_item("c1", decision=["select"]), screen_item("c2", topic=["x"], exam_focus={"a": 1}, story_type=["policy_decision"])]},
                             {"c1": "e1", "c2": "e2"}, profile)
    assert out.invalid_items == 1 and out.decisions["e2"]["decision"] == "reject"
    q = validate_mcqs({"questions": [{**mcq(), "subject": ["economy"]}]}, FACTS, 3)
    assert q.questions and q.questions[0]["subject"] == "economy"


def test_a_crashing_optional_step_never_costs_the_edition(make_pipeline, monkeypatch):
    import app.pipeline as pipeline_mod

    def boom(*a, **k):
        raise TypeError("unhashable type: 'list'")

    monkeypatch.setattr(pipeline_mod, "enrich_static_gk", boom)
    monkeypatch.setattr(pipeline_mod, "generate_mcqs", boom)
    tg = FakeTelegram()
    r = make_pipeline(telegram=tg).run(now=NOW)
    assert r.outcome == "delivered" and r.stories == 3 and tg.sent
    assert any("Static GK skipped" in e for e in r.errors) and any("MCQ generation failed" in e for e in r.errors)
