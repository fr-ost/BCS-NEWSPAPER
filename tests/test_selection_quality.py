"""Recalibrated selection: syllabus topics, story types, thresholds, filters."""

from dataclasses import replace

import pytest

from app.analysis import AnalysisRejected, validate_analysis
from app.collector import FeedItem
from app.config import ConfigError
from app.scoring import rank_stories
from app.screening import validate_screening
from app.syllabus import load_syllabus, parse_syllabus

from .conftest import NOW, FakeTelegram

WEIGHTS = {"syllabus_relevance": 0.30, "significance": 0.25, "factual_density": 0.20, "written_potential": 0.15, "novelty": 0.10}
SOURCE = "Bangladesh Bank raised the policy rate to 10 percent on Monday, citing inflation of 9.2 percent in September."


@pytest.fixture
def profile(settings):
    return load_syllabus(settings.config_dir)


def test_real_syllabus_profile_is_valid(profile):
    assert len(profile.topics) >= 40
    assert "bd_budget_fiscal" in profile.topics and "int_conflicts" in profile.topics
    assert "opinion" in profile.rejected_types and "policy_decision" in profile.accepted_types
    assert "bd_constitution" in profile.catalog_text() and "✗" in profile.examples_text()


def test_syllabus_validation():
    with pytest.raises(ConfigError):
        parse_syllabus({})
    with pytest.raises(ConfigError):
        parse_syllabus({"topic": {"x": {"label": "x", "subject": "astrology"}}, "policy": {"accepted_types": ["a"], "rejected_types": ["b"]}})
    with pytest.raises(ConfigError):
        parse_syllabus({"topic": {"x": {"label": "x", "subject": "economy"}}, "policy": {"accepted_types": ["a"], "rejected_types": ["a"]}})


def item(cid, decision="select", story_type="policy_decision", topic="bd_banking_monetary", priority=8):
    return {"id": cid, "decision": decision, "story_type": story_type, "topic": topic, "priority": priority,
            "categories": ["economy"], "exam_hook": "policy rate", "rationale": "r"}


def test_code_enforces_screening_policy(profile):
    id_map = {f"c{i}": f"ev_{i}" for i in range(1, 8)}
    raw = {"results": [
        item("c1"),                                              # good
        item("c2", story_type="opinion"),                        # opinion column
        item("c3", story_type="party_politics"),                 # procession / party statement
        item("c4", topic="none"),                                # no syllabus topic
        item("c5", topic="made_up_topic"),                       # hallucinated topic
        item("c6", decision="maybe", priority=5),                # weak maybe
        item("c7", decision="maybe", priority=7, story_type="diplomacy", topic="fp_bilateral"),  # strong maybe
    ]}
    out = validate_screening(raw, id_map, profile, maybe_min_priority=7)
    d = {eid: (r["decision"], r.get("forced_reject")) for eid, r in out.decisions.items()}
    assert d["ev_1"] == ("select", None)
    assert d["ev_2"] == ("reject", "story type opinion")
    assert d["ev_3"] == ("reject", "story type party_politics")
    assert d["ev_4"] == ("reject", "no syllabus topic")
    assert d["ev_5"] == ("reject", "no syllabus topic")
    assert d["ev_6"][0] == "reject" and "weak maybe" in d["ev_6"][1]
    assert d["ev_7"] == ("maybe", None)
    # Unknown story type is treated as low value.
    weird = validate_screening({"results": [item("c1", story_type="gossip")]}, {"c1": "e"}, profile)
    assert weird.decisions["e"]["decision"] == "reject" and weird.decisions["e"]["story_type"] == "other_low_value"


def analysis(**kw):
    raw = {
        "relevant": True, "evidence_sufficient": True, "headline_bn": "নীতি সুদহার বাড়ল",
        "syllabus_topic": "bd_banking_monetary", "categories": ["constitution_governance", "economy", "bangladesh_affairs"],
        "exam_focus": "prelims", "likely_question": "বর্তমান নীতি সুদহার কত?",
        "what_happened": "বাংলাদেশ ব্যাংক নীতি সুদহার 10 percent করেছে।", "why_it_matters": "মুদ্রানীতি।",
        "key_facts": [{"fact": "নীতি সুদহার 10 percent", "evidence": "raised the policy rate to 10 percent", "prelims": True}],
        "attributed_claims": [], "written_angle": "", "causes_effects": "", "background_context": "", "institutions": [],
        "scores": {"syllabus_relevance": 9, "significance": 8, "factual_density": 8, "written_potential": 6, "novelty": 8},
    }
    raw.update(kw)
    return raw


def test_analysis_requires_valid_topic_and_relevance(profile):
    note = validate_analysis(analysis(), SOURCE, "h", None, WEIGHTS, profile, 7)
    assert note["syllabus_topic"] == "bd_banking_monetary" and note["likely_question"]
    assert note["categories"] == ["economy", "constitution_governance"]  # topic subject first, max 2
    with pytest.raises(AnalysisRejected, match="syllabus topic"):
        validate_analysis(analysis(syllabus_topic="celebrity_news"), SOURCE, "h", None, WEIGHTS, profile, 7)
    low = analysis(scores={"syllabus_relevance": 6, "significance": 9, "factual_density": 9, "written_potential": 9, "novelty": 9})
    with pytest.raises(AnalysisRejected, match="relevance 6"):
        validate_analysis(low, SOURCE, "h", None, WEIGHTS, profile, 7)


def test_ranking_keeps_one_story_per_topic():
    def st(eid, score, topic):
        return {"event_id": eid, "score": score, "analysis": {"syllabus_topic": topic}}

    stories = [st("a", 86, "env_pollution_conservation"), st("b", 84, "env_pollution_conservation"),
               st("c", 80, "fp_bilateral"), st("d", 60, "int_conflicts")]
    picked = rank_stories(stories, min_score=65, max_stories=8, max_per_topic=1)
    assert [s["event_id"] for s in picked] == ["a", "c"]


def feed_item(url, title="A sufficiently long title about national policy"):
    return FeedItem(id="x", source_id="prothomalo", source_name="P", title=title, url=url, canonical_url=url,
                    excerpt="", published_at=None, retrieved_at="", date_status="missing", language="bn")


@pytest.mark.parametrize("url,title,dropped", [
    ("https://www.prothomalo.com/sports/cricket/abc", None, True),
    ("https://www.prothomalo.com/opinion/column/abc", None, True),
    ("https://www.prothomalo.com/bangladesh/district/abc", None, True),
    ("https://www.prothomalo.com/bangladesh/crime/abc", None, True),
    ("https://www.prothomalo.com/religion/islam/abc", None, True),
    ("https://www.prothomalo.com/sports", None, True),  # section root without trailing slash
    ("https://www.prothomalo.com/bangladesh/abc", None, False),
    ("https://www.prothomalo.com/business/economics/abc", None, False),
    ("https://www.prothomalo.com/world/middle-east/abc", None, False),
    ("https://www.prothomalo.com/bangladesh/abc", "আজ টিভিতে যা দেখবেন (২৫ অক্টোবর)", True),
    ("https://www.thedailystar.net/news/bangladesh/abc", "Daily horoscope for Libra", True),
])
def test_prefilter(make_pipeline, url, title, dropped):
    pipe = make_pipeline()
    assert pipe._excluded(feed_item(url, title or "জাতীয় বাজেট পাস হলো সংসদে")) is dropped


def test_prefiltered_items_cost_no_model_calls(make_pipeline, settings):
    from .test_pipeline import Programmable

    prov = Programmable()
    pipe = make_pipeline(provider=prov, telegram=FakeTelegram())
    pipe.run(now=NOW)
    state_titles = {a["title"] for a in __import__("json").loads(settings.state_path.read_text())["articles"].values()}
    assert not any("friendly football" in t or "ছবির শুটিং" in t for t in state_titles)  # /sports/, /entertainment/
    assert pipe.prefiltered >= 2


def test_rendered_story_shows_syllabus_and_question(make_pipeline):
    tg = FakeTelegram()
    make_pipeline(telegram=tg).run(now=NOW)
    text = "\n".join(tg.sent)
    assert "🎯 সিলেবাস:" in text and "📝 সম্ভাব্য লিখিত প্রশ্ন:" in text
    assert "<b>আজকের শিরোনাম</b>" in text and "📚 প্রাসঙ্গিক সাধারণ জ্ঞান" in text and "✍️ লিখিত উত্তরের পয়েন্ট" in text


def test_old_rule_items_are_rescreened_and_old_analyses_ignored(make_pipeline, settings):
    import json

    from .test_pipeline import Programmable

    make_pipeline(telegram=FakeTelegram()).run(now=NOW)
    state = json.loads(settings.state_path.read_text())
    # Simulate pre-calibration records: one 'maybe' screening without story_type,
    # one analysed story without syllabus_topic, neither yet sent.
    evs = [e for e in state["events"].values() if e["status"] == "included"]
    old_screen, old_analysis = evs[0], evs[1]
    old_screen.update(status="maybe", screening={"decision": "maybe", "priority": 9, "categories": [], "rationale": "", "exam_focus": "none"})
    old_analysis["status"] = "analysed"
    old_analysis["analysis"].pop("syllabus_topic")
    settings.state_path.write_text(json.dumps(state))
    prov = Programmable()
    tg = FakeTelegram()
    r = make_pipeline(provider=prov, telegram=tg).run(now=NOW, manual=True)
    assert "screening" in prov.tasks  # the old 'maybe' was screened again under the new rules
    after = json.loads(settings.state_path.read_text())["events"]
    assert "story_type" in (after[old_screen["event_id"]].get("screening") or {})
    assert old_analysis["event_id"] not in json.loads(settings.state_path.read_text())["editions"][r.edition_date]["story_event_ids"]
