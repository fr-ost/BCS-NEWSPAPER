import pytest

from app.config import ConfigError, validate_weights
from app.scoring import ScoreError, rank_stories, validate_criteria, weighted_score

WEIGHTS = {"syllabus_relevance": 0.30, "significance": 0.25, "factual_density": 0.20, "written_potential": 0.15, "novelty": 0.10}


def crit(**kw):
    base = dict.fromkeys(WEIGHTS, 0)
    base.update(kw)
    return base


def test_weighted_score_matches_rubric():
    assert weighted_score(validate_criteria(crit(**dict.fromkeys(WEIGHTS, 10))), WEIGHTS) == 100.0
    assert weighted_score(validate_criteria(crit(syllabus_relevance=10)), WEIGHTS) == 30.0
    assert weighted_score(validate_criteria(crit(significance=8, novelty=5)), WEIGHTS) == 25.0


@pytest.mark.parametrize("bad", [None, [], {"syllabus_relevance": 5}, crit(significance=11), crit(novelty=-1), crit(novelty="7"), crit(novelty=True)])
def test_invalid_scores_rejected(bad):
    with pytest.raises(ScoreError):
        validate_criteria(bad)


def test_weights_must_sum_to_one():
    validate_weights(WEIGHTS)
    with pytest.raises(ConfigError):
        validate_weights({**WEIGHTS, "novelty": 0.5})
    with pytest.raises(ConfigError):
        validate_weights({"syllabus_relevance": 1.0})


def test_ranking_never_pads_low_relevance_days():
    stories = [{"event_id": f"e{i}", "score": s} for i, s in enumerate([90, 40, 70, 30, 54.9])]
    picked = rank_stories(stories, min_score=55, max_stories=8)
    assert [s["score"] for s in picked] == [90, 70]


def test_ranking_respects_max():
    stories = [{"event_id": f"e{i}", "score": 60 + i} for i in range(12)]
    assert len(rank_stories(stories, 55, 8)) == 8
