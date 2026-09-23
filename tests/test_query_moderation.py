from types import SimpleNamespace

import pytest

from app.services.query_moderation import ModerationUnavailable, moderate_query


LABELS = ("toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit")


def predictor(*overrides):
    """Return classifier scores for successive clauses."""
    def predict(clauses):
        assert len(clauses) == len(overrides)
        return {
            label: [row.get(label, 0.0) for row in overrides]
            for label in LABELS
        }
    return SimpleNamespace(predict=predict)


def test_clothing_criticism_is_not_a_personal_attack():
    result = moderate_query(
        "Este vestido es una puta mierda, ¿me enseñas otro?",
        predictor({"toxicity": 0.991, "insult": 0.071}, {"toxicity": 0.005}),
    )
    assert not result.blocked


def test_sexual_vocabulary_in_a_fashion_question_is_allowed():
    result = moderate_query(
        "Quiero una camiseta para una clase de educación sexual",
        predictor({"toxicity": 0.864, "sexual_explicit": 0.021}),
    )
    assert not result.blocked


def test_direct_insult_after_fashion_question_is_blocked():
    result = moderate_query(
        "¿Combina este top con la falda? Eres un puto idiota.",
        predictor({"toxicity": 0.003}, {"toxicity": 0.992, "insult": 0.406}),
    )
    assert result.blocked
    assert result.reason == "toxicity"


def test_direct_threat_with_low_model_threat_score_is_blocked():
    result = moderate_query(
        "Te voy a matar",
        predictor({"toxicity": 0.990, "threat": 0.134}),
    )
    assert result.blocked
    assert result.reason == "threat"


def test_broken_classifier_fails_closed():
    with pytest.raises(ModerationUnavailable):
        moderate_query("What jacket matches?", SimpleNamespace(predict=lambda _: {}))
