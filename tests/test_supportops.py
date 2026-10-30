from pathlib import Path

from supportops import (
    Category,
    Method,
    Ticket,
    Urgency,
    classify_ticket,
    compare,
)

ROOT = Path(__file__).parents[1]
TRAINING_DATASET = ROOT / "data" / "training.csv"
EVALUATION_DATASET = ROOT / "data" / "evaluation.csv"


def test_rules_remain_an_explainable_baseline() -> None:
    prediction = classify_ticket(
        Ticket(text="My order arrived broken"),
        Method.RULES,
    )

    assert prediction.category is Category.DAMAGED_ITEM
    assert prediction.urgency is Urgency.URGENT
    assert prediction.matched_keywords == ["broken"]


def test_ml_handles_wording_missing_from_rules() -> None:
    prediction = classify_ticket(
        Ticket(text="The wooden panel is warped and will not fit"),
        Method.ML,
        TRAINING_DATASET,
    )

    assert prediction.category is Category.DAMAGED_ITEM
    assert prediction.method is Method.ML


def test_ml_requires_review_for_unrelated_text() -> None:
    prediction = classify_ticket(
        Ticket(text="Hello there"),
        Method.ML,
        TRAINING_DATASET,
    )

    assert prediction.category is Category.UNKNOWN


def test_ml_beats_rules_and_meets_starter_targets() -> None:
    result = compare(EVALUATION_DATASET, TRAINING_DATASET)

    assert result.ml.category_macro_f1 >= 0.80
    assert result.ml.urgent_recall >= 0.90
    assert result.macro_f1_delta > 0
