from pathlib import Path

from supportops import (
    Category,
    ErrorCause,
    Method,
    Ticket,
    Urgency,
    analyze_errors,
    classify_ticket,
    compare,
    load_dataset,
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

    assert result.ml.category_macro_f1 > 0.81
    assert result.ml.urgent_recall == 1
    assert result.macro_f1_delta > 0


def test_error_analysis_accounts_for_every_ml_failure() -> None:
    analysis = analyze_errors(EVALUATION_DATASET, TRAINING_DATASET)

    assert analysis.error_count == 11
    assert analysis.low_confidence_rejections == 6
    assert sum(item.count for item in analysis.confusion_counts) == 11
    assert sum(item.total for item in analysis.category_recall) == 50
    assert all(error.confidence > 0 for error in analysis.errors)
    assert all(
        error.confidence < analysis.confidence_threshold
        for error in analysis.errors
        if error.cause is ErrorCause.LOW_CONFIDENCE_REJECTION
    )
    assert {error.cause for error in analysis.errors} == {
        ErrorCause.LOW_CONFIDENCE_REJECTION,
        ErrorCause.CATEGORY_CONFUSION,
    }


def test_right_now_is_an_explicit_urgent_signal() -> None:
    prediction = classify_ticket(
        Ticket(text="I cannot get into my profile and need entry right now"),
        Method.ML,
        TRAINING_DATASET,
    )

    assert prediction.urgency is Urgency.URGENT


def test_error_analysis_handles_categories_absent_from_a_dataset(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "shipping-only.csv"
    dataset.write_text(
        "text,category,urgency\nWhere is my package,shipping,normal\n",
        encoding="utf-8",
    )

    analysis = analyze_errors(dataset, TRAINING_DATASET)
    account_recall = next(
        item for item in analysis.category_recall if item.category is Category.ACCOUNT
    )

    assert account_recall.total == 0
    assert account_recall.recall == 0


def test_fixed_corpora_have_expected_sizes_and_no_overlap() -> None:
    training = load_dataset(TRAINING_DATASET)
    evaluation = load_dataset(EVALUATION_DATASET)

    assert len(training) == 115
    assert len(evaluation) == 50
    assert {t.text for t in training}.isdisjoint({t.text for t in evaluation})
