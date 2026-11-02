from __future__ import annotations

import argparse
import csv
from collections import Counter
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import Pipeline


class Category(StrEnum):
    SHIPPING = "shipping"
    REFUND = "refund"
    DAMAGED_ITEM = "damaged_item"
    BILLING = "billing"
    ACCOUNT = "account"
    UNKNOWN = "unknown"


class Urgency(StrEnum):
    NORMAL = "normal"
    URGENT = "urgent"


class Method(StrEnum):
    RULES = "rules"
    ML = "ml"


class ErrorCause(StrEnum):
    LOW_CONFIDENCE_REJECTION = "low_confidence_rejection"
    NO_KEYWORD_MATCH = "no_keyword_match"
    CATEGORY_CONFUSION = "category_confusion"


class Ticket(BaseModel):
    text: str = Field(min_length=3)


class LabeledTicket(Ticket):
    category: Category
    urgency: Urgency


class Prediction(BaseModel):
    category: Category
    urgency: Urgency
    confidence: float = Field(ge=0, le=1)
    method: Method
    matched_keywords: list[str] = []


class ClassificationError(BaseModel):
    text: str
    expected: Category
    predicted: Category
    confidence: float = Field(ge=0, le=1)
    cause: ErrorCause


class EvaluationResult(BaseModel):
    method: Method
    sample_count: int
    category_accuracy: float
    category_macro_f1: float
    urgent_recall: float
    unknown_rate: float
    errors: list[ClassificationError]


class ComparisonResult(BaseModel):
    rules: EvaluationResult
    ml: EvaluationResult
    macro_f1_delta: float


class CategoryRecall(BaseModel):
    category: Category
    correct: int
    total: int
    recall: float = Field(ge=0, le=1)


class ConfusionCount(BaseModel):
    expected: Category
    predicted: Category
    count: int = Field(gt=0)


class ErrorAnalysis(BaseModel):
    method: Method
    sample_count: int
    error_count: int
    low_confidence_rejections: int
    confidence_threshold: float
    category_recall: list[CategoryRecall]
    confusion_counts: list[ConfusionCount]
    errors: list[ClassificationError]


CATEGORY_KEYWORDS = {
    Category.SHIPPING: (
        "shipping",
        "shipment",
        "delivery",
        "delivered",
        "package",
        "tracking",
        "courier",
        "arrive",
        "arrival",
        "lost order",
    ),
    Category.REFUND: (
        "refund",
        "return",
        "money back",
        "reimburse",
        "send back",
        "exchange",
    ),
    Category.DAMAGED_ITEM: (
        "damaged",
        "broken",
        "cracked",
        "defective",
        "scratched",
        "shattered",
        "missing pieces",
        "does not work",
        "doesn't work",
    ),
    Category.BILLING: (
        "charged",
        "charge",
        "payment",
        "invoice",
        "billing",
        "card",
        "receipt",
        "price",
        "coupon",
        "discount",
        "tax",
    ),
    Category.ACCOUNT: (
        "account",
        "login",
        "password",
        "email address",
        "sign in",
        "signin",
        "profile",
        "locked out",
        "reset",
    ),
}

CATEGORY_TIE_BREAK = {
    Category.DAMAGED_ITEM: 5,
    Category.BILLING: 4,
    Category.ACCOUNT: 3,
    Category.REFUND: 2,
    Category.SHIPPING: 1,
}

URGENT_KEYWORDS = (
    "urgent",
    "immediately",
    "asap",
    "damaged",
    "broken",
    "cracked",
    "shattered",
    "snapped",
    "lost",
    "missing",
    "charged twice",
    "duplicate charge",
    "pending payments",
    "unauthorized",
    "locked out",
    "right now",
    "another person's",
    "someone else's",
)

DATA_DIRECTORY = Path(__file__).parent / "data"
DEFAULT_TRAINING_DATASET = DATA_DIRECTORY / "training.csv"
DEFAULT_EVALUATION_DATASET = DATA_DIRECTORY / "evaluation.csv"
ML_CONFIDENCE_THRESHOLD = 0.24


def load_dataset(dataset: Path) -> list[LabeledTicket]:
    tickets: list[LabeledTicket] = []
    with dataset.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"text", "category", "urgency"}
        if set(reader.fieldnames or ()) != required:
            raise ValueError(
                f"{dataset} must contain exactly: {', '.join(sorted(required))}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                ticket = LabeledTicket.model_validate(row)
            except ValueError as error:
                raise ValueError(
                    f"Invalid row {line_number} in {dataset}: {error}"
                ) from error
            if ticket.category is Category.UNKNOWN:
                raise ValueError(
                    f"Invalid row {line_number}: expected category cannot be unknown"
                )
            tickets.append(ticket)

    if not tickets:
        raise ValueError(f"{dataset} contains no tickets")
    return tickets


def classify_urgency(text: str) -> Urgency:
    normalized = text.casefold()
    return (
        Urgency.URGENT
        if any(keyword in normalized for keyword in URGENT_KEYWORDS)
        else Urgency.NORMAL
    )


def classify_with_rules(ticket: Ticket) -> Prediction:
    text = ticket.text.casefold()
    matches = {
        category: [keyword for keyword in keywords if keyword in text]
        for category, keywords in CATEGORY_KEYWORDS.items()
    }
    scores = {
        category: sum(len(keyword) for keyword in keywords)
        for category, keywords in matches.items()
    }
    category = max(
        scores,
        key=lambda candidate: (scores[candidate], CATEGORY_TIE_BREAK[candidate]),
    )
    top_score = scores[category]
    total_score = sum(scores.values())
    if top_score == 0:
        category = Category.UNKNOWN

    return Prediction(
        category=category,
        urgency=classify_urgency(ticket.text),
        confidence=round(top_score / total_score, 3) if total_score else 0,
        method=Method.RULES,
        matched_keywords=matches.get(category, []),
    )


@lru_cache(maxsize=1)
def train_category_model(
    training_dataset: Path = DEFAULT_TRAINING_DATASET,
) -> Pipeline:
    tickets = load_dataset(training_dataset)
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=0,
                ),
            ),
        ]
    ).fit(
        [ticket.text for ticket in tickets],
        [ticket.category.value for ticket in tickets],
    )


def classify_with_ml(
    ticket: Ticket,
    training_dataset: Path = DEFAULT_TRAINING_DATASET,
) -> Prediction:
    model = train_category_model(training_dataset)
    probabilities = model.predict_proba([ticket.text])[0]
    best_index = int(probabilities.argmax())
    confidence = float(probabilities[best_index])
    category = (
        Category(model.classes_[best_index])
        if confidence >= ML_CONFIDENCE_THRESHOLD
        else Category.UNKNOWN
    )
    return Prediction(
        category=category,
        urgency=classify_urgency(ticket.text),
        confidence=round(confidence, 4),
        method=Method.ML,
    )


def classify_ticket(
    ticket: Ticket,
    method: Method = Method.ML,
    training_dataset: Path = DEFAULT_TRAINING_DATASET,
) -> Prediction:
    if method is Method.RULES:
        return classify_with_rules(ticket)
    return classify_with_ml(ticket, training_dataset)


def _classify_dataset(
    method: Method,
    dataset: Path,
    training_dataset: Path,
) -> list[tuple[LabeledTicket, Prediction]]:
    return [
        (
            labeled_ticket,
            classify_ticket(
                Ticket(text=labeled_ticket.text),
                method,
                training_dataset,
            ),
        )
        for labeled_ticket in load_dataset(dataset)
    ]


def _classification_error(
    labeled_ticket: LabeledTicket,
    prediction: Prediction,
) -> ClassificationError | None:
    if prediction.category is labeled_ticket.category:
        return None
    if prediction.category is not Category.UNKNOWN:
        cause = ErrorCause.CATEGORY_CONFUSION
    elif prediction.method is Method.ML:
        cause = ErrorCause.LOW_CONFIDENCE_REJECTION
    else:
        cause = ErrorCause.NO_KEYWORD_MATCH
    return ClassificationError(
        text=labeled_ticket.text,
        expected=labeled_ticket.category,
        predicted=prediction.category,
        confidence=prediction.confidence,
        cause=cause,
    )


def evaluate(
    method: Method = Method.ML,
    dataset: Path = DEFAULT_EVALUATION_DATASET,
    training_dataset: Path = DEFAULT_TRAINING_DATASET,
) -> EvaluationResult:
    expected_categories: list[str] = []
    predicted_categories: list[str] = []
    expected_urgency: list[int] = []
    predicted_urgency: list[int] = []
    errors: list[ClassificationError] = []

    for labeled_ticket, prediction in _classify_dataset(
        method, dataset, training_dataset
    ):
        expected_categories.append(labeled_ticket.category.value)
        predicted_categories.append(prediction.category.value)
        expected_urgency.append(labeled_ticket.urgency is Urgency.URGENT)
        predicted_urgency.append(prediction.urgency is Urgency.URGENT)
        if error := _classification_error(labeled_ticket, prediction):
            errors.append(error)

    labels = [
        category.value for category in Category if category is not Category.UNKNOWN
    ]
    return EvaluationResult(
        method=method,
        sample_count=len(expected_categories),
        category_accuracy=round(
            accuracy_score(expected_categories, predicted_categories), 3
        ),
        category_macro_f1=round(
            f1_score(
                expected_categories,
                predicted_categories,
                labels=labels,
                average="macro",
                zero_division=0,
            ),
            3,
        ),
        urgent_recall=round(
            recall_score(expected_urgency, predicted_urgency, zero_division=0), 3
        ),
        unknown_rate=round(
            predicted_categories.count(Category.UNKNOWN.value)
            / len(predicted_categories),
            3,
        ),
        errors=errors,
    )


def compare(
    dataset: Path = DEFAULT_EVALUATION_DATASET,
    training_dataset: Path = DEFAULT_TRAINING_DATASET,
) -> ComparisonResult:
    rules = evaluate(Method.RULES, dataset, training_dataset)
    ml = evaluate(Method.ML, dataset, training_dataset)
    return ComparisonResult(
        rules=rules,
        ml=ml,
        macro_f1_delta=round(ml.category_macro_f1 - rules.category_macro_f1, 3),
    )


def analyze_errors(
    dataset: Path = DEFAULT_EVALUATION_DATASET,
    training_dataset: Path = DEFAULT_TRAINING_DATASET,
) -> ErrorAnalysis:
    categories = [
        category for category in Category if category is not Category.UNKNOWN
    ]
    totals: Counter[Category] = Counter()
    correct: Counter[Category] = Counter()
    confusions: Counter[tuple[Category, Category]] = Counter()
    errors: list[ClassificationError] = []

    predictions = _classify_dataset(Method.ML, dataset, training_dataset)
    for labeled_ticket, prediction in predictions:
        totals[labeled_ticket.category] += 1
        if prediction.category is labeled_ticket.category:
            correct[labeled_ticket.category] += 1
            continue

        confusions[(labeled_ticket.category, prediction.category)] += 1
        error = _classification_error(labeled_ticket, prediction)
        if error:
            errors.append(error)

    return ErrorAnalysis(
        method=Method.ML,
        sample_count=len(predictions),
        error_count=len(errors),
        low_confidence_rejections=sum(
            error.cause is ErrorCause.LOW_CONFIDENCE_REJECTION for error in errors
        ),
        confidence_threshold=ML_CONFIDENCE_THRESHOLD,
        category_recall=[
            CategoryRecall(
                category=category,
                correct=correct[category],
                total=totals[category],
                recall=(
                    round(correct[category] / totals[category], 3)
                    if totals[category]
                    else 0
                ),
            )
            for category in categories
        ],
        confusion_counts=[
            ConfusionCount(expected=expected, predicted=predicted, count=count)
            for expected in categories
            for predicted in Category
            if (count := confusions[(expected, predicted)])
        ],
        errors=errors,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="supportops")
    commands = parser.add_subparsers(dest="command", required=True)

    classify_command = commands.add_parser("classify", help="classify one ticket")
    classify_command.add_argument("text")
    classify_command.add_argument(
        "--method",
        choices=Method,
        type=Method,
        default=Method.ML,
    )
    classify_command.add_argument(
        "--training-dataset",
        type=Path,
        default=DEFAULT_TRAINING_DATASET,
    )

    evaluate_command = commands.add_parser("evaluate", help="evaluate one method")
    evaluate_command.add_argument(
        "--method",
        choices=Method,
        type=Method,
        default=Method.ML,
    )
    evaluate_command.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EVALUATION_DATASET,
    )
    evaluate_command.add_argument(
        "--training-dataset",
        type=Path,
        default=DEFAULT_TRAINING_DATASET,
    )

    compare_command = commands.add_parser("compare", help="compare rules with ML")
    compare_command.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EVALUATION_DATASET,
    )
    compare_command.add_argument(
        "--training-dataset",
        type=Path,
        default=DEFAULT_TRAINING_DATASET,
    )

    analyze_command = commands.add_parser(
        "analyze-errors",
        help="summarize ML classification errors",
    )
    analyze_command.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EVALUATION_DATASET,
    )
    analyze_command.add_argument(
        "--training-dataset",
        type=Path,
        default=DEFAULT_TRAINING_DATASET,
    )

    arguments = parser.parse_args()
    if arguments.command == "classify":
        result = classify_ticket(
            Ticket(text=arguments.text),
            arguments.method,
            arguments.training_dataset,
        )
    elif arguments.command == "evaluate":
        result = evaluate(
            arguments.method,
            arguments.dataset,
            arguments.training_dataset,
        )
    elif arguments.command == "compare":
        result = compare(arguments.dataset, arguments.training_dataset)
    else:
        result = analyze_errors(arguments.dataset, arguments.training_dataset)

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
