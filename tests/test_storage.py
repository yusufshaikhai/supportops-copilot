from pathlib import Path

import pytest

from storage import (
    ReviewDecision,
    ReviewStatus,
    create_review,
    dashboard_metrics,
    list_audit_events,
    record_decision,
    save_draft,
)
from supportops import Ticket, classify_ticket, retrieve_knowledge


def create_shipping_review(database: Path):
    ticket = Ticket(text="My package is late and tracking has not changed")
    return create_review(
        ticket,
        classify_ticket(ticket),
        retrieve_knowledge(ticket),
        database,
    )


def test_review_lifecycle_persists_draft_edit_and_audit_history(
    tmp_path: Path,
) -> None:
    database = tmp_path / "supportops.db"
    review = create_shipping_review(database)

    assert review.status is ReviewStatus.READY_TO_DRAFT
    drafted = save_draft(
        review.id,
        "We will review the carrier scans [HH-SHIP-001].",
        ["HH-SHIP-001"],
        "test-model",
        database,
    )
    edited = record_decision(
        review.id,
        ReviewDecision.EDIT,
        "We are reviewing the latest carrier scans [HH-SHIP-001].",
        "Clarified the timing.",
        database,
    )

    assert drafted.status is ReviewStatus.DRAFTED
    assert edited.status is ReviewStatus.EDITED
    assert edited.draft_text.startswith("We are reviewing")
    assert [event.event_type for event in list_audit_events(review.id, database)] == [
        "draft_edited",
        "draft_created",
        "review_created",
    ]


def test_dashboard_summarizes_approve_edit_and_reject_feedback(
    tmp_path: Path,
) -> None:
    database = tmp_path / "supportops.db"
    decisions = [
        ReviewDecision.APPROVE,
        ReviewDecision.EDIT,
        ReviewDecision.REJECT,
    ]
    for decision in decisions:
        review = create_shipping_review(database)
        save_draft(
            review.id,
            "We will review the carrier scans [HH-SHIP-001].",
            ["HH-SHIP-001"],
            "test-model",
            database,
        )
        record_decision(
            review.id,
            decision,
            edited_draft=(
                "We are checking the carrier scans [HH-SHIP-001]."
                if decision is ReviewDecision.EDIT
                else None
            ),
            path=database,
        )

    metrics = dashboard_metrics(database)

    assert metrics.total_reviews == 3
    assert metrics.approved == 1
    assert metrics.edited == 1
    assert metrics.rejected == 1
    assert metrics.acceptance_rate == pytest.approx(0.667)


def test_final_decisions_are_immutable_and_edits_keep_original_and_citations(
    tmp_path: Path,
) -> None:
    database = tmp_path / "supportops.db"
    review = create_shipping_review(database)
    original = "We will review the carrier scans [HH-SHIP-001]."
    save_draft(review.id, original, ["HH-SHIP-001"], "test-model", database)

    with pytest.raises(ValueError, match="cite"):
        record_decision(
            review.id, ReviewDecision.EDIT, "No citation here.", path=database
        )
    with pytest.raises(ValueError, match="unsupported"):
        record_decision(
            review.id, ReviewDecision.EDIT, "Fake claim [HH-FAKE-999].", path=database
        )

    edited_text = "Updated reply [HH-SHIP-001]."
    record_decision(review.id, ReviewDecision.EDIT, edited_text, path=database)
    event = list_audit_events(review.id, database)[0]
    assert event.details["original_draft"] == original
    assert event.details["edited_draft"] == edited_text

    with pytest.raises(ValueError, match="final decision"):
        record_decision(review.id, ReviewDecision.REJECT, path=database)
    with pytest.raises(ValueError, match="cannot be drafted"):
        save_draft(review.id, original, ["HH-SHIP-001"], "test-model", database)
    assert dashboard_metrics(database).edited == 1
