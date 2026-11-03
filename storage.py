from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from drafting import CITATION_PATTERN
from supportops import Category, KnowledgeMatch, Method, Prediction, Ticket, Urgency

DEFAULT_DATABASE_PATH = Path(__file__).parent / "data" / "supportops.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_text TEXT NOT NULL,
    category TEXT NOT NULL,
    urgency TEXT NOT NULL,
    confidence REAL NOT NULL,
    method TEXT NOT NULL,
    knowledge_json TEXT NOT NULL,
    draft_text TEXT,
    draft_citations_json TEXT NOT NULL DEFAULT '[]',
    draft_model TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id INTEGER NOT NULL REFERENCES reviews(id),
    event_type TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""


class ReviewStatus(StrEnum):
    READY_TO_DRAFT = "ready_to_draft"
    NEEDS_EVIDENCE = "needs_evidence"
    DRAFTED = "drafted"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"


class ReviewDecision(StrEnum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


class ReviewRecord(BaseModel):
    id: int
    ticket_text: str
    category: Category
    urgency: Urgency
    confidence: float = Field(ge=0, le=1)
    method: Method
    knowledge_matches: list[KnowledgeMatch]
    draft_text: str | None
    draft_citations: list[str]
    draft_model: str | None
    status: ReviewStatus
    created_at: datetime
    updated_at: datetime


class AuditEvent(BaseModel):
    id: int
    review_id: int
    event_type: str
    details: dict[str, object]
    created_at: datetime


class DashboardMetrics(BaseModel):
    total_reviews: int
    ready_to_draft: int
    needs_evidence: int
    drafted: int
    approved: int
    edited: int
    rejected: int
    acceptance_rate: float


def _database_path(path: Path | None = None) -> Path:
    return path or Path(os.getenv("SUPPORTOPS_DATABASE_PATH", DEFAULT_DATABASE_PATH))


def _connect(path: Path | None = None) -> sqlite3.Connection:
    database = _database_path(path)
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(SCHEMA)
    return connection


def _insert_audit(
    connection: sqlite3.Connection,
    review_id: int,
    event_type: str,
    details: dict[str, object],
) -> None:
    connection.execute(
        """
        INSERT INTO audit_events (review_id, event_type, details_json)
        VALUES (?, ?, ?)
        """,
        (review_id, event_type, json.dumps(details, sort_keys=True)),
    )


def _row_to_review(row: sqlite3.Row) -> ReviewRecord:
    return ReviewRecord(
        id=row["id"],
        ticket_text=row["ticket_text"],
        category=row["category"],
        urgency=row["urgency"],
        confidence=row["confidence"],
        method=row["method"],
        knowledge_matches=[
            KnowledgeMatch.model_validate(item)
            for item in json.loads(row["knowledge_json"])
        ],
        draft_text=row["draft_text"],
        draft_citations=json.loads(row["draft_citations_json"]),
        draft_model=row["draft_model"],
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def create_review(
    ticket: Ticket,
    prediction: Prediction,
    knowledge_matches: list[KnowledgeMatch],
    path: Path | None = None,
) -> ReviewRecord:
    status = (
        ReviewStatus.READY_TO_DRAFT
        if knowledge_matches
        else ReviewStatus.NEEDS_EVIDENCE
    )
    with _connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO reviews (
                ticket_text,
                category,
                urgency,
                confidence,
                method,
                knowledge_json,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket.text,
                prediction.category.value,
                prediction.urgency.value,
                prediction.confidence,
                prediction.method.value,
                json.dumps(
                    [match.model_dump(mode="json") for match in knowledge_matches]
                ),
                status.value,
            ),
        )
        review_id = cursor.lastrowid
        if review_id is None:
            raise RuntimeError("SQLite did not return a review ID.")
        _insert_audit(
            connection,
            review_id,
            "review_created",
            {
                "status": status.value,
                "knowledge_sources": [
                    match.source for match in knowledge_matches
                ],
            },
        )
    review = get_review(review_id, path)
    if review is None:
        raise RuntimeError("Created review could not be loaded.")
    return review


def get_review(review_id: int, path: Path | None = None) -> ReviewRecord | None:
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT * FROM reviews WHERE id = ?",
            (review_id,),
        ).fetchone()
    return _row_to_review(row) if row else None


def save_draft(
    review_id: int,
    text: str,
    citations: list[str],
    model: str,
    path: Path | None = None,
) -> ReviewRecord:
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT status FROM reviews WHERE id = ?",
            (review_id,),
        ).fetchone()
        if not row:
            raise KeyError(review_id)
        if row["status"] not in {
            ReviewStatus.READY_TO_DRAFT.value,
            ReviewStatus.DRAFTED.value,
        }:
            raise ValueError("This review cannot be drafted in its current status.")
        connection.execute(
            """
            UPDATE reviews
            SET draft_text = ?,
                draft_citations_json = ?,
                draft_model = ?,
                status = ?,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?
            """,
            (
                text,
                json.dumps(citations),
                model,
                ReviewStatus.DRAFTED.value,
                review_id,
            ),
        )
        _insert_audit(
            connection,
            review_id,
            "draft_created",
            {"model": model, "citations": citations, "draft": text},
        )
    review = get_review(review_id, path)
    if review is None:
        raise RuntimeError("Updated review could not be loaded.")
    return review


def record_draft_failure(
    review_id: int,
    error: str,
    path: Path | None = None,
) -> None:
    with _connect(path) as connection:
        if not connection.execute(
            "SELECT 1 FROM reviews WHERE id = ?",
            (review_id,),
        ).fetchone():
            raise KeyError(review_id)
        _insert_audit(connection, review_id, "draft_failed", {"error": error})


def record_decision(
    review_id: int,
    decision: ReviewDecision,
    edited_draft: str | None = None,
    note: str = "",
    path: Path | None = None,
) -> ReviewRecord:
    review = get_review(review_id, path)
    if review is None:
        raise KeyError(review_id)
    if not review.draft_text:
        raise ValueError("A review decision requires a draft.")

    if review.status is not ReviewStatus.DRAFTED:
        raise ValueError("This review already has a final decision.")

    draft_text = review.draft_text
    citations = review.draft_citations
    details: dict[str, object] = {"note": note.strip()}
    if decision is ReviewDecision.APPROVE:
        status = ReviewStatus.APPROVED
    elif decision is ReviewDecision.EDIT:
        if not edited_draft or len(edited_draft.strip()) < 3:
            raise ValueError("An edited draft must contain at least three characters.")
        draft_text = edited_draft.strip()
        citations = sorted(set(CITATION_PATTERN.findall(draft_text)))
        allowed = {match.source for match in review.knowledge_matches}
        if not citations:
            raise ValueError("An edited draft must cite a retrieved source.")
        if invalid := set(citations) - allowed:
            raise ValueError(
                f"Edited draft cites unsupported sources: {', '.join(sorted(invalid))}"
            )
        details.update(
            original_draft=review.draft_text,
            edited_draft=draft_text,
            citations=citations,
        )
        status = ReviewStatus.EDITED
    else:
        status = ReviewStatus.REJECTED

    with _connect(path) as connection:
        cursor = connection.execute(
            """
            UPDATE reviews
            SET draft_text = ?,
                draft_citations_json = ?,
                status = ?,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ? AND status = ?
            """,
            (
                draft_text,
                json.dumps(citations),
                status.value,
                review_id,
                ReviewStatus.DRAFTED.value,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError("This review already has a final decision.")
        _insert_audit(
            connection,
            review_id,
            {
                ReviewDecision.APPROVE: "draft_approved",
                ReviewDecision.EDIT: "draft_edited",
                ReviewDecision.REJECT: "draft_rejected",
            }[decision],
            details,
        )
    updated = get_review(review_id, path)
    if updated is None:
        raise RuntimeError("Decided review could not be loaded.")
    return updated


def list_audit_events(
    review_id: int,
    path: Path | None = None,
) -> list[AuditEvent]:
    with _connect(path) as connection:
        rows = connection.execute(
            """
            SELECT * FROM audit_events
            WHERE review_id = ?
            ORDER BY id DESC
            """,
            (review_id,),
        ).fetchall()
    return [
        AuditEvent(
            id=row["id"],
            review_id=row["review_id"],
            event_type=row["event_type"],
            details=json.loads(row["details_json"]),
            created_at=row["created_at"],
        )
        for row in rows
    ]


def dashboard_metrics(path: Path | None = None) -> DashboardMetrics:
    with _connect(path) as connection:
        rows = connection.execute(
            "SELECT status, COUNT(*) AS count FROM reviews GROUP BY status"
        ).fetchall()
    counts = {row["status"]: row["count"] for row in rows}
    approved = counts.get(ReviewStatus.APPROVED.value, 0)
    edited = counts.get(ReviewStatus.EDITED.value, 0)
    rejected = counts.get(ReviewStatus.REJECTED.value, 0)
    decisions = approved + edited + rejected
    return DashboardMetrics(
        total_reviews=sum(counts.values()),
        ready_to_draft=counts.get(ReviewStatus.READY_TO_DRAFT.value, 0),
        needs_evidence=counts.get(ReviewStatus.NEEDS_EVIDENCE.value, 0),
        drafted=counts.get(ReviewStatus.DRAFTED.value, 0),
        approved=approved,
        edited=edited,
        rejected=rejected,
        acceptance_rate=round((approved + edited) / decisions, 3)
        if decisions
        else 0,
    )


def list_recent_reviews(
    limit: int = 10,
    path: Path | None = None,
) -> list[ReviewRecord]:
    with _connect(path) as connection:
        rows = connection.execute(
            "SELECT * FROM reviews ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_review(row) for row in rows]
