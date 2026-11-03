from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from drafting import (
    DraftingConfigurationError,
    DraftingError,
    generate_grounded_draft,
)
from storage import (
    ReviewStatus,
    ReviewDecision,
    ReviewRecord,
    create_review,
    dashboard_metrics,
    get_review,
    list_audit_events,
    list_recent_reviews,
    record_decision,
    record_draft_failure,
    save_draft,
)
from supportops import Prediction, Ticket, classify_ticket, retrieve_knowledge

ROOT = Path(__file__).parent
app = FastAPI(title="SupportOps Copilot")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")


def _review_or_404(review_id: int) -> ReviewRecord:
    review = get_review(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail="Review not found")
    return review


def _render_review(
    request: Request,
    review: ReviewRecord,
    error: str | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="review.html",
        context={
            "review": review,
            "audit_events": list_audit_events(review.id),
            "error": error,
        },
        status_code=status_code,
    )


@app.get("/", response_class=HTMLResponse)
def review_form(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"ticket_text": "", "error": None},
    )


@app.post("/review")
def review_ticket(
    request: Request,
    ticket_text: str = Form(),
) -> Response:
    try:
        ticket = Ticket(text=ticket_text)
    except ValidationError:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "ticket_text": ticket_text,
                "error": "Enter at least three characters.",
            },
            status_code=422,
        )

    review = create_review(
        ticket,
        classify_ticket(ticket),
        retrieve_knowledge(ticket),
    )
    return RedirectResponse(f"/reviews/{review.id}", status_code=303)


@app.get("/reviews/{review_id}", response_class=HTMLResponse)
def review_detail(request: Request, review_id: int) -> HTMLResponse:
    return _render_review(request, _review_or_404(review_id))


@app.post("/reviews/{review_id}/draft", response_class=HTMLResponse)
def draft_review(
    request: Request,
    review_id: int,
) -> Response:
    review = _review_or_404(review_id)
    if not review.knowledge_matches:
        return _render_review(
            request,
            review,
            "A draft cannot be generated without relevant knowledge.",
            409,
        )

    if review.status not in {ReviewStatus.READY_TO_DRAFT, ReviewStatus.DRAFTED}:
        return _render_review(
            request,
            review,
            "A review with a final decision cannot be redrafted.",
            409,
        )

    ticket = Ticket(text=review.ticket_text)
    prediction = Prediction(
        category=review.category,
        urgency=review.urgency,
        confidence=review.confidence,
        method=review.method,
    )
    try:
        draft = generate_grounded_draft(
            ticket,
            prediction,
            review.knowledge_matches,
        )
    except DraftingConfigurationError as error:
        record_draft_failure(review.id, str(error))
        return _render_review(request, review, str(error), 503)
    except DraftingError as error:
        record_draft_failure(review.id, str(error))
        return _render_review(request, review, str(error), 502)

    save_draft(
        review.id,
        draft.text,
        draft.citations,
        draft.model,
    )
    return RedirectResponse(f"/reviews/{review.id}", status_code=303)


@app.post("/reviews/{review_id}/decision", response_class=HTMLResponse)
def decide_review(
    request: Request,
    review_id: int,
    action: str = Form(),
    edited_draft: str | None = Form(default=None),
    note: str = Form(default=""),
) -> Response:
    review = _review_or_404(review_id)
    try:
        decision = ReviewDecision(action)
        record_decision(
            review.id,
            decision,
            edited_draft=edited_draft,
            note=note,
        )
    except ValueError as error:
        return _render_review(request, review, str(error), 422)
    return RedirectResponse(f"/reviews/{review.id}", status_code=303)


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "metrics": dashboard_metrics(),
            "reviews": list_recent_reviews(),
        },
    )
