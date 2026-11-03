from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from supportops import Ticket, classify_ticket, retrieve_knowledge

app = FastAPI(title="SupportOps Copilot")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@app.get("/", response_class=HTMLResponse)
def review_form(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "ticket_text": "",
            "prediction": None,
            "knowledge_matches": [],
            "error": None,
        },
    )


@app.post("/review", response_class=HTMLResponse)
def review_ticket(request: Request, ticket_text: str = Form()) -> HTMLResponse:
    try:
        ticket = Ticket(text=ticket_text)
    except ValidationError:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "ticket_text": ticket_text,
                "prediction": None,
                "knowledge_matches": [],
                "error": "Enter at least three characters.",
            },
            status_code=422,
        )

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "ticket_text": ticket.text,
            "prediction": classify_ticket(ticket),
            "knowledge_matches": retrieve_knowledge(ticket),
            "error": None,
        },
    )
