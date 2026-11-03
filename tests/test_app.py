import asyncio

import pytest
from httpx import ASGITransport, AsyncClient, Response

import app as app_module
from drafting import DraftingConfigurationError, DraftResult


def request(method: str, path: str, **kwargs: object) -> Response:
    async def send() -> Response:
        async with AsyncClient(
            transport=ASGITransport(app=app_module.app),
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


@pytest.fixture(autouse=True)
def isolated_database(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUPPORTOPS_DATABASE_PATH", str(tmp_path / "supportops.db"))


def create_review(ticket_text: str) -> str:
    response = request("POST", "/review", data={"ticket_text": ticket_text})
    assert response.status_code == 303
    return response.headers["location"]


def test_review_form_and_validation_are_available() -> None:
    response = request("GET", "/")
    invalid = request("POST", "/review", data={"ticket_text": "Hi"})

    assert response.status_code == 200
    assert "Support ticket review" in response.text
    assert invalid.status_code == 422
    assert "Enter at least three characters." in invalid.text


def test_review_persists_routing_and_cited_knowledge() -> None:
    location = create_review("The glass top is shattered and unsafe")
    response = request("GET", location)

    assert response.status_code == 200
    assert "Damaged Item" in response.text
    assert "HH-DMG-001" in response.text
    assert "Generate grounded draft" in response.text
    assert "Review Created" in response.text


def test_review_without_evidence_blocks_drafting() -> None:
    location = create_review("Hello there")
    response = request("GET", location)
    blocked = request("POST", f"{location}/draft")

    assert "Needs Evidence" in response.text
    assert "No relevant knowledge found. Do not answer from memory." in response.text
    assert "Drafting is blocked until relevant evidence exists." in response.text
    assert blocked.status_code == 409


def test_draft_approval_and_dashboard_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app_module,
        "generate_grounded_draft",
        lambda *_: DraftResult(
            text="Please send photos of the damage [HH-DMG-001].",
            citations=["HH-DMG-001"],
            model="test-model",
        ),
    )
    location = create_review("The glass top is shattered and unsafe")
    drafted = request("POST", f"{location}/draft")
    detail = request("GET", location)
    approved = request(
        "POST",
        f"{location}/decision",
        data={
            "action": "approve",
            "edited_draft": "Please send photos of the damage [HH-DMG-001].",
            "note": "Policy checked.",
        },
    )
    dashboard = request("GET", "/dashboard")

    assert drafted.status_code == 303
    assert "Please send photos of the damage" in detail.text
    assert "HH-DMG-001" in detail.text
    assert approved.status_code == 303
    approved_detail = request("GET", location)
    assert "Approved" in approved_detail.text
    assert "Policy checked." in approved_detail.text
    assert "100.0%" in dashboard.text


def test_drafting_configuration_error_is_visible_and_audited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_):
        raise DraftingConfigurationError("Set SUPPORTOPS_LLM_API_KEY.")

    monkeypatch.setattr(app_module, "generate_grounded_draft", fail)
    location = create_review("The glass top is shattered and unsafe")
    response = request("POST", f"{location}/draft")

    assert response.status_code == 503
    assert "Set SUPPORTOPS_LLM_API_KEY." in response.text
    assert "Draft Failed" in response.text
