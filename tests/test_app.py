import asyncio

from httpx import ASGITransport, AsyncClient, Response

from app import app


def request(method: str, path: str, **kwargs: object) -> Response:
    async def send() -> Response:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_review_form_is_available() -> None:
    response = request("GET", "/")

    assert response.status_code == 200
    assert "Support ticket review" in response.text
    assert 'action="/review"' in response.text


def test_review_page_shows_the_classifier_result() -> None:
    response = request(
        "POST",
        "/review",
        data={"ticket_text": "The glass top is shattered and unsafe"},
    )

    assert response.status_code == 200
    assert "Damaged Item" in response.text
    assert "Urgent" in response.text
    assert "Review the suggested route before taking action." in response.text
    assert "Relevant HarborHome knowledge" in response.text
    assert "HH-DMG-001" in response.text


def test_unknown_prediction_requires_manual_routing() -> None:
    response = request("POST", "/review", data={"ticket_text": "Hello there"})

    assert response.status_code == 200
    assert "Unknown" in response.text
    assert "Needs manual routing" in response.text
    assert "No relevant knowledge found. Do not answer from memory." in response.text


def test_review_page_rejects_invalid_ticket_text() -> None:
    response = request("POST", "/review", data={"ticket_text": "Hi"})

    assert response.status_code == 422
    assert "Enter at least three characters." in response.text
