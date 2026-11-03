import json

import httpx
import pytest

from drafting import (
    DraftingConfigurationError,
    DraftingError,
    LLMSettings,
    generate_grounded_draft,
)
from supportops import (
    Category,
    KnowledgeMatch,
    Method,
    Prediction,
    Ticket,
    Urgency,
)

TICKET = Ticket(text="My package is late and tracking has not changed")
PREDICTION = Prediction(
    category=Category.SHIPPING,
    urgency=Urgency.NORMAL,
    confidence=0.8,
    method=Method.ML,
)
KNOWLEDGE = [
    KnowledgeMatch(
        source="HH-SHIP-001",
        title="Shipping delays and tracking",
        content="Open a carrier investigation after five days without a scan.",
        score=0.4,
    )
]
SETTINGS = LLMSettings(
    base_url="http://localhost:11434/v1",
    model="qwen2.5:7b",
    timeout_seconds=5,
)


def test_grounded_draft_uses_openai_compatible_request_and_citations() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert payload["model"] == "qwen2.5:7b"
        assert "HH-SHIP-001" in payload["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                "I am sorry your delivery is delayed. We will "
                                "review the carrier scans [HH-SHIP-001]."
                            )
                        }
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = generate_grounded_draft(
            TICKET,
            PREDICTION,
            KNOWLEDGE,
            settings=SETTINGS,
            client=client,
        )

    assert result.citations == ["HH-SHIP-001"]
    assert result.model == "qwen2.5:7b"


def test_remote_request_sends_configured_bearer_token() -> None:
    settings = LLMSettings(
        base_url="https://example.test/v1",
        model="test-model",
        api_key="test-api-key",
        timeout_seconds=5,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-api-key"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "We will review the delay [HH-SHIP-001]."
                        }
                    }
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        generate_grounded_draft(
            TICKET,
            PREDICTION,
            KNOWLEDGE,
            settings=settings,
            client=client,
        )


def test_grounded_draft_rejects_missing_or_unsupported_citations() -> None:
    responses = iter(
        [
            "I am sorry your delivery is delayed.",
            "We will replace it immediately [HH-REF-999].",
        ]
    )

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": next(responses)}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DraftingError, match="did not include"):
            generate_grounded_draft(
                TICKET,
                PREDICTION,
                KNOWLEDGE,
                settings=SETTINGS,
                client=client,
            )
        with pytest.raises(DraftingError, match="unsupported sources"):
            generate_grounded_draft(
                TICKET,
                PREDICTION,
                KNOWLEDGE,
                settings=SETTINGS,
                client=client,
            )


def test_remote_endpoint_requires_api_key_but_local_endpoint_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUPPORTOPS_LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.delenv("SUPPORTOPS_LLM_API_KEY", raising=False)
    with pytest.raises(DraftingConfigurationError):
        LLMSettings.from_environment()

    monkeypatch.setenv("SUPPORTOPS_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("SUPPORTOPS_LLM_MODEL", "qwen2.5:7b")
    settings = LLMSettings.from_environment()

    assert settings.api_key is None
    assert settings.model == "qwen2.5:7b"
