from pathlib import Path

from supportops import Ticket, load_knowledge, retrieve_knowledge

ROOT = Path(__file__).parents[1]
KNOWLEDGE_DATASET = ROOT / "data" / "knowledge.csv"


def test_retrieval_returns_relevant_cited_knowledge() -> None:
    matches = retrieve_knowledge(
        Ticket(text="My package is late and tracking has not changed"),
        KNOWLEDGE_DATASET,
    )

    assert matches
    assert matches[0].source == "HH-SHIP-001"
    assert matches[0].score > 0


def test_retrieval_returns_no_unsupported_knowledge() -> None:
    matches = retrieve_knowledge(
        Ticket(text="Hello there"),
        KNOWLEDGE_DATASET,
    )

    assert matches == []


def test_knowledge_sources_are_fixed_and_unique() -> None:
    documents = load_knowledge(KNOWLEDGE_DATASET)

    assert len(documents) == 6
    assert len({document.source for document in documents}) == 6
