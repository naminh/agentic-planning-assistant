"""Evidence is judged by what a document is, not what its file is called."""

import asyncio

from domain import Issue
from llm import LLMClient
from pipeline.doctype import classify_documents, heuristic_kinds, kinds_of
from pipeline.ingest import CaseDocument
from pipeline.review import missing_evidence
from policy.config import load_config

CONFIG = load_config()
FRA_TEXT = "Flood Risk Assessment for 5 Test Road. The sequential test is passed."


class Scripted(LLMClient):
    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply

    async def complete(  # type: ignore[override]
        self, model, system, user, images=None, max_tokens=4000, stage="llm"
    ) -> str:
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_retitled_file_still_counts_by_its_content() -> None:
    doc = CaseDocument(name="doc_17", text=FRA_TEXT)
    assert "flood_risk_assessment" in heuristic_kinds(doc)


def test_file_name_alone_does_not_satisfy_a_requirement() -> None:
    doc = CaseDocument(name="flood_risk_assessment", text="Cover letter, nothing more.")
    assert kinds_of(doc) == {"other"}
    assert missing_evidence(Issue.FLOOD, CONFIG, [doc], _flood_site())


def test_passing_mention_deep_in_another_document_does_not_count() -> None:
    doc = CaseDocument(name="statement", text="x" * 2000 + " flood risk assessment")
    assert "flood_risk_assessment" not in heuristic_kinds(doc)


def test_drawing_description_counts_for_scanned_drawings() -> None:
    doc = CaseDocument(
        name="scan01", text="", drawing_description="Proposed rear elevation, 1:100"
    )
    assert "drawings" in heuristic_kinds(doc)


def test_model_labels_are_used_and_validated() -> None:
    docs = [CaseDocument(name="a", text=FRA_TEXT), CaseDocument(name="b", text="hi")]
    reply = '{"documents": [{"name": "a", "kinds": ["flood_risk_assessment"]}]}'
    out = asyncio.run(classify_documents(Scripted(reply), "m", docs))
    assert out[0].kinds == {"flood_risk_assessment"}
    assert out[1].kinds == {"other"}  # omitted by the model: heuristic stands in


def test_failed_call_falls_back_to_text_scan() -> None:
    docs = [CaseDocument(name="a", text=FRA_TEXT)]
    out = asyncio.run(classify_documents(Scripted(ValueError("bad json")), "m", docs))
    assert "flood_risk_assessment" in (out[0].kinds or set())


def _flood_site():
    from pipeline.site import lookup_constraints

    site = lookup_constraints("ZZ38 8VR").constraints
    assert site is not None
    return site
