"""Cross-document checks that need no model: is the pack about one site?"""

import re

from pipeline.ingest import CaseDocument
from pipeline.models import CaseFacts, ReviewItem

POSTCODE_RE = re.compile(r"\bZZ\d{1,2}\s?\d[A-Z]{2}\b")


def _compact(postcode: str) -> str:
    return "".join(postcode.upper().split())


def _named_like_form(doc: CaseDocument) -> bool:
    name = doc.name.lower()
    return "application" in name and "form" in name


def application_forms(docs: list[CaseDocument]) -> list[CaseDocument]:
    """The application form(s) in the pack.

    When Jev classified the pack by content, a form is a document of kind
    `application_form` (a retitled form is still found, a mere mention is not). If it
    found none, or Jev did not run, the file name decides, as before.
    """
    by_content = [
        d for d in docs if d.kinds_by == "jev" and "application_form" in (d.kinds or ())
    ]
    return by_content or [d for d in docs if _named_like_form(d)]


def form_postcodes(docs: list[CaseDocument]) -> set[str]:
    """Postcodes (whitespace removed) printed on the application form(s)."""
    return {
        _compact(m.group(0))
        for doc in application_forms(docs)
        for m in POSTCODE_RE.finditer(doc.text.upper())
    }


def postcode_conflict(facts: CaseFacts, docs: list[CaseDocument]) -> str | None:
    """Describe a clash between the extracted postcode and the application form.

    The constraints lookup is keyed on the postcode, so a postcode the form does not
    contain (for example one lifted from a note or representation) must not silently
    decide which constraints apply. Returns None when they agree or the form shows none.
    """
    printed = form_postcodes(docs)
    if not facts.postcode or not printed:
        return None
    if _compact(facts.postcode) in printed:
        return None
    return (
        f"Extracted postcode {facts.postcode} does not appear on the application form "
        f"(form shows {', '.join(sorted(printed))})."
    )


def consistency_review(facts: CaseFacts, docs: list[CaseDocument]) -> list[ReviewItem]:
    """Review items for contradictions across the pack."""
    items = []
    conflict = postcode_conflict(facts, docs)
    if conflict:
        items.append(
            ReviewItem(
                kind="inconsistent_documents",
                detail=conflict + " Site constraints were treated as UNKNOWN.",
                blocking=True,
            )
        )
    return items
