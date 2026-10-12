"""Offline tests for structure-aware chunking: policies, tables and split units."""

from domain import LOCAL_PLAN, NPPF, PolicyChunk
from policy import chunking
from policy.chunking import chunks_for_source
from policy.config import PolicySource
from policy.index import PolicyIndex
from policy.retrieval import _with_whole_units

LOCAL = PolicySource(path="lp.json", label=LOCAL_PLAN, version="2021")
NPPF_SOURCE = PolicySource(path="nppf.json", label=NPPF, version="2024")
SPD = PolicySource(path="spd.json", label="Some SPD", version="1")


def page(number: int, *texts: str, source_type: str = "narrative_text") -> dict:
    return {
        "page_number": number,
        "blocks": [
            {"block_type": "text", "source_type": source_type, "text": t} for t in texts
        ],
    }


def doc(*pages: dict) -> dict:
    return {"pages": list(pages)}


def lines(*rows: str) -> str:
    return "\r\n".join(rows)


POLICY_A = lines(
    "10 [go to Contents]",
    "",
    "Policy 28: Open Space Provision in New Developments",
    "",
    "To address local green space needs development proposals:",
    "A) of 20 dwellings or more will provide 10% of the site as on-site open space.",
    "B) of 10 to 20 dwellings will provide a commuted sum towards open space.",
)
POLICY_B = lines(
    "11 [go to Contents]",
    "C) for retirement living schemes will provide 16m2 of open space per person.",
    "D) all on-site open space must be managed and maintained.",
    "",
    "Explanation",
    "10.21. National policy states access to high quality open space is important.",
    "10.22. The policy applies to schemes of twenty family dwellings or more.",
)


def by_kind(chunks: list[PolicyChunk], kind: str) -> list[PolicyChunk]:
    return [c for c in chunks if c.kind == kind]


# --- policies --------------------------------------------------------------------


def test_policy_is_one_unit_across_a_page_break() -> None:
    chunks = chunks_for_source(LOCAL, doc(page(10, POLICY_A), page(11, POLICY_B)), {})
    (policy,) = by_kind(chunks, "policy")
    assert policy.policy_id == "Policy 28: Open Space Provision in New Developments"
    assert policy.page == 10 and policy.end_page == 11
    assert all(f"{x})" in policy.text for x in "ABCD")
    assert "10.21." not in policy.text  # supporting text is its own unit
    assert "[go to Contents]" not in policy.text  # running header removed


def test_supporting_text_keeps_its_policy_and_says_so() -> None:
    chunks = chunks_for_source(LOCAL, doc(page(10, POLICY_A), page(11, POLICY_B)), {})
    (support,) = by_kind(chunks, "supporting")
    assert support.policy_id == "Policy 28: Open Space Provision in New Developments"
    assert support.text.startswith(f"{support.policy_id} (supporting text)")
    assert "10.22." in support.text


def test_oversized_policy_splits_at_clause_boundaries_with_headings() -> None:
    clauses = [f"{chr(65 + i)}) " + ("word " * 60).strip() + "." for i in range(12)]
    text = lines("Policy 7: Big Policy", "", "Intro line:", *clauses)
    chunks = chunks_for_source(LOCAL, doc(page(1, text)), {})
    parts = by_kind(chunks, "policy")
    assert len(parts) > 1
    assert [p.part for p in parts] == list(range(1, len(parts) + 1))
    assert all(p.parts == len(parts) for p in parts)
    assert len({p.unit_id for p in parts}) == 1
    assert all(len(p.text) <= chunking.CHUNK_CHARS + 80 for p in parts)
    for p in parts[1:]:
        assert p.text.startswith("Policy 7: Big Policy (continued)")
        body = p.text.split("\n", 1)[1]
        assert body[:2] in {f"{chr(65 + i)})" for i in range(12)}  # starts on a clause
    joined = "\n".join(p.text for p in parts)
    assert all(c in joined for c in clauses)  # nothing lost


def test_cross_reference_is_not_mistaken_for_a_heading() -> None:
    text = lines(
        "Policy 44: Residential Design",
        "Homes must be adaptable and accessible in line with",
        "Policy 45. It must also be shown how housing for older people is increased.",
    )
    chunks = chunks_for_source(LOCAL, doc(page(1, text)), {})
    assert {c.policy_id for c in chunks} == {"Policy 44: Residential Design"}


def test_repeated_heading_in_a_monitoring_table_does_not_open_a_new_policy() -> None:
    text = lines(
        "Policy 28: Open Space Provision in New Developments",
        "A) provide open space.",
        "Chapter 20: Monitoring",
        "Policy 28: Open Space",
        "Provision in New",
    )
    chunks = chunks_for_source(LOCAL, doc(page(1, text)), {})
    assert [c.policy_id for c in chunks if c.kind == "policy"] == [
        "Policy 28: Open Space Provision in New Developments"
    ]
    assert all(c.policy_id is None for c in chunks if c.kind == "text")


def test_contents_pages_do_not_create_policies() -> None:
    contents = lines(
        *(f"Policy {n}: Title number {n} ........ {n + 90}" for n in range(1, 8))
    )
    real = lines(
        "Policy 3: Real Policy", "A) a real clause that is long enough to keep."
    )
    chunks = chunks_for_source(LOCAL, doc(page(2, contents), page(30, real)), {})
    assert [c.policy_id for c in chunks if c.kind == "policy"] == [
        "Policy 3: Real Policy"
    ]


# --- tables ----------------------------------------------------------------------


TABLE_P1 = lines(
    "Policy 5: Noise",
    "A) noise must be assessed.",
    "",
    "Table 1: Noise standards",
    "Outcome",
    "Daytime threshold",
    "Below 55dB",
    "LAeq,16hr",
)
TABLE_P2 = lines(
    "12 [go to Contents]",
    "Below 45dB",
    "LAeq,8hr",
    "5.7. Applicants should provide a noise assessment where thresholds are exceeded.",
)


def test_table_runs_from_caption_to_next_numbered_paragraph_across_pages() -> None:
    chunks = chunks_for_source(LOCAL, doc(page(11, TABLE_P1), page(12, TABLE_P2)), {})
    (table,) = by_kind(chunks, "table")
    assert table.heading == "Table 1: Noise standards"
    assert table.policy_id == "Policy 5: Noise"
    assert table.page == 11 and table.end_page == 12
    assert "Below 55dB" in table.text and "LAeq,8hr" in table.text
    assert "5.7." not in table.text  # the paragraph after the table is not in it
    after = [c for c in chunks if "5.7." in c.text]
    assert after and all(c.kind != "table" for c in after)


def test_table_ends_where_running_prose_begins() -> None:
    prose = [
        "This paragraph is ordinary running prose that continues for a long way across the page, so it is wide.",
        "A second prose line of similar width carries on explaining something quite unrelated to the table above.",
        "A third long line makes it clear that the table has finished and the narrative has resumed at last.",
    ]
    text = lines("Table 9: Species thresholds", "Species", "Bats", "30 animals", *prose)
    chunks = chunks_for_source(SPD, doc(page(1, text)), {})
    (table,) = by_kind(chunks, "table")
    assert "30 animals" in table.text and "ordinary running prose" not in table.text
    assert any("ordinary running prose" in c.text for c in by_kind(chunks, "text"))


def test_table_with_one_wide_cell_is_not_cut_short() -> None:
    wide = "Noise can be heard and causes small changes in behaviour and attitude, eg turning up volume"
    text = lines(
        "Table 2: Effects", "Level", "Observed", wide, "Between 55dB", "and 63dB"
    )
    chunks = chunks_for_source(SPD, doc(page(1, text)), {})
    (table,) = by_kind(chunks, "table")
    assert "and 63dB" in table.text


def test_table_is_split_by_lines_with_caption_repeated_only_when_too_big() -> None:
    rows = [f"Row {i} value {i * 3} metres" for i in range(500)]
    text = lines("Table 3: Big table", *rows)
    chunks = chunks_for_source(SPD, doc(page(1, text)), {})
    parts = by_kind(chunks, "table")
    assert len(parts) > 1
    for p in parts[1:]:
        assert p.text.startswith("Table 3: Big table (continued)")
    joined = "\n".join(p.text for p in parts)
    assert all(r in joined for r in rows[::50])  # rows survive intact


def test_scrambled_table_block_duplicating_prose_is_dropped() -> None:
    prose = (
        "5.41. It is likely that land will be required to be safeguarded for a school."
    )
    scrambled = "5.41. It is likely\tthat land will be required\tto be safeguarded\tfor a school."
    p = {
        "page_number": 3,
        "blocks": [
            {"block_type": "text", "source_type": "table", "text": scrambled},
            {"block_type": "text", "source_type": "narrative_text", "text": prose},
        ],
    }
    chunks = chunks_for_source(SPD, doc(p), {})
    assert "".join(c.text for c in chunks).count("safeguarded") == 1


# --- text hygiene and generic documents ------------------------------------------


def test_extractor_artefacts_are_cleaned() -> None:
    text = lines("12", "Policy 9: Sites", "Off\x02site provision  is allowed.")
    (chunk,) = chunks_for_source(LOCAL, doc(page(5, text)), {})
    assert "Off-site provision" in chunk.text and "\x02" not in chunk.text
    assert not chunk.text.startswith("12")  # bare page number dropped


def test_plain_text_continuations_overlap_and_merge_across_pages() -> None:
    para = " ".join(f"Sentence number {i} about housing design." for i in range(80))
    chunks = chunks_for_source(SPD, doc(page(1, para[:1500]), page(2, para[1500:])), {})
    parts = by_kind(chunks, "text")
    assert len(parts) >= 2 and parts[0].page == 1
    assert parts[1].text.startswith("(continued)\n…")  # carries context


# --- NPPF ------------------------------------------------------------------------


def test_nppf_paragraphs_get_ranges_and_ignore_contents_and_footnotes() -> None:
    contents = lines(*(f"{n}. Chapter title number {n} {n + 3}" for n in range(1, 8)))
    body = lines(
        "6. Planning policies and decisions should apply national policy consistently.",
        "7. The purpose of the planning system is to contribute to sustainable development.",
        "2. Achieving sustainable development",
        "8. Achieving sustainable development means three overarching objectives:",
        "49 Previously-developed land is defined in the glossary.",
        "9. Planning policies and decisions should:",
        "a) encourage multiple benefits from both urban and rural land;",
        "b) recognise that some undeveloped land can perform many functions.",
    )
    chunks = chunks_for_source(NPPF_SOURCE, doc(page(2, contents), page(36, body)), {})
    assert [(c.policy_id, c.heading) for c in chunks] == [
        ("paragraphs 6-7", None),
        ("paragraphs 8-9", "2. Achieving sustainable development"),
    ]
    assert (
        "49 Previously-developed" in chunks[1].text
    )  # footnote text is not a paragraph
    index = PolicyIndex(chunks)
    assert index.nppf_paragraph(9) and index.nppf_paragraph(6)


# --- retrieval over split units --------------------------------------------------


def test_policy_text_never_cuts_the_policy_itself() -> None:
    clauses = [f"{chr(65 + i)}) " + ("word " * 60).strip() + "." for i in range(12)]
    text = lines("Policy 7: Big Policy", "", *clauses, "7.1. Supporting paragraph one.")
    chunks = chunks_for_source(LOCAL, doc(page(1, text)), {})
    index = PolicyIndex(chunks)
    got = index.policy_text(LOCAL_PLAN, "Policy 7:", limit=100)
    assert [c.kind for c in got].count("policy") == len(by_kind(chunks, "policy")) > 1


def test_search_hit_on_one_table_part_brings_the_other_parts() -> None:
    rows = [f"Row {i} value {i * 3} metres" for i in range(500)]
    chunks = chunks_for_source(
        SPD, doc(page(1, lines("Table 3: Big table", *rows))), {}
    )
    index = PolicyIndex(chunks)
    parts = by_kind(chunks, "table")
    expanded = _with_whole_units(index, [parts[-1]])
    assert {c.part for c in expanded} == {p.part for p in parts}


# --- hybrid search ---------------------------------------------------------------


def hybrid_index() -> PolicyIndex:
    texts = [
        "Drainage of surface water must use sustainable drainage systems.",
        "Surface water drained to a watercourse requires consent.",
        "Hedgerows and trees should be retained where possible.",
        "Parking spaces must be provided for each dwelling.",
        "Residential design should respect local character.",
    ]
    return PolicyIndex(
        [PolicyChunk(source="S", page=i + 1, text=t) for i, t in enumerate(texts)]
    )


def test_hybrid_keeps_bm25_hits_first_and_adds_fuzzy_extras() -> None:
    index = hybrid_index()
    bm25 = index.search("drainage", k=1)
    assert [c.page for c in bm25] == [1]  # "drained" is a different word to BM25
    hybrid = index.hybrid_search(["drainage"], k=1)
    assert next(c.page for c in hybrid) == 1
    assert 2 in [c.page for c in hybrid]  # the fuzzy index finds "drained"


def test_hybrid_has_no_duplicates_and_respects_sources() -> None:
    index = hybrid_index()
    pages = [c.page for c in index.hybrid_search(["surface water drainage"], k=3)]
    assert len(pages) == len(set(pages))
    assert index.hybrid_search(["drainage"], k=3, sources={"Other"}) == []


def test_extra_queries_widen_the_search() -> None:
    index = hybrid_index()
    plain = {c.page for c in index.hybrid_search(["parking"], k=1)}
    widened = {c.page for c in index.hybrid_search(["parking", "hedgerows"], k=1)}
    assert plain < widened
