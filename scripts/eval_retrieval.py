#!/usr/bin/env python3
"""Offline retrieval evaluation on the eval cases. Makes no LLM or network calls.

Reports, per case: how many of the policies the officer cited reached the assessor
(retrieval recall) and how large the retrieved context was; the overlap between the
policies cited in our reasons and the officer's; and, with --variants, the same recall
when retrieval is replayed with the curated list and/or keyword search switched off.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from config import load_settings  # noqa: E402
from evaluator.diagnostics import (  # noqa: E402
    DEV_CASES,
    baseline_name,
    list_runs,
    load_reference,
    load_run,
    policy_seen,
)
from evaluator.retrieval import (  # noqa: E402
    VARIANTS,
    RetrievalScore,
    citations_of,
    officer_citations,
    reasons_overlap,
    replay_retrieval,
    score_retrieval,
)
from policy.config import load_config  # noqa: E402
from policy.index import PolicyIndex  # noqa: E402


def md_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub markdown table (no extra dependency)."""
    rows = frame.reset_index() if frame.index.name else frame
    header = "| " + " | ".join(str(c) for c in rows.columns) + " |"
    rule = "|" + "|".join("---" for _ in rows.columns) + "|"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in rows.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body])


def retrieval_frame(scores: list[RetrievalScore]) -> pd.DataFrame:
    """One row per (case, variant) with recall and context size."""
    rows = [
        {
            "case": s.case_id,
            "variant": s.variant,
            "cited": len(s.cited),
            "recall": round(s.recall, 2),
            "missed": sorted(s.missed),
            "chunks": s.chunks,
            "chars": s.chars,
        }
        for s in scores
    ]
    return pd.DataFrame(rows)


def main(run: str | None, cases: list[str], variants: bool, out: Path) -> None:
    """Build the retrieval, overlap and ablation tables and write them as markdown."""
    settings = load_settings(require_key=False)
    config = load_config()
    run = run or baseline_name(settings)
    if run not in list_runs(settings):
        raise SystemExit(f"No run {run!r}; available: {list_runs(settings)}")
    logger.info(f"Run: {run}   NPPF edition in repo: {config.nppf_edition}")

    scores: list[RetrievalScore] = []
    overlap_rows: list[dict[str, object]] = []
    nppf_rows: list[dict[str, object]] = []
    index = PolicyIndex.load(settings, config) if variants else None
    if index is not None:
        index.with_embeddings(settings)
    for case_id in cases:
        reference = load_reference(settings, case_id)
        saved = load_run(settings, case_id, run)
        if reference is None or saved.bundle is None:
            logger.warning(f"{case_id}: needs a reference decision and bundle.json")
            continue
        officer = officer_citations(reference)
        chars = (
            sum(
                len(policy_seen(saved, a.issue.value)) for a in saved.bundle.assessments
            )
            if saved.trace
            else None
        )
        as_run = score_retrieval(
            case_id, "as run", officer, citations_of(saved.bundle), chars
        )
        scores.append(as_run)
        overlap = reasons_overlap(reference, saved.decision)
        overlap_rows.append(
            {
                "case": case_id,
                "precision": round(overlap.precision, 2),
                "recall": round(overlap.recall, 2),
                "f1": round(overlap.f1, 2),
                "we_missed": sorted(overlap.missing),
                "we_added": sorted(overlap.extra),
            }
        )
        nppf_rows.append(
            {
                "case": case_id,
                "officer_cited": sorted(as_run.nppf_cited),
                "in_retrieved_ranges": sorted(as_run.nppf_covered),
            }
        )
        if index is not None:
            for name in VARIANTS:
                cites, size = replay_retrieval(index, config, saved.bundle, name)
                scores.append(score_retrieval(case_id, name, officer, cites, size))

    retrieval = retrieval_frame(scores)
    summary = (
        retrieval.groupby("variant", sort=False)[["recall", "chunks", "chars"]]
        .mean()
        .round(2)
    )
    overlap_frame = pd.DataFrame(overlap_rows)
    sections = [
        f"# Retrieval evaluation (run `{run}`)",
        "## Officer-cited Local Plan policies that reached the assessor\n"
        + md_table(retrieval),
        "## Mean by variant\n" + md_table(summary),
        "## Policies cited in our reasons vs the officer's (Local Plan)\n"
        + md_table(overlap_frame)
        + f"\n\nMean F1: {overlap_frame['f1'].mean():.2f}",
        f"## NPPF paragraphs (edition drift caveat: repo holds {config.nppf_edition}; "
        "reference decisions mix 2021 and 2023-24 numbering)\n"
        + md_table(pd.DataFrame(nppf_rows)),
    ]
    text = "\n\n".join(sections) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    logger.info("\n" + text)
    logger.info(f"Written to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", default=DEV_CASES)
    parser.add_argument(
        "--run", help="Run snapshot to read (default: current baseline)"
    )
    parser.add_argument(
        "--variants", action="store_true", help="Also replay retrieval ablations"
    )
    parser.add_argument(
        "--out", type=Path, default=REPO / "outputs" / "retrieval_eval.md"
    )
    args = parser.parse_args()
    main(args.run, args.cases, args.variants, args.out)
