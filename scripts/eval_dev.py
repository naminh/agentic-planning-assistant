#!/usr/bin/env python3
"""Run the pipeline on the labelled eval cases and score it with the evaluator."""

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

import tracing  # noqa: E402
from config import load_settings  # noqa: E402
from decision import Decision  # noqa: E402
from evaluator.diagnostics import RobustEvaluator  # noqa: E402
from pipeline.run import Runner  # noqa: E402

DEV_CASES = [f"case-{n:03d}" for n in range(1, 6)]
DATASET = "doncaster-dev"
METRICS = ("decision_match", "precision", "recall", "f1")


def ensure_dataset(cases: list[str], decisions_dir: Path) -> None:
    """Create the LangSmith dataset of eval cases (reference = ground truth) if absent."""
    from langsmith import Client

    ls = Client()
    if not ls.has_dataset(dataset_name=DATASET):
        dataset = ls.create_dataset(DATASET, description="Doncaster eval cases 001-005")
        ls.create_examples(
            dataset_id=dataset.id,
            inputs=[{"case_id": c} for c in cases],
            outputs=[
                json.loads((decisions_dir / c / "decision.json").read_text("utf-8"))
                for c in cases
            ],
        )


async def run_experiment(
    one: Callable[[str], Awaitable[dict[str, Any]]], run: str | None
) -> None:
    """Run `one` over the dataset as a LangSmith experiment with the dev metrics."""
    from langsmith import Client
    from langsmith.evaluation import aevaluate

    async def target(inputs: dict[str, Any]) -> dict[str, Any]:
        return await one(inputs["case_id"])

    def metric(name: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def evaluate(outputs: dict[str, Any]) -> dict[str, Any]:
            return {"key": name, "score": outputs["scores"][name]}

        evaluate.__name__ = name
        return evaluate

    def cost_usd(outputs: dict[str, Any]) -> dict[str, Any]:
        return {"key": "cost_usd", "score": outputs["cost_usd"]}

    await aevaluate(
        target,
        data=DATASET,
        evaluators=[*(metric(m) for m in METRICS), cost_usd],
        experiment_prefix=run or "dev",
        max_concurrency=len(DEV_CASES),
        client=Client(),
    )


async def main(
    cases: list[str], cheap_all: bool, run: str | None, cache_only: bool
) -> None:
    """Assess each eval case, score against its reference decision, print a table."""
    settings = load_settings()
    if cheap_all:
        settings = settings.all_cheap()
    tracing.run_name.set(run)
    runner = await Runner.create(settings, cache_only=cache_only)
    evaluator = RobustEvaluator(settings.judge_model, runner.client.raw)

    async def one(case_id: str) -> dict[str, Any]:
        result = await runner.assess(case_id)
        result.save(settings, run=run)
        decision = result.decision
        reference = (
            settings.data_dir / "doncaster" / "decisions" / case_id / "decision.json"
        )
        expected = Decision.model_validate_json(reference.read_text(encoding="utf-8"))
        scores = await evaluator.ascore_item(case_id, expected, decision)
        return {
            "decision": decision.decision.value,
            "scores": scores,
            "cost_usd": result.cost_usd,
        }

    if tracing.enabled() and set(cases) == set(DEV_CASES):
        ensure_dataset(cases, settings.data_dir / "doncaster" / "decisions")
        await run_experiment(one, run)
    else:
        await asyncio.gather(*(one(c) for c in cases))
    tracing.flush()
    frame = evaluator.df_scores
    logger.info(
        "\n"
        + frame[["decision_match", "precision", "recall", "f1"]].round(2).to_string()
    )
    logger.info(f"Aggregate:\n{frame.mean(numeric_only=True).round(3).to_string()}")
    out = settings.outputs_dir / "eval_dev.json"
    out.write_text(json.dumps(frame.to_dict(orient="index"), indent=2, default=float))
    usage = runner.client.usage
    logger.info(f"cost ${usage.cost_usd:.3f} {usage.by_model}")
    pd.options.display.width = 120


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", default=DEV_CASES)
    parser.add_argument("--cheap-all", action="store_true")
    parser.add_argument("--cache-only", action="store_true", help="Never call the API")
    parser.add_argument("--run", help="Snapshot into outputs/runs/<run>/<case>/")
    args = parser.parse_args()
    asyncio.run(main(args.cases, args.cheap_all, args.run, args.cache_only))
