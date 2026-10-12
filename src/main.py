"""CLI: draft planning decisions for one or more application packs."""

import asyncio

import typer
from loguru import logger

import tracing
from config import Settings, load_settings
from pipeline.run import Runner

app = typer.Typer(
    name="assess",
    help="Augmented planning decision-making",
    add_completion=False,
    no_args_is_help=True,
)


async def run_cases(
    settings: Settings,
    cases: list[str],
    write_prediction: bool,
    exclude: frozenset[str],
    run: str | None = None,
    cache_only: bool = False,
) -> None:
    """Assess cases concurrently, save each, and log the cost summary.

    Args:
        settings: Models, paths and API configuration.
        cases: Case ids to assess.
        write_prediction: Also write `predictions/<case>-prediction.json`.
        exclude: Pack documents to leave out (ablation). When set, results are saved
            under a `<case>-ablation` label and no prediction is written.
        run: Name under which to keep a copy in `outputs/runs/`.
        cache_only: Replay from the LLM cache and fail on any miss (no API calls).
    """
    tracing.run_name.set(run)
    runner = await Runner.create(settings, cache_only=cache_only)

    async def one(case_id: str) -> None:
        result = await runner.assess(case_id, exclude)
        result.save(
            settings,
            run=run,
            prediction=write_prediction,
            label=f"{case_id}-ablation" if exclude else None,
        )

    await asyncio.gather(*(one(c) for c in cases))
    tracing.flush()
    u = runner.client.usage
    logger.info(
        f"{u.calls} calls ({u.cached} cached), {u.prompt_tokens} in / "
        f"{u.completion_tokens} out tokens, ${u.cost_usd:.3f} {u.by_model}"
    )
    if runner.jev:
        j = runner.jev.usage
        logger.info(
            f"Jev: {j.calls} calls ({j.cached} cached), {j.questions} questions, "
            f"${j.cost_usd:.5f} {sorted(j.models)}"
        )


@app.command()
def assess(
    cases: list[str] = typer.Argument(..., help="Case ids, e.g. case-006."),
    cheap_all: bool = typer.Option(
        False, help="Route every stage to the cheap model (plumbing dry run)."
    ),
    write_prediction: bool = typer.Option(
        False, help="Also write predictions/<case>-prediction.json."
    ),
    run: str | None = typer.Option(
        None, help="Also snapshot into outputs/runs/<run>/<case>/ for comparison."
    ),
    cache_only: bool = typer.Option(
        False, help="Never call the API; fail on any uncached call (free replay)."
    ),
    exclude: list[str] = typer.Option(
        [],
        help="Document names to leave out (ablation); output goes to outputs/<case>-ablation.",
    ),
) -> None:
    """Generate a draft decision and audit brief for each case."""
    settings = load_settings()
    if cheap_all:
        settings = settings.all_cheap()
    asyncio.run(
        run_cases(
            settings,
            cases,
            write_prediction and not exclude,
            frozenset(exclude),
            run,
            cache_only,
        )
    )


if __name__ == "__main__":
    app()
