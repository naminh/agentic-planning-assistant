import asyncio

import pandas as pd
from openai import AsyncOpenAI

from decision import Decision
from evaluator.prompt import JUDGEMENT_PROMPT_TEMPLATE
from evaluator.scores import (
    ReasoningAlignment,
    ReasoningAlignmentScores,
    SemanticMatchCounts,
)


class PlanningDecisionSimpleEvaluator:
    """
    Evaluate planning decision extraction and reasoning alignment.

    This evaluator compares the extracted decision and reasoning from a planning document
    against the expected decision and reasoning.
    """

    def __init__(self, model: str, client: AsyncOpenAI | None):
        """
        Initialize the evaluator.

        Args:
            model: Model name for semantic judging.
            client: OpenAI client for making API calls.
        """
        self._item_scores: dict[str, dict[str, float | int]] = {}
        self._lock = asyncio.Lock()
        self.model = model
        self.client = client

    @property
    def df_scores(self) -> pd.DataFrame:
        """Return cached scores as a pandas DataFrame."""
        return pd.DataFrame.from_dict(self._item_scores, orient="index")

    async def _cache_result(self, item_id: str, result: dict[str, float | int]) -> None:
        """Cache a result in a thread-safe manner.

        Args:
            item_id: ID of the item.
            result: Dictionary containing the scores to cache.
        """
        async with self._lock:
            self._item_scores[item_id] = result

    async def _judge_reasoning_alignment(
        self, expected: list[str], generated: list[str]
    ) -> ReasoningAlignment:
        """Judge the alignment between expected and generated reasoning for a decision.

        Args:
            expected: List of expected decision reasoning items.
            generated: List of generated decision reasoning items.

        Returns:
            ReasoningAlignment object containing the alignment metrics.
        """
        if self.client is None:
            raise ValueError(
                "An OpenAI client is required to judge reasoning alignment"
            )

        resp = await self.client.responses.parse(
            model=self.model,
            instructions="You compare planning reasons semantically.",
            input=JUDGEMENT_PROMPT_TEMPLATE.render(
                real=expected,
                generated=generated,
            ),
            text_format=ReasoningAlignment,
        )
        if resp.output_parsed is None:
            raise ValueError("Failed to parse reasons alignment from LLM response")
        return resp.output_parsed

    async def _calculate_reasoning_scores(
        self, expected: list[str], generated: list[str]
    ) -> ReasoningAlignmentScores:
        """Calculate reasoning alignment scores, handling edge cases.

        Args:
            expected: List of expected decision reasoning items.
            generated: List of generated decision reasoning items.

        Returns:
            ReasoningAlignmentScores object containing the calculated scores.
        """
        if len(expected) > 0 and len(generated) > 0:
            alignment = await self._judge_reasoning_alignment(expected, generated)
            return ReasoningAlignmentScores.from_alignment(alignment)

        # Handle edge cases where one or both lists are empty
        if not expected and not generated:
            counts = SemanticMatchCounts(weighted_tp=0.0, fp=0, fn=0, tp=0)
        elif not generated:
            counts = SemanticMatchCounts(weighted_tp=0.0, fp=0, fn=len(expected), tp=0)
        else:  # not expected
            counts = SemanticMatchCounts(weighted_tp=0.0, fp=len(generated), fn=0, tp=0)

        return ReasoningAlignmentScores.from_counts(
            counts=counts,
            similarity_scores=[],
            mean_similarity=None,
        )

    async def ascore_item(
        self, item_id: str, expected: Decision, predicted: Decision
    ) -> dict[str, float | int]:
        """Score a single item and cache the result.

        Args:
            item_id: ID of the item to score.
            expected: Expected Decision object.
            predicted: Predicted Decision object.

        Returns:
            Dictionary containing the scores for the item.
        """
        reasoning_scores = await self._calculate_reasoning_scores(
            expected.reasons, predicted.reasons
        )

        result = {
            "decision_match": int(expected.decision == predicted.decision),
            **reasoning_scores.model_dump(),
        }

        await self._cache_result(item_id, result)
        return result

    def aggregate(self) -> dict[str, float]:
        """Aggregate scores across all items.

        Returns:
            Dictionary containing the aggregated scores across the dataset.
        """
        result: dict[str, float] = self.df_scores.mean(numeric_only=True).to_dict()
        return result
