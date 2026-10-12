"""Core scoring logic and schemas."""

from enum import StrEnum

from pydantic import BaseModel, Field


class F1Scores(BaseModel):
    """F1 scores for a metric."""

    precision: float
    recall: float
    f1: float


class SemanticMatchCounts(BaseModel):
    """Counts from comparing two lists of string items with semantic matching."""

    weighted_tp: float = Field(
        default=0, description="Weighted true positives (sum of similarity scores)"
    )
    fp: int = Field(
        default=0, description="False positives (extra items in generation)"
    )
    fn: int = Field(
        default=0, description="False negatives (missing items in generation)"
    )
    tp: int = Field(default=0, description="True positives (count of matched pairs)")

    def calculate_prf(self) -> F1Scores:
        """Calculate similarity-weighted precision, recall, and F1.

        Precision = weighted_tp / (matched_count + fp)
        Recall = weighted_tp / (matched_count + fn)
        F1 = harmonic mean of precision and recall

        This weights PRF by match quality, so partial matches contribute
        proportionally rather than giving full credit.
        """
        generated_count = self.tp + self.fp
        expected_count = self.tp + self.fn

        p = self.weighted_tp / generated_count if generated_count > 0 else 0.0
        r = self.weighted_tp / expected_count if expected_count > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0  # harmonic mean
        return F1Scores(
            precision=round(p, 4),
            recall=round(r, 4),
            f1=round(f1, 4),
        )


class SimilarityLevel(StrEnum):
    PERFECT = "perfect"
    EXCELLENT = "excellent"
    VERY_GOOD = "very_good"
    GOOD = "good"
    OKAY = "okay"
    BAD = "bad"


class MatchedPair(BaseModel):
    _SIMILARITY_SCORES: dict[SimilarityLevel, float] = {
        SimilarityLevel.PERFECT: 1.0,
        SimilarityLevel.EXCELLENT: 0.85,
        SimilarityLevel.VERY_GOOD: 0.65,
        SimilarityLevel.GOOD: 0.6,
        SimilarityLevel.OKAY: 0.2,
        SimilarityLevel.BAD: 0.0,
    }

    real: str
    generated: str
    similarity: SimilarityLevel = Field(
        ...,
        description=(
            "Semantic similarity level: "
            "perfect = essentially identical wording and meaning, "
            "excellent = same requirement with minor phrasing differences, "
            "very_good = same core requirement with some detail variation, "
            "good = captures main intent with moderate specificity differences, "
            "okay = partial overlap addressing same topic but missing aspects, "
            "bad = weak semantic overlap with substantially different requirements. "
            "Below bad is not a match; place in missing/extra instead."
        ),
    )

    @property
    def score(self) -> float:
        """Return the similarity score for this pair."""
        return self._SIMILARITY_SCORES[self.similarity]


class ReasoningAlignment(BaseModel):
    """Semantic alignment between two lists of reasons."""

    matched: list[MatchedPair] = Field(
        default_factory=list,
        description="Pairs of semantically related items with similarity scores",
    )
    missing_in_generation: list[str] = Field(
        default_factory=list,
        description="Real items with no match in the generation",
    )
    extra_in_generation: list[str] = Field(
        default_factory=list,
        description="Generated items with no match in the real decision",
    )

    @property
    def weighted_true_positives(self) -> float:
        """Sum of similarity scores (0-1 range)."""
        return round(sum(m.score for m in self.matched), 4)

    @property
    def false_positives(self) -> int:
        """Count of items in generation with no match in real."""
        return len(self.extra_in_generation)

    @property
    def false_negatives(self) -> int:
        """Count of items in real with no match in generation."""
        return len(self.missing_in_generation)

    @property
    def matched_count(self) -> int:
        """Number of matched pairs."""
        return len(self.matched)

    @property
    def similarity_scores(self) -> list[float]:
        """List of similarity scores for all matched pairs (0-1 range)."""
        return [m.score for m in self.matched]

    @property
    def mean_similarity(self) -> float | None:
        """Average similarity score, or None if no matches."""
        scores = self.similarity_scores
        return round(sum(scores) / len(scores), 1) if scores else None


class ReasoningAlignmentScores(SemanticMatchCounts, F1Scores):
    """All scores for reasoning alignment between expected and generated reasoning."""

    similarity_scores: list[float] = Field(default_factory=list)
    mean_similarity: float | None = Field(default=None)

    @classmethod
    def from_alignment(
        cls, alignment: "ReasoningAlignment"
    ) -> "ReasoningAlignmentScores":
        """Create scores from a ReasoningAlignment instance."""
        counts = SemanticMatchCounts(
            weighted_tp=alignment.weighted_true_positives,
            fp=alignment.false_positives,
            fn=alignment.false_negatives,
            tp=alignment.matched_count,
        )
        f1_scores = counts.calculate_prf()

        return cls(
            weighted_tp=counts.weighted_tp,
            fp=counts.fp,
            fn=counts.fn,
            tp=counts.tp,
            precision=f1_scores.precision,
            recall=f1_scores.recall,
            f1=f1_scores.f1,
            similarity_scores=alignment.similarity_scores,
            mean_similarity=alignment.mean_similarity,
        )

    @classmethod
    def from_counts(
        cls,
        counts: SemanticMatchCounts,
        similarity_scores: list[float],
        mean_similarity: float | None,
    ) -> "ReasoningAlignmentScores":
        """Create scores from SemanticMatchCounts."""
        f1_scores = counts.calculate_prf()

        return cls(
            weighted_tp=counts.weighted_tp,
            fp=counts.fp,
            fn=counts.fn,
            tp=counts.tp,
            precision=f1_scores.precision,
            recall=f1_scores.recall,
            f1=f1_scores.f1,
            similarity_scores=similarity_scores,
            mean_similarity=mean_similarity,
        )
