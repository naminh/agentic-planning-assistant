# Planning decision evaluator

The evaluator module assesses planning decision accuracy and the semantic alignment of supporting reasons.

## Overview

`PlanningDecisionSimpleEvaluator` compares predicted and expected `Decision` objects. It scores:

- **Decision accuracy**: whether the predicted outcome matches the expected `approve` or `refuse` outcome
- **Reasoning alignment**: how closely the predicted reasons match the expected reasons

Planning conditions are not currently scored.

## Metrics

The evaluator returns:

- `decision_match`: `1` when the outcomes match, otherwise `0`
- `precision`: similarity-weighted precision for predicted reasons
- `recall`: similarity-weighted recall for expected reasons
- `f1`: harmonic mean of precision and recall
- `mean_similarity`: average similarity across matched reason pairs
- `weighted_tp`, `tp`, `fp`, and `fn`: weighted and unweighted match counts
- `similarity_scores`: individual scores for matched reason pairs

Results are cached by item ID and can be aggregated across all evaluated items.

## Usage

```python
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from decision import Decision, DecisionType
from evaluator.evaluator import PlanningDecisionSimpleEvaluator


async def evaluate() -> None:
    load_dotenv()
    evaluator = PlanningDecisionSimpleEvaluator(
        model="your-chosen-model",
        client=AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url=os.environ["OPENROUTER_BASE_URL"],
        ),
    )

    expected = Decision(
        decision=DecisionType.APPROVE,
        reasons=[
            "Revised drainage plans must be submitted.",
            "The building height must not exceed 12 metres.",
        ],
    )
    predicted = Decision(
        decision=DecisionType.APPROVE,
        reasons=[
            "The drainage plans require revision.",
            "The height limit is 12 metres.",
        ],
    )

    scores = await evaluator.ascore_item("item_001", expected, predicted)
    aggregate = evaluator.aggregate()

    print(scores)
    print(aggregate)
```

## Semantic matching

The evaluator asks an LLM to pair semantically related expected and predicted reasons and assign a similarity level to each pair. Missing expected reasons count as false negatives; additional predicted reasons count as false positives. Partial matches receive proportional credit.

## Model configuration

The evaluator does not hard-code a model. Pass the model name when constructing `PlanningDecisionSimpleEvaluator`.
