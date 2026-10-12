#!/usr/bin/env python3
"""Write a ranked, human-reviewable list of candidate policies per issue (offline)."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from loguru import logger  # noqa: E402

from config import load_settings  # noqa: E402
from policy.candidates import apply_candidates_file, candidate_report  # noqa: E402
from policy.config import config_dir, load_config  # noqa: E402
from policy.index import PolicyIndex  # noqa: E402


def main(top: int, out: Path, apply: bool) -> None:
    """Write the candidate report; with `apply`, also add the candidates to the recipes."""
    config = load_config()
    index = PolicyIndex.load(load_settings(require_key=False), config)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(candidate_report(index, config.recipes, top), encoding="utf-8")
    logger.info(f"Written to {out}")
    if apply:
        target = config_dir() / "issues.yaml"
        added = apply_candidates_file(target, index, top)
        logger.info(f"Added {len(added)} generated entries to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=8, help="Candidates shown per list")
    parser.add_argument(
        "--out", type=Path, default=REPO / "outputs" / "policy_candidates.md"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Add each issue's top-N candidates missing from its recipe to issues.yaml "
        "(flagged generated: true)",
    )
    args = parser.parse_args()
    main(args.top, args.out, args.apply)
