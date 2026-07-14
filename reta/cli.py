"""Command-line helpers for ReTA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from dataclasses import asdict, dataclass
from typing import List, Optional

from reta.knowledge.templates import load_templates_jsonl, validate_template_pool


@dataclass
class TrainingStage:
    name: str
    objective: str
    inputs: List[str]
    outputs: List[str]
    metrics: List[str]


def default_training_stages() -> List[TrainingStage]:
    return [
        TrainingStage(
            name="stage1_encoder_warmup",
            objective="Walk patient trajectories in order while training the decoupled encoder and predictor with Bernoulli exposure to stochastic Soft/Hard imports.",
            inputs=["processed visit trajectories", "clustered knowledge templates"],
            outputs=["checkpoints/warmup.pt"],
            metrics=["BCE loss", "AUPRC", "Micro-F1", "Acc@20"],
        ),
        TrainingStage(
            name="stage2_reinforce_policy_refinement",
            objective="Optimize the Soft/Hard/Skip policy with paired rewards while continuing supervised encoder refinement.",
            inputs=["checkpoints/warmup.pt", "processed visit trajectories", "clustered knowledge templates"],
            outputs=["checkpoints/rl_iter*.pt"],
            metrics=["paired reward", "policy entropy", "skip rate", "AUPRC", "Micro-F1", "Acc@20"],
        ),
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="reta")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-template-pool")
    validate.add_argument("--templates_jsonl", required=True)
    validate.add_argument("--expected_dim", type=int, default=None)

    release = subparsers.add_parser("validate-pool-release")
    release.add_argument(
        "--release-dir",
        "--release_dir",
        default=str(Path(__file__).resolve().parents[1] / "artifacts" / "pool_v1"),
    )
    release.add_argument("--allow-incomplete", "--allow_incomplete", action="store_true")
    release.add_argument("--json", action="store_true", dest="as_json")

    subparsers.add_parser("show-training-stages")

    args = parser.parse_args(argv)
    if args.command == "validate-template-pool":
        return _validate_template_pool(args.templates_jsonl, args.expected_dim)
    if args.command == "validate-pool-release":
        return _validate_pool_release(args.release_dir, args.allow_incomplete, args.as_json)
    if args.command == "show-training-stages":
        print(json.dumps([asdict(stage) for stage in default_training_stages()], indent=2))
        return 0
    return 1


def _validate_template_pool(path: str, expected_dim: Optional[int]) -> int:
    try:
        templates = load_templates_jsonl(path)
        errors = validate_template_pool(templates, expected_dim=expected_dim)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 2
    print(f"valid template pool: {len(templates)} templates")
    return 0


def _validate_pool_release(path: str, allow_incomplete: bool, as_json: bool) -> int:
    from artifacts.pool_v1.verify_release import verify_release

    report = verify_release(path)
    ok = report.ok(allow_incomplete)
    output = (
        json.dumps(report.to_dict(allow_incomplete), indent=2, sort_keys=True)
        if as_json
        else report.format_text(allow_incomplete)
    )
    print(output, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
