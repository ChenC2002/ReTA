"""Command-line helpers for ReTA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from reta.knowledge.pool import KnowledgeTemplate, load_templates_jsonl, validate_template_pool


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
            outputs=["checkpoints/warmup.pt", "logs/warmup.log"],
            metrics=["mean BCE loss"],
        ),
        TrainingStage(
            name="stage2_reinforce_policy_refinement",
            objective="Optimize the Soft/Hard/Skip policy with paired rewards while continuing supervised encoder refinement.",
            inputs=["checkpoints/warmup.pt", "processed visit trajectories", "clustered knowledge templates"],
            outputs=["checkpoints/rl_iter*.pt", "logs/rl_train.log"],
            metrics=[
                "policy loss",
                "policy entropy",
                "mean return",
                "running baseline",
                "Soft/Hard/Skip action rates",
                "supervised BCE loss",
            ],
        ),
    ]


def build_external_entity_mapping(
    templates: Sequence[KnowledgeTemplate],
    base_mapping: Mapping[str, int],
) -> Dict[str, int]:
    """Allocate deterministic token IDs for template nodes outside ICD/CCS."""

    if any(isinstance(value, bool) or not isinstance(value, int) for value in base_mapping.values()):
        raise ValueError("base token mapping values must be integer token IDs")
    base_ids = [int(value) for value in base_mapping.values()]
    if set(base_ids) != set(range(len(base_mapping))):
        raise ValueError("base token mapping must use contiguous IDs starting at zero")

    external_entities = set()
    for template in templates:
        medoid = template.medoid
        entities = set(medoid.subgraph_nodes)
        entities.add(medoid.root_concept_id)
        for entity in entities:
            name = str(entity).strip()
            if name and name not in base_mapping:
                external_entities.add(name)

    start = len(base_mapping)
    return {
        entity: start + offset
        for offset, entity in enumerate(sorted(external_entities))
    }


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
        default=str(
            Path(__file__).resolve().parent / "knowledge" / "releases" / "pool_v1"
        ),
    )
    release.add_argument("--json", action="store_true", dest="as_json")

    entity_map = subparsers.add_parser("build-entity-map")
    entity_map.add_argument("--processed_path", required=True)
    entity_map.add_argument("--templates_jsonl", required=True)
    entity_map.add_argument("--out", required=True)

    subparsers.add_parser("show-training-stages")

    args = parser.parse_args(argv)
    if args.command == "validate-template-pool":
        return _validate_template_pool(args.templates_jsonl, args.expected_dim)
    if args.command == "validate-pool-release":
        return _validate_pool_release(args.release_dir, args.as_json)
    if args.command == "build-entity-map":
        return _build_entity_map(
            args.processed_path,
            args.templates_jsonl,
            args.out,
        )
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


def _validate_pool_release(path: str, as_json: bool) -> int:
    from reta.knowledge.release import verify_release

    report = verify_release(path)
    ok = report.ok()
    output = (
        json.dumps(report.to_dict(), indent=2, sort_keys=True)
        if as_json
        else report.format_text()
    )
    print(output, file=sys.stdout if ok else sys.stderr)
    return 0 if ok else 2


def _build_entity_map(processed_path: str, templates_path: str, out_path: str) -> int:
    from reta.learning.runtime import load_processed_data

    try:
        data = load_processed_data(processed_path)
        base_mapping = data.get("vocab", {}).get("name_to_token")
        if not isinstance(base_mapping, dict):
            raise ValueError("processed data is missing vocab.name_to_token")
        templates = load_templates_jsonl(templates_path)
        mapping = build_external_entity_mapping(templates, base_mapping)
        destination = Path(out_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"wrote {len(mapping)} external entity tokens to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
