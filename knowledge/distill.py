"""
LLM offline distillation (Definition / Clinical Cascade).

Outputs bounded, parseable artifacts:
- Definition: one sentence.
- Clinical Cascade: exactly 3 items.

Input: concepts.csv with columns [concept_id, concept_name]
Output: artifacts.jsonl (DistilledArtifact)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import pandas as pd

try:
    from .templates import DistilledArtifact
except ImportError:  # allow running as a script
    from templates import DistilledArtifact  # type: ignore


class BaseDistiller:
    def distill(self, concept_id: str, concept_name: str) -> DistilledArtifact:
        raise NotImplementedError


class HeuristicDistiller(BaseDistiller):
    """Deterministic placeholder distiller."""

    def distill(self, concept_id: str, concept_name: str) -> DistilledArtifact:
        definition = f"{concept_name}." if concept_name and not concept_name.endswith(".") else (concept_name or "")
        cascade = ["complication", "comorbidity", "organ dysfunction"]  # exactly 3
        return DistilledArtifact(
            concept_id=str(concept_id),
            concept_name=str(concept_name),
            definition=definition,
            cascade=cascade,
            meta={"distiller": "heuristic"},
        )


def load_concepts_csv(path: str, id_col: str = "concept_id", name_col: str = "concept_name") -> pd.DataFrame:
    df = pd.read_csv(path)
    if id_col not in df.columns or name_col not in df.columns:
        raise ValueError(f"concepts.csv must contain columns: {id_col}, {name_col}")
    return df[[id_col, name_col]].copy()


def save_artifacts_jsonl(items: List[DistilledArtifact], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline distillation of knowledge artifacts (Definition/Cascade).")
    p.add_argument("--concepts_csv", type=str, required=True)
    p.add_argument("--out_jsonl", type=str, required=True)
    p.add_argument("--id_col", type=str, default="concept_id")
    p.add_argument("--name_col", type=str, default="concept_name")
    p.add_argument(
        "--backend",
        type=str,
        default="heuristic",
        choices=["heuristic"],
        help="Distillation backend (replace with your LLM backend in your environment).",
    )
    return p


def main():
    args = build_argparser().parse_args()
    df = load_concepts_csv(args.concepts_csv, id_col=args.id_col, name_col=args.name_col)

    if args.backend == "heuristic":
        distiller = HeuristicDistiller()
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    artifacts: List[DistilledArtifact] = []
    for concept_id, concept_name in zip(df[args.id_col].tolist(), df[args.name_col].tolist()):
        artifacts.append(distiller.distill(concept_id, concept_name))

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    save_artifacts_jsonl(artifacts, args.out_jsonl)
    print(f"[distill] wrote {len(artifacts)} artifacts to {args.out_jsonl}")


if __name__ == "__main__":
    main()
