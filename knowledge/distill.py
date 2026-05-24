"""Offline distillation (Definition / adaptive Clinical Cascade).

Outputs bounded, parseable artifacts:
- Definition: one sentence focused on pathology.
- Clinical Cascade: 1-5 complications/comorbidities, with longer cascades for
  sparse KG neighborhoods and shorter cascades for dense neighborhoods.

Input: concepts.csv with columns [concept_id, concept_name] and optionally
[density] where density is one of sparse/moderate/dense.
Output: artifacts.jsonl (DistilledArtifact)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import pandas as pd

try:
    from .templates import DistilledArtifact
except ImportError:  # allow running as a script
    from templates import DistilledArtifact  # type: ignore


class BaseDistiller:
    def distill(self, concept_id: str, concept_name: str, density: Optional[str] = None) -> DistilledArtifact:
        raise NotImplementedError


class HeuristicDistiller(BaseDistiller):
    """Deterministic local distiller for reproducible offline artifacts."""

    def distill(self, concept_id: str, concept_name: str, density: Optional[str] = None) -> DistilledArtifact:
        density_norm = (density or "moderate").strip().lower()
        cascade_len = {"dense": 1, "moderate": 3, "sparse": 5}.get(density_norm, 3)
        definition = f"{concept_name}." if concept_name and not concept_name.endswith(".") else (concept_name or "")
        candidates = [
            "downstream complication",
            "related comorbidity",
            "organ dysfunction",
            "acute deterioration",
            "chronic sequela",
        ]
        cascade = candidates[:cascade_len]
        return DistilledArtifact(
            concept_id=str(concept_id),
            concept_name=str(concept_name),
            definition=definition,
            cascade=cascade,
            meta={"distiller": "heuristic", "density": density_norm, "cascade_len": cascade_len},
        )


def load_concepts_csv(path: str, id_col: str = "concept_id", name_col: str = "concept_name", density_col: str = "density") -> pd.DataFrame:
    df = pd.read_csv(path)
    if id_col not in df.columns or name_col not in df.columns:
        raise ValueError(f"concepts.csv must contain columns: {id_col}, {name_col}")
    cols = [id_col, name_col]
    if density_col in df.columns:
        cols.append(density_col)
    return df[cols].copy()


def save_artifacts_jsonl(items: List[DistilledArtifact], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline distillation of knowledge artifacts (Definition/adaptive Cascade).")
    p.add_argument("--concepts_csv", type=str, required=True)
    p.add_argument("--out_jsonl", type=str, required=True)
    p.add_argument("--id_col", type=str, default="concept_id")
    p.add_argument("--name_col", type=str, default="concept_name")
    p.add_argument("--density_col", type=str, default="density", help="Optional sparse/moderate/dense neighborhood column.")
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
    df = load_concepts_csv(args.concepts_csv, id_col=args.id_col, name_col=args.name_col, density_col=args.density_col)

    if args.backend == "heuristic":
        distiller = HeuristicDistiller()
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    artifacts: List[DistilledArtifact] = []
    densities = df[args.density_col].tolist() if args.density_col in df.columns else [None] * len(df)
    for concept_id, concept_name, density in zip(df[args.id_col].tolist(), df[args.name_col].tolist(), densities):
        artifacts.append(distiller.distill(concept_id, concept_name, density=density))

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    save_artifacts_jsonl(artifacts, args.out_jsonl)
    print(f"[distill] wrote {len(artifacts)} artifacts to {args.out_jsonl}")


if __name__ == "__main__":
    main()
