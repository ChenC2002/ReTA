"""Build distilled artifacts from structured clinical-model responses.

The response JSONL must contain one object per row in ``concepts.csv``, in the
same order. Each object follows the bundled ``llm_response.schema.json``
contract: a non-empty ``definition`` and a ``clinical_cascade`` of one to five
unique, non-empty strings.

This module deliberately performs no network requests and stores no API key.
Generate responses with the versioned prompts in the release directory, then
use this command to validate and serialize them reproducibly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .pool import DistilledArtifact, strict_json_loads


VALID_DENSITIES = {"sparse", "moderate", "dense"}
RESPONSE_FIELDS = {"definition", "clinical_cascade"}


def load_concepts_csv(
    path: str,
    id_col: str = "concept_id",
    name_col: str = "concept_name",
    density_col: str = "density",
) -> List[Dict[str, str]]:
    """Load and validate concept rows without changing their order."""
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = [name for name in (id_col, name_col) if name not in fields]
        if missing:
            raise ValueError(f"concepts CSV is missing required column(s): {', '.join(missing)}")

        concepts: List[Dict[str, str]] = []
        seen_ids = set()
        for line_no, row in enumerate(reader, start=2):
            concept_id = (row.get(id_col) or "").strip()
            concept_name = (row.get(name_col) or "").strip()
            if not concept_id or not concept_name:
                raise ValueError(f"{path}:{line_no}: concept ID and name must be non-empty")
            if concept_id in seen_ids:
                raise ValueError(f"{path}:{line_no}: duplicate concept ID {concept_id!r}")
            seen_ids.add(concept_id)

            density = (row.get(density_col) or "").strip().lower()
            if density and density not in VALID_DENSITIES:
                allowed = ", ".join(sorted(VALID_DENSITIES))
                raise ValueError(f"{path}:{line_no}: density must be one of {allowed}")
            concepts.append(
                {
                    "concept_id": concept_id,
                    "concept_name": concept_name,
                    "density": density,
                }
            )
    if not concepts:
        raise ValueError(f"{path}: concepts CSV is empty")
    return concepts


def load_responses_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load structured response objects while preserving line order."""
    responses: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = strict_json_loads(line)
            except (json.JSONDecodeError, ValueError) as exc:
                message = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
                raise ValueError(f"{path}:{line_no}: invalid JSON: {message}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: response must be a JSON object")
            responses.append(value)
    if not responses:
        raise ValueError(f"{path}: response JSONL is empty")
    return responses


def validate_response(response: Dict[str, Any], where: str) -> Tuple[str, List[str]]:
    """Validate one response against the release's structured-output contract."""
    unexpected = sorted(set(response) - RESPONSE_FIELDS)
    missing = sorted(RESPONSE_FIELDS - set(response))
    if missing:
        raise ValueError(f"{where}: missing field(s): {', '.join(missing)}")
    if unexpected:
        raise ValueError(f"{where}: unexpected field(s): {', '.join(unexpected)}")

    definition = response["definition"]
    if not isinstance(definition, str) or not definition.strip():
        raise ValueError(f"{where}: definition must be a non-empty string")

    cascade = response["clinical_cascade"]
    if not isinstance(cascade, list) or not 1 <= len(cascade) <= 5:
        raise ValueError(f"{where}: clinical_cascade must contain 1-5 strings")
    if any(not isinstance(item, str) or not item.strip() for item in cascade):
        raise ValueError(f"{where}: clinical_cascade contains an empty or non-string item")

    normalized = [item.strip() for item in cascade]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{where}: clinical_cascade items must be unique")
    return definition.strip(), normalized


def build_artifacts(
    concepts: Sequence[Dict[str, str]],
    responses: Sequence[Dict[str, Any]],
    model_family: str,
) -> List[DistilledArtifact]:
    """Pair concept rows with validated responses by stable row order."""
    if len(concepts) != len(responses):
        raise ValueError(
            f"concept/response count mismatch: {len(concepts)} concepts, "
            f"{len(responses)} responses"
        )
    if not concepts:
        raise ValueError("concept and response inputs must be non-empty")
    model_family = model_family.strip()
    if not model_family:
        raise ValueError("model_family must be non-empty")

    artifacts: List[DistilledArtifact] = []
    for index, (concept, response) in enumerate(zip(concepts, responses), start=1):
        definition, cascade = validate_response(response, f"response record {index}")
        meta: Dict[str, Any] = {"model_family": model_family}
        if concept.get("density"):
            meta["density"] = concept["density"]
        artifacts.append(
            DistilledArtifact(
                concept_id=concept["concept_id"],
                concept_name=concept["concept_name"],
                definition=definition,
                cascade=cascade,
                meta=meta,
            )
        )
    return artifacts


def save_artifacts_jsonl(items: Sequence[DistilledArtifact], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate structured clinical-model responses and build distilled ReTA artifacts."
    )
    parser.add_argument("--concepts_csv", required=True)
    parser.add_argument(
        "--responses_jsonl",
        required=True,
        help="One schema-valid response per concepts CSV row, in matching order.",
    )
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--id_col", default="concept_id")
    parser.add_argument("--name_col", default="concept_name")
    parser.add_argument("--density_col", default="density")
    parser.add_argument(
        "--model_family",
        required=True,
        help="Model family that generated the supplied responses (recorded as provenance).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    concepts = load_concepts_csv(
        args.concepts_csv,
        id_col=args.id_col,
        name_col=args.name_col,
        density_col=args.density_col,
    )
    responses = load_responses_jsonl(args.responses_jsonl)
    artifacts = build_artifacts(concepts, responses, model_family=args.model_family)
    save_artifacts_jsonl(artifacts, args.out_jsonl)
    print(f"[distill] wrote {len(artifacts)} validated artifacts to {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
