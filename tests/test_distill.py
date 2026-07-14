"""Tests for validated structured-response distillation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from reta.knowledge.distill import (
    build_artifacts,
    load_concepts_csv,
    load_responses_jsonl,
    validate_response,
)


class DistillationTests(unittest.TestCase):
    def test_builds_artifacts_from_ordered_responses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            concepts_path = root / "concepts.csv"
            responses_path = root / "responses.jsonl"
            concepts_path.write_text(
                "concept_id,concept_name,density\nCCS:49,Diabetes mellitus,moderate\n",
                encoding="utf-8",
            )
            responses_path.write_text(
                json.dumps(
                    {
                        "definition": "A metabolic disorder characterized by chronic hyperglycemia.",
                        "clinical_cascade": ["Diabetic retinopathy", "Peripheral neuropathy"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            artifacts = build_artifacts(
                load_concepts_csv(str(concepts_path)),
                load_responses_jsonl(str(responses_path)),
                model_family="GPT-4o",
            )

            self.assertEqual(len(artifacts), 1)
            self.assertEqual(artifacts[0].concept_id, "CCS:49")
            self.assertEqual(artifacts[0].cascade, ["Diabetic retinopathy", "Peripheral neuropathy"])
            self.assertEqual(artifacts[0].meta, {"model_family": "GPT-4o", "density": "moderate"})

    def test_rejects_concept_response_count_mismatch(self) -> None:
        concepts = [{"concept_id": "CCS:49", "concept_name": "Diabetes", "density": ""}]
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            build_artifacts(concepts, [], model_family="GPT-4o")

    def test_rejects_non_schema_response(self) -> None:
        with self.assertRaisesRegex(ValueError, "1-5 strings"):
            validate_response(
                {"definition": "Definition.", "clinical_cascade": []},
                "response record 1",
            )

    def test_rejects_duplicate_concepts_and_nonfinite_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            concepts_path = root / "concepts.csv"
            responses_path = root / "responses.jsonl"
            concepts_path.write_text(
                "concept_id,concept_name\nA,First\nA,Duplicate\n",
                encoding="utf-8",
            )
            responses_path.write_text(
                '{"definition":"Definition","clinical_cascade":[NaN]}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate concept ID"):
                load_concepts_csv(str(concepts_path))
            with self.assertRaisesRegex(ValueError, "non-finite JSON"):
                load_responses_jsonl(str(responses_path))

    def test_rejects_duplicate_response_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            responses_path = Path(tmp) / "responses.jsonl"
            responses_path.write_text(
                '{"definition":"First","definition":"Second","clinical_cascade":["Event"]}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key 'definition'"):
                load_responses_jsonl(str(responses_path))


if __name__ == "__main__":
    unittest.main()
