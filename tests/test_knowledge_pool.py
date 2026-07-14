"""Validation and retrieval tests for the consolidated knowledge pool."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from reta.knowledge.pool import (
    GroundedEntity,
    GroundedTemplate,
    KnowledgeTemplate,
    load_templates_jsonl,
    validate_grounded_template,
    validate_template_pool,
)


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def valid_template(template_id: int = 0, member_index: int = 0) -> KnowledgeTemplate:
    medoid = GroundedTemplate(
        root_concept_id="CCS:ROOT",
        root_name="Root",
        definition="Definition.",
        cascade_entities=[GroundedEntity("CCS:TARGET", "Target", "CCS", 1.0)],
        subgraph_nodes=["CCS:ROOT", "CCS:TARGET"],
        subgraph_edges=[("CCS:ROOT", "CCS:TARGET")],
    )
    return KnowledgeTemplate(template_id, [1.0, 0.0], member_index, medoid, [member_index])


class KnowledgeValidationTests(unittest.TestCase):
    def test_malformed_edges_and_nonfinite_scores_are_errors_not_crashes(self) -> None:
        medoid = valid_template().medoid
        medoid.subgraph_edges = [("CCS:ROOT", "CCS:TARGET", "EXTRA")]  # type: ignore[list-item]
        medoid.cascade_entities[0].score = float("nan")
        errors = validate_grounded_template(medoid)
        self.assertTrue(any("invalid edge" in error for error in errors))
        self.assertTrue(any("finite" in error for error in errors))

    def test_loader_rejects_nonstandard_nan_json(self) -> None:
        value = valid_template().to_dict()
        value["vector"][0] = float("nan")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "templates.jsonl"
            path.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON"):
                load_templates_jsonl(str(path))

    def test_loader_rejects_overflowed_json_float(self) -> None:
        payload = json.dumps(valid_template().to_dict()).replace("1.0", "1e999", 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "templates.jsonl"
            path.write_text(payload + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON number '1e999'"):
                load_templates_jsonl(str(path))

    def test_loader_rejects_duplicate_json_object_keys(self) -> None:
        payload = json.dumps(valid_template().to_dict()).replace(
            '"template_id": 0',
            '"template_id": 0, "template_id": 1',
            1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "templates.jsonl"
            path.write_text(payload + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key 'template_id'"):
                load_templates_jsonl(str(path))

    def test_pool_rejects_member_overlap(self) -> None:
        errors = validate_template_pool([valid_template(0, 0), valid_template(1, 0)], expected_dim=2)
        self.assertTrue(any("appears in templates" in error for error in errors))

    @unittest.skipUnless(HAS_NUMPY, "NumPy is not installed")
    def test_retrieval_ties_use_template_id(self) -> None:
        import numpy as np
        from reta.knowledge.pool import KnowledgePool

        templates = [valid_template(5, 0), valid_template(1, 1), valid_template(3, 2)]
        pool = KnowledgePool(templates)
        result = pool.retrieve_topk([7], lambda _token: np.array([1.0, 0.0]), K=2)
        self.assertEqual(result.template_ids, [1, 3])


if __name__ == "__main__":
    unittest.main()
