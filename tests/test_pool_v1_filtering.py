"""Focused tests for the frozen pool_v1 grounding/filtering contract."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from artifacts.pool_v1.filtering import (
    EmbeddingCandidate,
    EmbeddingCandidateIndex,
    Entity,
    Inventory,
    SupportIndex,
    filter_artifact_objects,
    main,
)


def artifact(cascade, **updates):
    value = {
        "concept_id": "ROOT",
        "concept_name": "Root disease",
        "definition": "A pathology-focused definition.",
        "cascade": list(cascade),
        "meta": {},
    }
    value.update(updates)
    return value


class PoolV1FilteringTests(unittest.TestCase):
    def setUp(self):
        self.inventory = Inventory(
            [
                Entity("ROOT", "Root disease", "ICD-10"),
                Entity("TARGET", "Target complication", "PrimeKG"),
            ]
        )

    def test_schema_failure_is_first_and_prevents_mapping(self):
        malformed = artifact(["Target complication"])
        malformed["cascade"] = "Target complication"  # cascade must be an array
        malformed["unexpected"] = True
        result = filter_artifact_objects(
            [malformed],
            self.inventory,
            SupportIndex(primekg_edges=[("ROOT", "TARGET", "causes")]),
        )

        self.assertEqual(result.grounded, [])
        self.assertEqual(len(result.audit), 1)
        self.assertEqual(result.audit[0]["first_failure"], "format")
        self.assertFalse(result.audit[0]["mapping"]["attempted"])
        errors = result.audit[0]["format_evidence"]["errors"]
        self.assertIn({"field": "cascade", "reason": "array_required"}, errors)
        self.assertIn({"field": "unexpected", "reason": "additional_property_not_allowed"}, errors)
        self.assertEqual(result.summary["first_failure_counts"]["format_violation"], 1)

    def test_exact_match_precedes_embedding_and_uses_stable_source_priority(self):
        inventory = Inventory(
            [
                Entity("ROOT", "Root disease", "ICD-10"),
                Entity("PKG", "Shared name", "PrimeKG"),
                Entity("CCS", "Shared name", "CCS"),
            ]
        )
        candidates = EmbeddingCandidateIndex([EmbeddingCandidate("Shared name", "PKG", 0.99)])
        result = filter_artifact_objects(
            [artifact(["Shared name"])],
            inventory,
            SupportIndex(primekg_edges=[("ROOT", "CCS", None)]),
            candidates,
        )

        self.assertEqual(result.grounded[0]["cascade_entities"][0]["entity_id"], "CCS")
        mapping = result.audit[0]["mapping"]
        self.assertEqual(mapping["method"], "exact")
        self.assertEqual(mapping["evidence"]["exact_candidate_ids"], ["CCS", "PKG"])
        self.assertFalse(mapping["evidence"]["embedding_candidate_consulted"])

    def test_embedding_threshold_is_strictly_greater_than_point_nine(self):
        equal = EmbeddingCandidateIndex([EmbeddingCandidate("Approximate", "TARGET", 0.90)])
        result_equal = filter_artifact_objects(
            [artifact(["Approximate"])],
            self.inventory,
            SupportIndex(primekg_edges=[("ROOT", "TARGET", None)]),
            equal,
        )
        self.assertEqual(result_equal.grounded, [])
        self.assertEqual(result_equal.audit[0]["first_failure"], "mapping")
        self.assertEqual(result_equal.audit[0]["reason"], "embedding_score_not_strictly_above_threshold")
        self.assertFalse(result_equal.audit[0]["support"]["attempted"])

        above = EmbeddingCandidateIndex([EmbeddingCandidate("Approximate", "TARGET", 0.900001)])
        result_above = filter_artifact_objects(
            [artifact(["Approximate"])],
            self.inventory,
            SupportIndex(primekg_edges=[("ROOT", "TARGET", None)]),
            above,
        )
        self.assertEqual(len(result_above.grounded), 1)
        self.assertEqual(result_above.audit[0]["mapping"]["method"], "precomputed_top1")
        self.assertEqual(result_above.audit[0]["status"], "accepted")

    def test_missing_precomputed_entity_fails_closed_before_support(self):
        candidates = EmbeddingCandidateIndex([EmbeddingCandidate("Approximate", "MISSING", 0.99)])
        result = filter_artifact_objects(
            [artifact(["Approximate"])],
            self.inventory,
            SupportIndex(primekg_edges=[("ROOT", "MISSING", None)]),
            candidates,
        )
        self.assertEqual(result.grounded, [])
        self.assertEqual(result.audit[0]["reason"], "embedding_candidate_entity_not_in_inventory")
        self.assertFalse(result.audit[0]["support"]["attempted"])

    def test_direct_primekg_edges_are_deduplicated_and_disconnected_edges_are_absent(self):
        support = SupportIndex(
            primekg_edges=[
                ("TARGET", "ROOT", "complication_of"),
                ("ROOT", "TARGET", "causes"),
                ("X", "Y", "unrelated"),
                ("ROOT", "ROOT", "self"),
            ]
        )
        result = filter_artifact_objects([artifact(["Target complication"])], self.inventory, support)

        grounded = result.grounded[0]
        self.assertEqual(grounded["subgraph_nodes"], ["ROOT", "TARGET"])
        self.assertEqual(grounded["subgraph_edges"], [["ROOT", "TARGET"]])
        self.assertEqual(result.audit[0]["support"]["kind"], "primekg_direct")
        self.assertEqual(result.audit[0]["support"]["relations"], ["causes", "complication_of"])

    def test_two_hop_primekg_path_does_not_count_as_direct_support(self):
        result = filter_artifact_objects(
            [artifact(["Target complication"])],
            self.inventory,
            SupportIndex(
                primekg_edges=[
                    ("ROOT", "INTERMEDIATE", None),
                    ("INTERMEDIATE", "TARGET", None),
                ]
            ),
        )

        self.assertEqual(result.grounded, [])
        self.assertEqual(result.audit[0]["first_failure"], "support")
        self.assertEqual(result.audit[0]["reason"], "no_direct_primekg_or_ccs_ancestor_descendant_support")

    def test_ccs_two_hop_path_preserves_intermediate_node_and_actual_edges(self):
        inventory = Inventory(
            [
                Entity("ROOT", "Root disease", "ICD-10"),
                Entity("TARGET", "Target complication", "CCS"),
                Entity("MID", "Intermediate category", "CCS"),
            ]
        )
        support = SupportIndex(ccs_edges=[("ROOT", "MID"), ("MID", "TARGET")])
        result = filter_artifact_objects([artifact(["Target complication"])], inventory, support)

        grounded = result.grounded[0]
        self.assertEqual(grounded["subgraph_nodes"], ["ROOT", "MID", "TARGET"])
        self.assertEqual(grounded["subgraph_edges"], [["MID", "ROOT"], ["MID", "TARGET"]])
        evidence = result.audit[0]["support"]
        self.assertEqual(evidence["kind"], "ccs_hierarchy")
        self.assertEqual(evidence["direction"], "root_is_ancestor")
        self.assertEqual(evidence["hops"], 2)
        self.assertEqual(evidence["path_nodes"], ["ROOT", "MID", "TARGET"])

    def test_ccs_descendant_path_is_allowed_but_sibling_and_three_hop_paths_are_not(self):
        descendant_inventory = Inventory(
            [
                Entity("ROOT", "Root disease", "CCS"),
                Entity("ANCESTOR", "Ancestor", "CCS"),
                Entity("MID", "Intermediate", "CCS"),
            ]
        )
        descendant = filter_artifact_objects(
            [artifact(["Ancestor"])],
            descendant_inventory,
            SupportIndex(ccs_edges=[("ANCESTOR", "MID"), ("MID", "ROOT")]),
        )
        self.assertEqual(descendant.audit[0]["support"]["direction"], "root_is_descendant")
        self.assertEqual(descendant.audit[0]["support"]["path_nodes"], ["ROOT", "MID", "ANCESTOR"])

        sibling_inventory = Inventory(
            [
                Entity("ROOT", "Root disease", "CCS"),
                Entity("TARGET", "Target complication", "CCS"),
                Entity("PARENT", "Parent", "CCS"),
            ]
        )
        sibling = filter_artifact_objects(
            [artifact(["Target complication"])],
            sibling_inventory,
            SupportIndex(ccs_edges=[("PARENT", "ROOT"), ("PARENT", "TARGET")]),
        )
        self.assertEqual(sibling.grounded, [])
        self.assertEqual(sibling.audit[0]["first_failure"], "support")

        long_inventory = Inventory(
            [
                Entity("ROOT", "Root disease", "CCS"),
                Entity("TARGET", "Target complication", "CCS"),
            ]
        )
        too_long = filter_artifact_objects(
            [artifact(["Target complication"])],
            long_inventory,
            SupportIndex(ccs_edges=[("ROOT", "A"), ("A", "B"), ("B", "TARGET")]),
        )
        self.assertEqual(too_long.grounded, [])
        self.assertEqual(too_long.audit[0]["reason"], "no_direct_primekg_or_ccs_ancestor_descendant_support")

    def test_output_deduplicates_entities_and_never_emits_self_loops(self):
        result = filter_artifact_objects(
            [artifact(["Target complication", " target complication ", "Root disease"])],
            self.inventory,
            SupportIndex(primekg_edges=[("ROOT", "TARGET", None), ("ROOT", "ROOT", None)]),
        )

        grounded = result.grounded[0]
        self.assertEqual([entity["entity_id"] for entity in grounded["cascade_entities"]], ["TARGET"])
        self.assertEqual(grounded["subgraph_nodes"], ["ROOT", "TARGET"])
        self.assertEqual(grounded["subgraph_edges"], [["ROOT", "TARGET"]])
        self.assertEqual([item["status"] for item in result.audit], ["accepted", "accepted", "rejected"])
        self.assertEqual(result.audit[-1]["reason"], "self_reference_or_missing_endpoint")

    def test_cli_writes_grounded_audit_and_summary_with_invalid_json_first_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts.jsonl"
            inventory = root / "inventory.csv"
            primekg = root / "primekg.csv"
            grounded = root / "grounded.jsonl"
            audit = root / "audit.jsonl"
            summary = root / "summary.json"

            valid = artifact(["Target complication"])
            invalid_schema = artifact([])
            artifacts.write_text(
                json.dumps(valid) + "\n" + "{not-json\n" + json.dumps(invalid_schema) + "\n",
                encoding="utf-8",
            )
            inventory.write_text(
                "entity_id,name,source\nROOT,Root disease,ICD-10\nTARGET,Target complication,PrimeKG\n",
                encoding="utf-8",
            )
            primekg.write_text("u,v\nROOT,TARGET\n", encoding="utf-8")

            exit_code = main(
                [
                    "--artifacts-jsonl",
                    str(artifacts),
                    "--inventory-csv",
                    str(inventory),
                    "--primekg-edges-csv",
                    str(primekg),
                    "--out-grounded-jsonl",
                    str(grounded),
                    "--out-audit-jsonl",
                    str(audit),
                    "--out-summary-json",
                    str(summary),
                ]
            )

            self.assertEqual(exit_code, 0)
            grounded_values = [json.loads(line) for line in grounded.read_text(encoding="utf-8").splitlines()]
            audit_values = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
            summary_value = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(len(grounded_values), 1)
            self.assertEqual(len(audit_values), 3)
            self.assertEqual([item["first_failure"] for item in audit_values], [None, "format", "format"])
            self.assertEqual(summary_value["counts"]["input_records"], 3)
            self.assertEqual(summary_value["counts"]["format_passed_records"], 1)
            self.assertEqual(summary_value["counts"]["format_violation_records"], 2)
            self.assertEqual(summary_value["counts"]["grounded_templates"], 1)


if __name__ == "__main__":
    unittest.main()
