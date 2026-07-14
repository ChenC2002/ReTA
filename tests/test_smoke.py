"""Dependency-light checks for repository wiring and public contracts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_TORCH = importlib.util.find_spec("torch") is not None

from reta.cli import (
    build_external_entity_mapping,
    default_training_stages,
    main as reta_cli_main,
)
from reta.knowledge.pool import load_templates_jsonl, validate_template_pool


def strict_json_loads(payload: str):
    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON number: {value}")

    return json.loads(
        payload,
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
    )


class SmokeTests(unittest.TestCase):
    def test_paper_results_and_log(self) -> None:
        result_path = ROOT / "results" / "paper_results.json"
        log_path = ROOT / "logs" / "paper_results.jsonl"
        result_text = result_path.read_text(encoding="utf-8")
        result = strict_json_loads(result_text)

        self.assertEqual(result["schema_version"], "1.0.0")
        self.assertEqual(
            result["paper_title"],
            "Import What You Need: Learning When and How to Augment EHR Graphs with External Knowledge",
        )
        self.assertEqual(result["scope"], "paper_wide_reported_numeric_results")
        self.assertEqual(result["record_count"], len(result["records"]))
        self.assertGreaterEqual(result["record_count"], 140)
        self.assertEqual(
            set(result["protocol"]["tasks"]),
            {"diagnosis_prediction", "in_hospital_mortality", "readmission_30_day"},
        )
        self.assertEqual(set(result["protocol"]["datasets"]), {"mimic_iii", "mimic_iv"})
        self.assertEqual(result["protocol"]["default_reta_knowledge_source"], "primekg")
        self.assertEqual(result["protocol"]["evaluation_splits"]["performance"], "test")
        self.assertEqual(
            result["protocol"]["evaluation_splits"]["mechanism_diagnostics"],
            {
                "split": "validation",
                "locations": ["figure_10", "table_10", "table_11"],
            },
        )
        self.assertEqual(result["protocol"]["percentage_metric_scale"], "percentage_points")

        required_locations = {
            "table_1",
            "figure_3",
            "figure_4",
            "figure_5",
            "figure_6",
            "table_4",
            "table_6",
            "table_7",
            "table_8",
            "figure_9",
            "table_9",
            "figure_10",
            "table_10",
            "table_11",
            "appendix_e_4",
        }
        self.assertTrue(required_locations.issubset(result["coverage"]))

        ids = [record["id"] for record in result["records"]]
        self.assertEqual(len(ids), len(set(ids)))
        for record in result["records"]:
            self.assertTrue(record["section"])
            self.assertTrue(record["location"])
            self.assertTrue(record["dimensions"])
            self.assertTrue(record["metrics"])
            metric_names = [metric["name"] for metric in record["metrics"]]
            self.assertEqual(len(metric_names), len(set(metric_names)))
            for released_metric in record["metrics"]:
                self.assertIsInstance(released_metric["value"], (int, float))
                self.assertTrue(released_metric["unit"])
                self.assertTrue(released_metric["direction"])
                if "std" in released_metric:
                    self.assertGreaterEqual(released_metric["std"], 0)

        log_events = [
            strict_json_loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(log_events[0]["event"], "manifest")
        self.assertEqual(log_events[1]["event"], "protocol")
        self.assertEqual(log_events[-1]["event"], "complete")
        self.assertEqual(log_events[0]["paper_title"], result["paper_title"])
        self.assertEqual(log_events[0]["scope"], result["scope"])
        self.assertEqual(log_events[1]["protocol"], result["protocol"])
        logged_records = [event["record"] for event in log_events if event["event"] == "result"]
        self.assertEqual(logged_records, result["records"])
        self.assertEqual(log_events[0]["record_count"], result["record_count"])
        self.assertEqual(log_events[-1]["record_count"], result["record_count"])
        public_text = result_text.lower() + log_path.read_text(encoding="utf-8").lower()
        for unfinished_term in ("placeholder", "warning", "simulated", "stimulated"):
            self.assertNotIn(unfinished_term, public_text)

    def test_template_pool_cli(self) -> None:
        templates_path = ROOT / "examples" / "tiny_templates.jsonl"
        templates = load_templates_jsonl(str(templates_path))
        self.assertEqual(validate_template_pool(templates, expected_dim=4), [])

        self.assertEqual(
            reta_cli_main(
                [
                    "validate-template-pool",
                    "--templates_jsonl",
                    str(templates_path),
                    "--expected_dim",
                    "4",
                ]
            ),
            0,
        )

    def test_external_entity_mapping_is_complete_and_deterministic(self) -> None:
        templates = load_templates_jsonl(str(ROOT / "examples" / "tiny_templates.jsonl"))
        mapping = build_external_entity_mapping(
            templates,
            {"ICD:25000": 0, "CCS:49": 1},
        )
        self.assertEqual(
            mapping,
            {
                "UMLS:C0035309": 2,
                "UMLS:C0442874": 3,
            },
        )

    def test_training_stage_contract(self) -> None:
        stages = default_training_stages()
        self.assertEqual(
            [stage.name for stage in stages],
            ["stage1_encoder_warmup", "stage2_reinforce_policy_refinement"],
        )
        self.assertEqual(stages[0].metrics, ["mean BCE loss"])
        self.assertEqual(
            stages[1].metrics,
            [
                "policy loss",
                "policy entropy",
                "mean return",
                "running baseline",
                "Soft/Hard/Skip action rates",
                "supervised BCE loss",
            ],
        )

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed in the dependency-light test environment")
    def test_consolidated_policy_action_contract(self) -> None:
        from reta.learning.policy import HARD, SKIP, SOFT, action_size, decode_action, encode_action

        self.assertEqual(action_size(2), 5)
        self.assertEqual(decode_action(0, [7, 8]).mode, SOFT)
        self.assertEqual(decode_action(3, [7, 8]).mode, HARD)
        self.assertEqual(decode_action(4, [7, 8]).mode, SKIP)
        self.assertEqual(encode_action(None, SKIP, 2), 4)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed in the dependency-light test environment")
    def test_consolidated_prediction_head(self) -> None:
        import torch

        from reta.learning.model import NextVisitPredictor

        predictor = NextVisitPredictor(in_dim=4, num_labels=3, dropout=0.0)
        logits = predictor(torch.zeros((2, 4)))
        self.assertEqual(tuple(logits.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
