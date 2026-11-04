"""Dependency-light checks for repository wiring and public contracts."""

from __future__ import annotations

import importlib.util
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


class SmokeTests(unittest.TestCase):
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
