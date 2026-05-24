"""Dependency-light checks for ReTA repository wiring and artifact contracts."""

from __future__ import annotations

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from reta.cli import default_training_stages, main as reta_cli_main
from reta.knowledge.templates import load_templates_jsonl, validate_template_pool
from reta.policy.action import HARD, SKIP, SOFT, action_size, decode_action, encode_action


def main() -> None:
    templates_path = os.path.join(ROOT, "examples", "tiny_templates.jsonl")
    templates = load_templates_jsonl(templates_path)
    errors = validate_template_pool(templates, expected_dim=4)
    assert errors == [], errors

    assert action_size(2) == 5
    assert decode_action(0, [7, 8]).mode == SOFT
    assert decode_action(3, [7, 8]).mode == HARD
    assert decode_action(4, [7, 8]).mode == SKIP
    assert encode_action(None, SKIP, 2) == 4

    assert reta_cli_main(["validate-template-pool", "--templates_jsonl", templates_path, "--expected_dim", "4"]) == 0
    assert [stage.name for stage in default_training_stages()] == [
        "stage1_encoder_warmup",
        "stage2_reinforce_policy_refinement",
    ]
    print("smoke ok")


if __name__ == "__main__":
    main()
