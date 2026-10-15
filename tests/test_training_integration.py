"""Synthetic two-stage training, exact CPU recovery and CUDA smoke validation.

No clinical data or paper-performance assertions are used here.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
import yaml

from reta.learning import warmup, rl_train, inference
from reta.learning.runtime import Config, load_checkpoint


class TrainingIntegrationTests(unittest.TestCase):
    def test_two_stage_resume_and_inference(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Config()
            cfg.model.dim = 4
            cfg.model.attn_heads = 2
            cfg.model.gnn_layers = 1
            cfg.model.dropout = 0.1
            cfg.data.batch_size = 3
            cfg.train.device = os.environ.get("RETA_TEST_DEVICE", "cpu")
            cfg.train.warmup_epochs = 2
            cfg.train.rl_iters = 2
            cfg.train.rollout_patients = 2
            cfg.train.encoder_updates_per_iter = 1
            cfg.knowledge.retrieval_K = 1
            cfg.knowledge.templates_jsonl = str(Path(__file__).resolve().parents[1] / "examples/tiny_templates.jsonl")
            cfg.knowledge.entity_to_token_json = str(root / "entities.json")
            (root / "entities.json").write_text(json.dumps({"UMLS:C0035309": 3, "UMLS:C0442874": 4}))
            cfg.data.processed_path = str(root / "processed.pt")
            cfg.data.split_json = str(root / "splits.json")
            (root / "splits.json").write_text(json.dumps({"train": ["a", "b"], "val": ["c"], "test": ["d"]}))
            trajectories = {}
            for pid in "abcd":
                trajectories[pid] = [dict(node_ids=[0, 1], edge_index=[[0, 1], [1, 0]],
                                          icd_tokens=[0], ccs_ancestors=[1], label_ccs=[1 + t % 2])
                                     for t in range(3)]
            torch.save(dict(trajectories=trajectories,
                            meta=dict(icd_vocab_size=1, ccs_vocab_size=2, ccs_label_vocab_size=2),
                            vocab=dict(name_to_token={"ICD:25000": 0, "CCS:49": 1, "CCS:50": 2})),
                       cfg.data.processed_path)

            def run(module, *args):
                config = root / "config.yaml"
                config.write_text(yaml.safe_dump(asdict(cfg)))
                with mock.patch.object(sys, "argv", [module.__name__, "--config", str(config), *args]):
                    module.main()

            def outputs(name):
                cfg.outputs.checkpoints_dir = str(root / name)
                cfg.outputs.logs_dir = str(root / (name + "-logs"))

            def equal_states(a, b, names):
                # CUDA scatter reductions need not be bitwise deterministic.
                tolerance = 0 if cfg.train.device == "cpu" else 1e-5
                for name in names:
                    for key in a[name]:
                        torch.testing.assert_close(a[name][key], b[name][key], rtol=tolerance, atol=tolerance)

            outputs("full")
            run(warmup)
            full_warm = load_checkpoint(root / "full/warmup.pt")
            # Verify reward uses live raw BCE, independently of frozen uncertainty.
            original_reward = rl_train.compute_paired_reward
            reward_inputs = []
            def reward(raw, edited, **kwargs):
                reward_inputs.append((raw.item(), edited.item(), kwargs["is_skip"]))
                if kwargs["is_skip"]:
                    torch.testing.assert_close(raw, edited, atol=1e-6, rtol=1e-6)
                return original_reward(raw, edited, **kwargs)
            with mock.patch.object(rl_train, "compute_paired_reward", side_effect=reward):
                run(rl_train)
            self.assertTrue(reward_inputs)
            full_rl = load_checkpoint(root / "full/rl_iter2.pt")

            outputs("resumed")
            cfg.train.warmup_epochs = 1
            run(warmup)
            cfg.train.warmup_epochs = 2
            run(warmup, "--resume", str(root / "resumed/warmup.pt"))
            equal_states(full_warm, load_checkpoint(root / "resumed/warmup.pt"), ["encoder", "predictor"])
            cfg.train.rl_iters = 1
            run(rl_train)
            cfg.train.rl_iters = 2
            run(rl_train, "--resume", str(root / "resumed/rl_iter1.pt"))
            resumed = load_checkpoint(root / "resumed/rl_iter2.pt")
            equal_states(full_rl, resumed, ["encoder", "predictor", "policy", "state_encoder", "baseline_encoder", "baseline_predictor"])
            self.assertEqual(set(full_rl["template_utilities"]), set(resumed["template_utilities"]))
            for key in full_rl["template_utilities"]:
                self.assertAlmostEqual(full_rl["template_utilities"][key], resumed["template_utilities"][key], places=5)
            self.assertEqual(resumed["training_state"]["completed"], 2)
            from reta.learning.select_checkpoint import select_checkpoint
            selection = select_checkpoint(
                [root / "resumed/rl_iter1.pt", root / "resumed/rl_iter2.pt"], cfg,
                root / "resumed/best.pt", root / "selection.json",
            )
            self.assertEqual(selection["selection"]["split"], "val")
            scores = [record["metrics"]["AUPRC_micro"] for record in selection["candidates"]]
            self.assertEqual(selection["selection"]["score"], max(scores))
            self.assertEqual(load_checkpoint(root / "resumed/best.pt")["selection"], selection["selection"])
            with mock.patch.object(sys, "argv", ["inference", "--checkpoint", str(root / "resumed/best.pt"),
                                   "--processed_path", cfg.data.processed_path,
                                   "--templates_jsonl", cfg.knowledge.templates_jsonl,
                                   "--split_json", cfg.data.split_json,
                                   "--device", cfg.train.device,
                                   "--log_dir", cfg.outputs.logs_dir,
                                   "--split", "test", "--out", str(root / "inference.json")]):
                inference.main()
            self.assertTrue((root / "inference.json").is_file())


if __name__ == "__main__":
    unittest.main()
