"""Fail-fast training contracts and held-out checkpoint selection."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from reta.learning.checkpointing import validate_resume
from reta.learning.runtime import Config, TrajectorySampler, validate_config
from reta.learning.select_checkpoint import select_checkpoint


class TrainingValidationTests(unittest.TestCase):
    def test_invalid_seed_is_rejected_before_numpy(self):
        for seed in (-1, 2**32):
            cfg = Config()
            cfg.train.seed = seed
            with self.assertRaisesRegex(ValueError, "seed"):
                validate_config(cfg)

    def test_sampling_excludes_unlabeled_and_short_trajectories(self):
        sampler = TrajectorySampler.__new__(TrajectorySampler)
        sampler.trajectories = {"short": [{"label_ccs": [1]}], "unlabeled": [{}, {}],
                                "late": [{}, {"label_ccs": [1]}, {}],
                                "valid": [{"label_ccs": []}, {}]}
        sampler.patient_ids = list(sampler.trajectories)
        sampler.filter_trainable_patients(max_visits=2)
        self.assertEqual(sampler.patient_ids, ["valid"])
        sampler.patient_ids = ["short", "unlabeled"]
        with self.assertRaisesRegex(ValueError, "no labeled"):
            sampler.filter_trainable_patients()

    def test_invalid_resume_boundaries_and_missing_state(self):
        cfg = Config()
        for boundary in (0, -1, True, "1", cfg.train.warmup_epochs):
            checkpoint = {"cfg": asdict(cfg), "training_state": {"stage": "warmup", "completed": boundary}}
            with self.assertRaises(ValueError):
                validate_resume(checkpoint, cfg, "warmup")
        checkpoint["training_state"]["completed"] = 1
        with self.assertRaisesRegex(ValueError, "optimizer"):
            validate_resume(checkpoint, cfg, "warmup")
        checkpoint["training_state"]["optimizers"] = {"encoder": {}}
        with self.assertRaisesRegex(ValueError, "RNG"):
            validate_resume(checkpoint, cfg, "warmup")

    def test_selection_never_uses_test_split_and_resolves_ties(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Config()
            cfg.data.split_json = str(root / "splits.json")
            a, b = root / "a.pt", root / "b.pt"
            for path in (a, b):
                torch.save({"contract": {"training_split": "train"}}, path)
            with mock.patch("reta.learning.select_checkpoint.run_inference",
                            return_value={"metrics": {"AUPRC_micro": 0.5}}) as infer:
                result = select_checkpoint([b, a], cfg, root / "best.pt", root / "report.json")
                self.assertEqual(result["selection"]["source"], str(a))
                self.assertTrue(all(call.kwargs["split"] == "val" for call in infer.call_args_list))
            with self.assertRaisesRegex(ValueError, "overwrite"):
                select_checkpoint([a], cfg, a, root / "report.json")
            torch.save({"contract": {"training_split": "all"}}, a)
            with mock.patch("reta.learning.select_checkpoint.run_inference") as infer:
                with self.assertRaisesRegex(ValueError, "train split"):
                    select_checkpoint([a], cfg, root / "best.pt", root / "report.json")
                infer.assert_not_called()

    def test_selection_rejects_nonfinite_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Config()
            cfg.data.split_json = str(root / "splits.json")
            path = root / "candidate.pt"
            torch.save({"contract": {"training_split": "train"}}, path)
            with mock.patch("reta.learning.select_checkpoint.run_inference",
                            return_value={"metrics": {"AUPRC_micro": float("nan")}}):
                with self.assertRaisesRegex(ValueError, "Non-finite"):
                    select_checkpoint([path], cfg, root / "best.pt", root / "report.json")
            self.assertFalse((root / "best.pt").exists())
