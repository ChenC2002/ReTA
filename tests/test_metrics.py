"""Regression tests for paper-aligned evaluation metrics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_LEARNING_DEPS = (
    importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("torch") is not None
)


@unittest.skipUnless(HAS_LEARNING_DEPS, "NumPy and PyTorch are required")
class AccuracyAtKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import numpy as np

        from reta.learning.inference import acc_at_k, compute_all

        cls.np = np
        cls.acc_at_k = staticmethod(acc_at_k)
        cls.compute_all = staticmethod(compute_all)

    def test_normalizes_each_sample_by_available_positive_labels(self) -> None:
        true = self.np.asarray(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 1],
            ]
        )
        scores = self.np.asarray(
            [
                [0.9, 0.1, 0.8, 0.7],
                [0.1, 0.9, 0.8, 0.7],
            ]
        )

        # Per-sample scores are 1/2 and 2/2, rather than two binary hits.
        self.assertAlmostEqual(self.acc_at_k(true, scores, k=2), 0.75)

    def test_caps_denominator_at_number_of_positive_labels(self) -> None:
        true = self.np.asarray([[0, 1, 0, 0]])
        scores = self.np.asarray([[0.9, 0.8, 0.7, 0.6]])

        self.assertAlmostEqual(self.acc_at_k(true, scores, k=3), 1.0)

    def test_average_precision_handles_ties_independently_of_row_order(self):
        from reta.learning.inference import auprc_micro
        true = self.np.asarray([[1, 0]])
        scores = self.np.asarray([[0.5, 0.5]])
        self.assertEqual(auprc_micro(true, scores), 0.5)
        self.assertEqual(auprc_micro(true[:, ::-1], scores), 0.5)

    def test_single_label_micro_f1_does_not_count_true_negatives(self):
        from reta.learning.inference import micro_f1
        self.assertEqual(micro_f1([[1], [0], [0], [0]], [[0], [0], [0], [0]]), 0.)

    def test_macro_ap_is_label_wise_and_keeps_empty_labels(self):
        from reta.learning.inference import auprc_macro, auprc_micro
        true = self.np.asarray([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
        scores = self.np.asarray([[.9, .8, .1], [.7, .1, .2], [.6, .2, .3]])
        self.assertAlmostEqual(auprc_macro(true, scores), (1 + 1/3 + 0) / 3)
        self.assertNotAlmostEqual(auprc_macro(true, scores), auprc_micro(true, scores))

    def test_validation_threshold_improves_f1_and_matches_exhaustive_search(self):
        from reta.learning.inference import select_f1_threshold, micro_f1, run_inference
        true = self.np.asarray([[1, 0], [0, 1]])
        probs = self.np.asarray([[.4, .1], [.2, .3]])
        threshold = select_f1_threshold(true, probs)
        self.assertEqual(threshold, .3)
        expected = max(micro_f1(true, probs >= t) for t in self.np.unique(probs))
        self.assertEqual(micro_f1(true, probs >= threshold), expected)
        self.assertGreater(expected, micro_f1(true, probs >= .5))
        for split in ("test", "train", "all"):
            with self.assertRaisesRegex(ValueError, "validation split"):
                run_inference("missing", "missing", "missing", None, None, split, calibrate_threshold=True)

    def test_compute_all_uses_normalized_accuracy(self) -> None:
        true = self.np.asarray([[1, 1, 0], [0, 1, 1]])
        probabilities = self.np.asarray([[0.9, 0.1, 0.8], [0.1, 0.9, 0.8]])

        metrics = self.compute_all(true, probs=probabilities, k=2)

        self.assertAlmostEqual(metrics["Acc@2"], 0.75)


if __name__ == "__main__":
    unittest.main()
