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

    def test_compute_all_uses_normalized_accuracy(self) -> None:
        true = self.np.asarray([[1, 1, 0], [0, 1, 1]])
        probabilities = self.np.asarray([[0.9, 0.1, 0.8], [0.1, 0.9, 0.8]])

        metrics = self.compute_all(true, probs=probabilities, k=2)

        self.assertAlmostEqual(metrics["Acc@2"], 0.75)


if __name__ == "__main__":
    unittest.main()
