"""Recovery and mixed-action batch equivalence, including optional CUDA checks."""
import copy
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from reta.learning.checkpointing import atomic_save, capture_rng, restore_rng
from reta.learning.model import DecoupledMultiGATEncoder, batch_graphs, build_pyg_data
from reta.learning.runtime import load_checkpoint


class RecoveryTests(unittest.TestCase):
    def test_rng_weights_only_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            atomic_save({"rng": capture_rng()}, path)
            expected = (random.random(), np.random.rand(), torch.rand(3))
            restore_rng(load_checkpoint(path)["rng"])
            actual = (random.random(), np.random.rand(), torch.rand(3))
            self.assertEqual(expected[:2], actual[:2])
            torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)

    def test_failed_write_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.pt"
            atomic_save({"value": 1}, path)
            with mock.patch("torch.save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    atomic_save({"value": 2}, path)
            self.assertEqual(load_checkpoint(path)["value"], 1)
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_mixed_action_batch_matches_separate_outputs_and_gradients(self):
        device = torch.device(os.environ.get("RETA_TEST_DEVICE", "cpu"))
        torch.manual_seed(91)
        encoder = DecoupledMultiGATEncoder(8, dim=4, gnn_layers=1, attn_heads=2, dropout=0).to(device)
        reference = copy.deepcopy(encoder)
        graphs = [build_pyg_data([0, 1], torch.tensor([[0, 1], [1, 0]])),
                  build_pyg_data([2, 3, 4], torch.tensor([[0, 1], [1, 2]])),
                  build_pyg_data([5], torch.empty((2, 0), dtype=torch.long))]
        offsets = torch.randn(3, 4, device=device)
        memories = [torch.randn(2, 4, device=device), torch.empty(0, 4, device=device),
                    torch.randn(1, 4, device=device)]
        output, _ = encoder(batch_graphs(graphs).to(device), past_visit_memory=memories,
                            soft_offset_per_graph=offsets,
                            soft_import_mask=torch.tensor([True, False, False], device=device))
        separate = []
        for i, graph in enumerate(graphs):
            value, _ = reference(batch_graphs([graph]).to(device), past_visit_memory=[memories[i]],
                                 soft_offset_per_graph=offsets[i:i+1] if i == 0 else None)
            separate.append(value)
        expected = torch.cat(separate)
        torch.testing.assert_close(output, expected, atol=2e-5, rtol=2e-5)
        output.square().mean().backward()
        expected.square().mean().backward()
        for p, q in zip(encoder.parameters(), reference.parameters()):
            if p.grad is None or q.grad is None:
                self.assertIs(p.grad, q.grad)
            else:
                torch.testing.assert_close(p.grad, q.grad, atol=3e-5, rtol=3e-5)


if __name__ == "__main__":
    unittest.main()
