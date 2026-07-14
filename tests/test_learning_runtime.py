"""Focused tests for learning-space and checkpoint integrity contracts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HAS_LEARNING_DEPS = (
    importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("torch") is not None
)
HAS_PYG = HAS_LEARNING_DEPS and importlib.util.find_spec("torch_geometric") is not None


@unittest.skipUnless(HAS_LEARNING_DEPS, "NumPy and PyTorch are required")
class LearningRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import numpy as np

        from reta.learning import runtime

        cls.np = np
        cls.runtime = runtime

    def test_external_tokens_extend_namespace_without_aliasing(self) -> None:
        mapper = self.runtime.EntityTokenMapper(
            {"ICD:A": 0, "CCS:X": 1, "CCS:ROOT": 2},
            {"UMLS:1": 3, "PrimeKG:2": 4},
            base_vocab_size=3,
        )
        self.assertEqual(mapper.vocab_size, 5)
        self.assertEqual(mapper.to_token("UMLS:1"), 3)

        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.runtime.EntityTokenMapper(
                {"ICD:A": 0, "CCS:X": 1, "CCS:ROOT": 2},
                {"UMLS:1": 9},
                base_vocab_size=3,
            )
        with self.assertRaisesRegex(ValueError, "redefines"):
            self.runtime.EntityTokenMapper(
                {"ICD:A": 0, "CCS:X": 1, "CCS:ROOT": 2},
                {"ICD:A": 3},
                base_vocab_size=3,
            )

    def test_pool_index_initializes_live_code_embeddings(self) -> None:
        import torch

        mapper = self.runtime.EntityTokenMapper(
            {"ICD:A": 0, "CCS:X": 1, "CCS:ROOT": 2},
            {},
            base_vocab_size=3,
        )
        templates = [
            SimpleNamespace(
                template_id=0,
                medoid=SimpleNamespace(
                    root_concept_id="CCS:X",
                    subgraph_nodes=["CCS:X", "ICD:A"],
                ),
            ),
            SimpleNamespace(
                template_id=1,
                medoid=SimpleNamespace(
                    root_concept_id="CCS:ROOT",
                    subgraph_nodes=["CCS:ROOT"],
                ),
            ),
        ]
        pool = SimpleNamespace(
            templates=templates,
            P=self.np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=self.np.float32),
        )
        index = self.runtime.PoolAlignedTokenIndex(pool, mapper)
        tokens = self.runtime.visit_retrieval_tokens(
            {"icd_tokens": [0], "ccs_ancestors": [1, 2, 1]}
        )
        self.assertEqual(tokens, [0])

        encoder = SimpleNamespace(
            embed=SimpleNamespace(
                emb=SimpleNamespace(weight=torch.nn.Parameter(torch.zeros(3, 2)))
            )
        )
        self.assertEqual(index.initialize_embedding_table(encoder.embed.emb.weight), 3)
        lookup = self.runtime.encoder_code_embedding_lookup(encoder)
        self.np.testing.assert_allclose(lookup(0), [1.0, 0.0])
        with torch.no_grad():
            encoder.embed.emb.weight[0].copy_(torch.tensor([0.0, 2.0]))
        self.np.testing.assert_allclose(lookup(0), [0.0, 2.0])

    def test_retrieval_uses_observed_icd_and_folded_gru_state(self) -> None:
        self.assertEqual(
            self.runtime.RETRIEVAL_SPACE,
            "learnable_code_embedding_gru_state_v1",
        )
        tokens = self.runtime.visit_retrieval_tokens(
            {"icd_tokens": [4, 4, 2], "ccs_ancestors": [8, 9]}
        )
        self.assertEqual(tokens, [4, 2])
        folded = self.runtime.fold_policy_state_for_retrieval(
            self.np.asarray([3.0, 0.0, 0.0, 4.0], dtype=self.np.float32),
            dim=2,
        )
        self.np.testing.assert_allclose(folded, [0.6, 0.8], atol=1e-6)
        with self.assertRaisesRegex(ValueError, "observed ICD"):
            self.runtime.visit_retrieval_tokens({"ccs_ancestors": [8]})

    def test_pool_index_rejects_incomplete_external_entity_map(self) -> None:
        mapper = self.runtime.EntityTokenMapper(
            {"ICD:A": 0, "CCS:X": 1},
            {},
            base_vocab_size=2,
        )
        pool = SimpleNamespace(
            templates=[
                SimpleNamespace(
                    template_id=0,
                    medoid=SimpleNamespace(
                        root_concept_id="CCS:X",
                        subgraph_nodes=["CCS:X", "UMLS:MISSING"],
                    ),
                )
            ],
            P=self.np.asarray([[1.0, 0.0]], dtype=self.np.float32),
        )
        with self.assertRaisesRegex(ValueError, "build-entity-map"):
            self.runtime.PoolAlignedTokenIndex(pool, mapper)

    def test_checkpoint_contract_rejects_pool_or_mapping_drift(self) -> None:
        mapper = self.runtime.EntityTokenMapper(
            {"ICD:A": 0, "CCS:X": 1},
            {},
            base_vocab_size=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            templates = Path(tmp) / "templates.jsonl"
            templates.write_text("{}\n", encoding="utf-8")
            processed = Path(tmp) / "processed.pt"
            processed.write_bytes(b"processed")
            contract = self.runtime.build_checkpoint_contract(
                mapper=mapper,
                processed_path=str(processed),
                templates_path=str(templates),
                split_json=None,
                training_split="all",
                icd_vocab_size=1,
                ccs_vocab_size=1,
                ccs_label_vocab_size=1,
                model_dim=2,
                gnn_layers=2,
                attn_heads=2,
                dropout=0.1,
                pool_dim=2,
                retrieval_K=2,
                retrieval_alpha=0.2,
                soft_xi=0.5,
            )
            self.assertIn("processed_sha256", contract)
            self.assertIn("training_split_sha256", contract)
            self.runtime.require_checkpoint_contract({"contract": contract}, contract)
            changed = dict(contract)
            changed["templates_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "templates_sha256"):
                self.runtime.require_checkpoint_contract({"contract": contract}, changed)
            with self.assertRaisesRegex(ValueError, "format-v2"):
                self.runtime.require_checkpoint_contract({}, contract)

    def test_split_manifest_rejects_patient_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splits.json"
            path.write_text(
                json.dumps({"train": ["1", "2"], "test": ["2"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "both"):
                self.runtime.load_split_manifest(str(path))

            path.write_text(
                json.dumps({"train": ["1"], "test": ["2"], "val": ["3"]}),
                encoding="utf-8",
            )
            selected = self.runtime.select_patient_ids(
                ["1", "2", "3"],
                str(path),
                "test",
            )
            self.assertEqual(selected, ["2"])

            path.write_text(
                json.dumps({"train": ["1"], "test": ["2"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly cover"):
                self.runtime.select_patient_ids(
                    ["1", "2", "3"],
                    str(path),
                    "train",
                )

    def test_strict_json_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mapping.json"
            path.write_text('{"UMLS:1": 2, "UMLS:1": 3}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate key"):
                self.runtime.load_json_object_strict(
                    str(path),
                    "external token mapping",
                )

    def test_strict_json_rejects_overflowed_float(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mapping.json"
            path.write_text('{"UMLS:1": 1e999}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite value 1e999"):
                self.runtime.load_json_object_strict(
                    str(path),
                    "external token mapping",
                )

    def test_config_rejects_duplicate_root_mapping_key(self) -> None:
        if self.runtime.yaml is None:
            self.skipTest("PyYAML is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "train:\n  seed: 1\ntrain:\n  seed: 2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate YAML mapping key 'train'"):
                self.runtime.load_config(str(path))

    def test_config_rejects_duplicate_nested_mapping_key(self) -> None:
        if self.runtime.yaml is None:
            self.skipTest("PyYAML is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "train:\n  seed: 1\n  seed: 2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate YAML mapping key 'seed'"):
                self.runtime.load_config(str(path))

    def test_checkpoint_loader_enables_weights_only(self) -> None:
        with mock.patch.object(
            self.runtime.torch,
            "load",
            return_value={"encoder": {}},
        ) as load:
            self.runtime.load_checkpoint("checkpoint.pt")
        load.assert_called_once_with(
            "checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )

    def test_config_validation_fails_before_model_construction(self) -> None:
        cfg = self.runtime.Config()
        cfg.model.dim = 7
        cfg.model.attn_heads = 2
        with self.assertRaisesRegex(ValueError, "divisible"):
            self.runtime.validate_config(cfg)

        cfg = self.runtime.Config()
        cfg.knowledge.retrieval_K = 0
        with self.assertRaisesRegex(ValueError, "retrieval_K"):
            self.runtime.validate_config(cfg)

    def test_metrics_reject_invalid_inputs(self) -> None:
        from reta.learning.inference import compute_all

        with self.assertRaisesRegex(ValueError, "shape"):
            compute_all([[1, 0]], probs=[[0.9]])
        with self.assertRaisesRegex(ValueError, "binary"):
            compute_all([[2, 0]], probs=[[0.9, 0.1]])
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_all([[1, 0]], probs=[[float("nan"), 0.1]])
        with self.assertRaisesRegex(ValueError, "positive"):
            compute_all([[1, 0]], probs=[[0.9, 0.1]], k=0)


@unittest.skipUnless(HAS_LEARNING_DEPS, "PyTorch is required")
class FixedPolicyActionTests(unittest.TestCase):
    def test_unfilled_candidate_slots_are_masked(self) -> None:
        from reta.learning.policy import (
            HARD,
            SKIP,
            decode_policy_action,
            valid_action_mask,
        )

        mask = valid_action_mask(candidate_count=1, K=3).tolist()
        self.assertEqual(mask, [True, False, False, True, False, False, True])
        self.assertEqual(
            valid_action_mask(candidate_count=0, K=3).tolist(),
            [False, False, False, False, False, False, True],
        )
        self.assertEqual(decode_policy_action(3, [17], 3).mode, HARD)
        self.assertEqual(decode_policy_action(6, [17], 3).mode, SKIP)
        with self.assertRaisesRegex(ValueError, "unavailable"):
            decode_policy_action(1, [17], 3)


@unittest.skipUnless(HAS_PYG, "PyTorch Geometric is required")
class VisitGraphContractTests(unittest.TestCase):
    def test_visit_graph_validates_nodes_and_masks(self) -> None:
        import torch

        from reta.learning.model import build_pyg_data

        edges = torch.tensor([[0, 1], [1, 0]])
        with self.assertRaisesRegex(ValueError, "unique"):
            build_pyg_data([2, 2], edges)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_pyg_data([], torch.zeros((2, 0), dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "orig_mask"):
            build_pyg_data([2, 3], edges, orig_mask=torch.tensor([True]))

    def test_hard_import_returns_original_for_duplicate_edges(self) -> None:
        import torch

        from reta.learning.model import TemplateSubgraph, build_pyg_data, graft_hard_import

        base = build_pyg_data(
            [10, 11],
            torch.tensor([[0, 1], [1, 0]]),
        )
        duplicate = TemplateSubgraph(
            node_ids=[10, 11],
            edge_index=torch.tensor([[0, 1], [1, 0]]),
        )
        self.assertIs(graft_hard_import(base, duplicate), base)


if __name__ == "__main__":
    unittest.main()
