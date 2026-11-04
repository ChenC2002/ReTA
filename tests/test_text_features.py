"""Shared text-space projection, frozen feature cache, and vocabulary binding."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from reta.knowledge.clustering import fit_projection, project_to_dim
from reta.knowledge.features import cache_concept_features, load_concept_features
from reta.knowledge.pool import KnowledgePool
from reta.learning.model import ConceptEmbedding, DecoupledMultiGATEncoder, text_embedding_kwargs
from reta.learning.runtime import EntityTokenMapper, encoder_code_embedding_lookup


class TextFeatureTests(unittest.TestCase):
    def test_saved_projection_reproduces_svd_padding_and_small_pool(self):
        rng = np.random.default_rng(3)
        for shape, dim in [((8, 5), 3), ((2, 5), 3), ((2, 3), 5), ((3, 3), 3)]:
            values = rng.normal(size=shape).astype(np.float32)
            np.testing.assert_allclose(values @ fit_projection(values, dim).T,
                                       project_to_dim(values, dim), atol=1e-6)

    def test_freezes_text_features_and_shares_trainable_projection_with_retrieval(self):
        features = torch.tensor([[1., 2., 3.], [2., 1., 0.]])
        projection = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        encoder = DecoupledMultiGATEncoder(2, dim=2, attn_heads=1, gnn_layers=1,
                                           text_features=features, text_projection=projection)
        lookup = encoder_code_embedding_lookup(encoder)
        before = lookup(0)
        optimizer = torch.optim.SGD(encoder.parameters(), lr=0.1)
        encoder.embed(torch.tensor([0])).sum().backward()
        optimizer.step()
        self.assertIsNone(encoder.embed.emb.weight.grad)
        torch.testing.assert_close(encoder.embed.emb.weight, features)
        self.assertFalse(np.allclose(before, lookup(0)))
        torch.testing.assert_close(torch.from_numpy(lookup(0)), encoder.embed(torch.tensor(0)))
        restored = DecoupledMultiGATEncoder(2, dim=2, attn_heads=1, gnn_layers=1,
                                           **text_embedding_kwargs(encoder.state_dict()))
        restored.load_state_dict(encoder.state_dict())
        torch.testing.assert_close(restored.embed(torch.tensor([0, 1])), encoder.embed(torch.tensor([0, 1])))
        self.assertFalse(restored.embed.emb.weight.requires_grad)

    def test_cache_requires_complete_descriptions_and_matching_pool_and_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool_path = Path(__file__).resolve().parents[1] / "tests/fixtures/tiny_templates.jsonl"
            pool = KnowledgePool.load_jsonl(str(pool_path))
            mapping = {"ICD:25000": 0, "CCS:49": 1}
            entities = {"UMLS:C0035309": 2, "UMLS:C0442874": 3}
            torch.save({"meta": {"icd_vocab_size": 1, "ccs_vocab_size": 1},
                        "vocab": {"name_to_token": mapping}, "trajectories": {}}, root / "data.pt")
            (root / "entities.json").write_text(json.dumps(entities))
            (root / "descriptions.json").write_text(json.dumps({key: key + " description" for key in {**mapping, **entities}}))
            torch.save({"weight": torch.eye(4), "template_vectors": torch.tensor(pool.P),
                        "model_name": "fixture", "model_revision": "fixed", "backend": "transformers_mean_pooling"},
                       root / "projection.pt")
            args = [str(root / "data.pt"), str(root / "entities.json"), str(root / "descriptions.json"),
                    str(pool_path), str(root / "projection.pt"), str(root / "features.pt")]
            with patch("reta.knowledge.features.TextEmbedder") as backend:
                backend.return_value.encode.return_value = np.eye(4, dtype=np.float32)
                cache_concept_features(*args)
                self.assertEqual(len(backend.return_value.encode.call_args.args[0]), 4)
            mapper = EntityTokenMapper(mapping, entities, 2)
            kwargs = load_concept_features(root / "features.pt", mapper, 4, pool_path)
            model = ConceptEmbedding(4, 4, **kwargs)
            torch.testing.assert_close(model(torch.arange(4)), torch.eye(4))
            artifact = torch.load(root / "features.pt", weights_only=True)
            artifact["mapping_sha256"] = "wrong"
            torch.save(artifact, root / "features.pt")
            with self.assertRaisesRegex(ValueError, "token mapping"):
                load_concept_features(root / "features.pt", mapper, 4, pool_path)
            (root / "descriptions.json").write_text('{}')
            with self.assertRaisesRegex(ValueError, "Missing canonical"):
                cache_concept_features(*args)
