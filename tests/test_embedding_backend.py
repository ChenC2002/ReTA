"""Exercise the real Transformers loading/pooling path without network or patient data."""
from pathlib import Path
import json
import torch
import tempfile
import unittest

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from reta.knowledge.clustering import TextEmbedder


class EmbeddingBackendTests(unittest.TestCase):
    def test_local_transformer_loading_and_masked_mean_pooling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "clinical": 2, "concept": 3}, unk_token="[UNK]"))
            tokenizer.pre_tokenizer = Whitespace()
            PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(path)
            model = BertModel(BertConfig(vocab_size=4, hidden_size=8, num_hidden_layers=1,
                                         num_attention_heads=2, intermediate_size=16))
            model.save_pretrained(path)
            embedder = TextEmbedder(model_name=str(path), revision="local-test")
            batched = embedder.encode(["clinical", "clinical concept"])
            separate = np.concatenate([embedder.encode([text]) for text in ["clinical", "clinical concept"]])
            self.assertEqual(batched.shape, (2, 8))
            self.assertTrue(np.isfinite(batched).all())
            np.testing.assert_allclose(batched, separate, atol=1e-6, rtol=1e-5)

            # Exercise the persisted projection -> canonical cache interface with
            # a real local Transformer, not a mocked embedding backend.
            from reta.knowledge.clustering import cluster_templates, save_templates_jsonl
            from reta.knowledge.features import cache_concept_features, load_concept_features
            from reta.knowledge.pool import KnowledgePool
            from reta.learning.runtime import EntityTokenMapper
            from reta.learning.model import ConceptEmbedding
            source = Path(__file__).resolve().parents[1] / "examples/tiny_templates.jsonl"
            grounded = [KnowledgePool.load_jsonl(str(source)).templates[0].medoid]
            templates = cluster_templates(grounded, model_name=str(path), model_revision="local-test",
                                          projection_dim=4, projection_path=str(path / "projection.pt"))
            save_templates_jsonl(templates, str(path / "templates.jsonl"))
            base = {"ICD:25000": 0, "CCS:49": 1}
            external = {"UMLS:C0035309": 2, "UMLS:C0442874": 3}
            (path / "entities.json").write_text(json.dumps(external))
            (path / "descriptions.json").write_text(json.dumps({key: "clinical concept" for key in {**base, **external}}))
            torch.save({"trajectories": {}, "meta": {"icd_vocab_size": 1, "ccs_vocab_size": 1},
                        "vocab": {"name_to_token": base}}, path / "processed.pt")
            cache_concept_features(str(path / "processed.pt"), str(path / "entities.json"),
                                   str(path / "descriptions.json"), str(path / "templates.jsonl"),
                                   str(path / "projection.pt"), str(path / "features.pt"))
            kwargs = load_concept_features(path / "features.pt", EntityTokenMapper(base, external, 2),
                                           4, str(path / "templates.jsonl"))
            torch.testing.assert_close(kwargs["text_features"][0], torch.from_numpy(embedder.encode(["clinical concept"])[0]))
            embedding = ConceptEmbedding(4, 4, **kwargs)
            self.assertEqual(tuple(embedding(torch.arange(4)).shape), (4, 4))
            self.assertFalse(embedding.emb.weight.requires_grad)
