"""Exercise the real Transformers loading/pooling path without network or patient data."""
from pathlib import Path
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
