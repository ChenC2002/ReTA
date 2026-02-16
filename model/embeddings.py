"""
We use a shared token namespace (as in `data/ontology.py`):
- ICD leaf tokens first
- CCS tokens next

This module provides:
- a learnable embedding table
- utilities to map CCS token ids (shared namespace) -> contiguous label indices
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def ccs_token_to_label_index(ccs_token: torch.Tensor, icd_vocab_size: int) -> torch.Tensor:
    """CCS token ids (shared namespace) -> contiguous label indices."""
    return ccs_token - int(icd_vocab_size)


def labels_to_multihot_from_ccs_tokens(ccs_tokens, num_ccs_labels: int, icd_vocab_size: int, device=None) -> torch.Tensor:
    """Build multi-hot vector from a list/1D tensor of CCS token ids."""
    if isinstance(ccs_tokens, (list, tuple)):
        ccs_tokens = torch.tensor(ccs_tokens, dtype=torch.long, device=device)
    else:
        ccs_tokens = ccs_tokens.to(device=device)

    idx = ccs_token_to_label_index(ccs_tokens, icd_vocab_size)
    y = torch.zeros(num_ccs_labels, dtype=torch.float32, device=device)
    if idx.numel() > 0:
        y[idx] = 1.0
    return y


class ConceptEmbedding(nn.Module):
    """Shared embedding table for ICD/CCS/KG concept tokens."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        padding_idx: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.emb = nn.Embedding(self.vocab_size, self.dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed token ids -> (*, dim)."""
        x = self.emb(token_ids)
        return self.dropout(x)

    def load_pretrained(self, weights: torch.Tensor, strict: bool = True):
        """Load pretrained weights (partial load allowed if strict=False)."""
        if weights.shape != self.emb.weight.shape:
            msg = f"Pretrained embedding shape {tuple(weights.shape)} != {tuple(self.emb.weight.shape)}"
            if strict:
                raise ValueError(msg)
        with torch.no_grad():
            n = min(weights.shape[0], self.emb.weight.shape[0])
            d = min(weights.shape[1], self.emb.weight.shape[1])
            self.emb.weight[:n, :d].copy_(weights[:n, :d])
