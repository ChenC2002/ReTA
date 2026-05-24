"""
Semantic channel encoder (feature-only self-attention).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _require_pyg():
    try:
        from torch_geometric.utils import to_dense_batch  # noqa: F401
    except Exception as e:
        raise ImportError("torch-geometric is required for semantic batching via to_dense_batch.") from e


class SemanticEncoder(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = int(dim)
        self.attn = nn.MultiheadAttention(self.dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.dim * 4, self.dim),
        )
        self.ln1 = nn.LayerNorm(self.dim)
        self.ln2 = nn.LayerNorm(self.dim)

    def forward(self, x: torch.Tensor, batch: torch.Tensor, code_mask: torch.Tensor) -> torch.Tensor:
        """Return h_sem (N,d). Nodes outside the observed-code mask -> 0."""
        _require_pyg()
        from torch_geometric.utils import to_dense_batch

        x_dense, mask = to_dense_batch(x, batch=batch)  # (B,T,d), (B,T)
        orig_dense, _ = to_dense_batch(code_mask.float().unsqueeze(-1), batch=batch)
        orig_dense = orig_dense.squeeze(-1).bool()
        valid_mask = mask & orig_dense
        key_padding_mask = ~valid_mask
        x_dense = x_dense.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        h, _ = self.attn(x_dense, x_dense, x_dense, key_padding_mask=key_padding_mask, need_weights=False)
        x_dense = self.ln1(x_dense + h)

        h2 = self.ffn(x_dense)
        x_dense = self.ln2(x_dense + h2)

        h_sem = torch.zeros_like(x)
        B = x_dense.size(0)
        for g in range(B):
            idx = (batch == g).nonzero(as_tuple=False).view(-1)
            tlen = int(mask[g].sum().item())
            h_sem[idx] = x_dense[g, :tlen]

        h_sem[~code_mask] = 0.0
        return h_sem
