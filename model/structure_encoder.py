"""
Structure channel encoder (GAT + temporal attention).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


def _require_pyg():
    try:
        from torch_geometric.nn import GATConv  # noqa: F401
    except Exception as e:
        raise ImportError("torch-geometric is required for GATConv.") from e


class StructureEncoder(nn.Module):
    def __init__(self, dim: int = 256, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        _require_pyg()
        from torch_geometric.nn import GATConv

        self.dim = int(dim)
        self.dropout = float(dropout)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(int(num_layers)):
            self.convs.append(GATConv(self.dim, self.dim, heads=num_heads, concat=False, dropout=self.dropout, add_self_loops=False))
            self.norms.append(nn.LayerNorm(self.dim))

        self.W_T = nn.Linear(self.dim, self.dim, bias=False)
        self.fuse_gate = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        past_visit_memory: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        # intra
        h = x
        for conv, ln in zip(self.convs, self.norms):
            h2 = conv(h, edge_index)
            h2 = torch.dropout(h2, p=self.dropout, train=self.training)
            h = ln(h + h2)
        h_intra = h

        if past_visit_memory is None:
            return h_intra

        # inter + fuse (per graph)
        h_struct = torch.zeros_like(h_intra)
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        for g in range(B):
            idx = (batch == g).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            mem = past_visit_memory[g]
            if mem is None or mem.numel() == 0:
                h_struct[idx] = h_intra[idx]
                continue

            mem = mem.to(h_intra.device)           # (H,d)
            q = self.W_T(h_intra[idx])             # (n_g,d)
            attn = torch.softmax(q @ mem.T, dim=1) # (n_g,H)
            h_inter = attn @ mem                   # (n_g,d)

            gate = torch.sigmoid(self.fuse_gate(torch.cat([h_intra[idx], h_inter], dim=1)))  # (n_g,1)
            h_struct[idx] = gate * h_intra[idx] + (1.0 - gate) * h_inter

        return h_struct
