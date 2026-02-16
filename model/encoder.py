"""
Decoupled Multi-GAT encoder.

Expects PyG Batch with:
- node_ids (N,)
- edge_index (2,E)
- orig_mask (N,)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .embeddings import ConceptEmbedding
from .semantic_encoder import SemanticEncoder
from .structure_encoder import StructureEncoder
from .fusion import AdaptiveFusion


def _require_pyg():
    try:
        from torch_geometric.data import Batch  # noqa: F401
    except Exception as e:
        raise ImportError("torch-geometric is required.") from e


class DecoupledMultiGATEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256, gnn_layers: int = 2, attn_heads: int = 4, dropout: float = 0.3, padding_idx: Optional[int] = None):
        super().__init__()
        self.dim = int(dim)
        self.embed = ConceptEmbedding(vocab_size=vocab_size, dim=dim, padding_idx=padding_idx, dropout=0.0)

        # φ(p_k) for Soft Import feature offsets (Eq.6)
        self.phi = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim, self.dim),
        )

        self.semantic = SemanticEncoder(dim=self.dim, num_heads=attn_heads, dropout=dropout)
        self.structure = StructureEncoder(dim=self.dim, num_layers=gnn_layers, num_heads=attn_heads, dropout=dropout)
        self.fusion = AdaptiveFusion(dim=self.dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        batch_data: "Batch",
        past_visit_memory: Optional[List[torch.Tensor]] = None,
        soft_offset_per_graph: Optional[torch.Tensor] = None,
        xi: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
        - h_G: (B,d) pooled visit embeddings (sum over original nodes)
        - z: (N,d) node embeddings after fusion
        """
        _require_pyg()

        node_ids = batch_data.node_ids
        edge_index = batch_data.edge_index
        batch = batch_data.batch
        orig_mask = batch_data.orig_mask

        x = self.embed(node_ids)

        # Optional Soft Import (batched): add xi * φ(p_k) to original nodes
        if soft_offset_per_graph is not None:
            B = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
            if soft_offset_per_graph.shape[0] != B:
                raise ValueError(f"soft_offset_per_graph shape {soft_offset_per_graph.shape}, expected (B,d) with B={B}")
            offs = self.phi(soft_offset_per_graph.to(x.device))  # (B,d)
            x = x.clone()
            for g in range(B):
                idx = (batch == g).nonzero(as_tuple=False).view(-1)
                m = orig_mask[idx]
                if m.any():
                    x[idx[m]] = x[idx[m]] + float(xi) * offs[g]

        x = self.dropout(x)

        h_sem = self.semantic(x, batch=batch, orig_mask=orig_mask)
        h_struct = self.structure(x, edge_index=edge_index, batch=batch, past_visit_memory=past_visit_memory)

        z = self.fusion(h_sem, h_struct)

        # Pool: sum over original nodes only (Eq.13)
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        h_G = torch.zeros((B, self.dim), dtype=z.dtype, device=z.device)
        for g in range(B):
            idx = (batch == g).nonzero(as_tuple=False).view(-1)
            m = orig_mask[idx]
            if m.any():
                h_G[g] = z[idx[m]].sum(dim=0)

        return h_G, z
