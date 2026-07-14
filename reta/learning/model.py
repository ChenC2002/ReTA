"""Neural model components for ReTA.

The model uses a shared ICD/CCS concept embedding, independent semantic and
structural encoders, adaptive cross-view fusion, and a multi-label next-visit
prediction head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


__all__ = [
    "AdaptiveFusion",
    "ConceptEmbedding",
    "DecoupledMultiGATEncoder",
    "NextVisitPredictor",
    "SemanticEncoder",
    "StructureEncoder",
    "TemplateSubgraph",
    "batch_graphs",
    "build_pyg_data",
    "ccs_token_to_label_index",
    "graft_hard_import",
    "labels_to_multihot_from_ccs_tokens",
]


def ccs_token_to_label_index(ccs_token: torch.Tensor, icd_vocab_size: int) -> torch.Tensor:
    """Convert CCS token IDs in the shared namespace to label indices."""
    return ccs_token - int(icd_vocab_size)


def labels_to_multihot_from_ccs_tokens(
    ccs_tokens,
    num_ccs_labels: int,
    icd_vocab_size: int,
    device=None,
) -> torch.Tensor:
    """Build a multi-hot vector from a list or 1D tensor of CCS token IDs."""
    if isinstance(ccs_tokens, (list, tuple)):
        ccs_tokens = torch.tensor(ccs_tokens, dtype=torch.long, device=device)
    else:
        ccs_tokens = ccs_tokens.to(device=device)

    idx = ccs_token_to_label_index(ccs_tokens, icd_vocab_size)
    y = torch.zeros(num_ccs_labels, dtype=torch.float32, device=device)
    if idx.numel() > 0:
        if int(idx.min()) < 0 or int(idx.max()) >= int(num_ccs_labels):
            raise ValueError("CCS token ID is outside the configured label space.")
        y[idx] = 1.0
    return y


class ConceptEmbedding(nn.Module):
    """Shared embedding table for ICD, CCS, and knowledge-graph tokens."""

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
        """Embed token IDs into vectors of shape ``(*, dim)``."""
        x = self.emb(token_ids)
        return self.dropout(x)

    def load_pretrained(self, weights: torch.Tensor, strict: bool = True):
        """Load pretrained weights, allowing a partial load when non-strict."""
        if weights.shape != self.emb.weight.shape:
            msg = f"Pretrained embedding shape {tuple(weights.shape)} != {tuple(self.emb.weight.shape)}"
            if strict:
                raise ValueError(msg)
        with torch.no_grad():
            n = min(weights.shape[0], self.emb.weight.shape[0])
            d = min(weights.shape[1], self.emb.weight.shape[1])
            self.emb.weight[:n, :d].copy_(weights[:n, :d])


def _require_pyg_semantic() -> None:
    try:
        from torch_geometric.utils import to_dense_batch  # noqa: F401
    except Exception as e:
        raise ImportError("torch-geometric is required for semantic batching via to_dense_batch.") from e


class SemanticEncoder(nn.Module):
    """Feature-only self-attention encoder for the semantic channel."""

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
        """Return semantic node embeddings; non-observed nodes are zeroed."""
        _require_pyg_semantic()
        from torch_geometric.utils import to_dense_batch

        x_dense, mask = to_dense_batch(x, batch=batch)
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
        batch_size = x_dense.size(0)
        for graph_index in range(batch_size):
            idx = (batch == graph_index).nonzero(as_tuple=False).view(-1)
            graph_length = int(mask[graph_index].sum().item())
            h_sem[idx] = x_dense[graph_index, :graph_length]

        h_sem[~code_mask] = 0.0
        return h_sem


def _require_pyg_structure() -> None:
    try:
        from torch_geometric.nn import GATConv  # noqa: F401
    except Exception as e:
        raise ImportError("torch-geometric is required for GATConv.") from e


class StructureEncoder(nn.Module):
    """Graph-attention and temporal-attention structural encoder."""

    def __init__(self, dim: int = 256, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        _require_pyg_structure()
        from torch_geometric.nn import GATConv

        self.dim = int(dim)
        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(int(num_layers)):
            self.convs.append(
                GATConv(
                    self.dim,
                    self.dim,
                    heads=num_heads,
                    concat=False,
                    dropout=self.dropout,
                    add_self_loops=False,
                )
            )
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
        h = x
        for conv, layer_norm in zip(self.convs, self.norms):
            h2 = conv(h, edge_index)
            h2 = torch.dropout(h2, p=self.dropout, train=self.training)
            h = layer_norm(h + h2)
        h_intra = h

        if past_visit_memory is None:
            return h_intra

        h_struct = torch.zeros_like(h_intra)
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        for graph_index in range(batch_size):
            idx = (batch == graph_index).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            memory = past_visit_memory[graph_index] if graph_index < len(past_visit_memory) else None
            if memory is None or memory.numel() == 0:
                h_struct[idx] = h_intra[idx]
                continue

            memory = memory.to(h_intra.device)
            query = self.W_T(h_intra[idx])
            attention = torch.softmax(query @ memory.T, dim=1)
            h_inter = attention @ memory
            gate = torch.sigmoid(self.fuse_gate(torch.cat([h_intra[idx], h_inter], dim=1)))
            h_struct[idx] = gate * h_intra[idx] + (1.0 - gate) * h_inter

        return h_struct


class AdaptiveFusion(nn.Module):
    """Adaptively combine semantic and structural node representations."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = int(dim)
        self.gate = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim, 1),
        )

    def forward(self, h_sem: torch.Tensor, h_struct: torch.Tensor) -> torch.Tensor:
        beta = torch.sigmoid(self.gate(torch.cat([h_sem, h_struct], dim=1)))
        return beta * h_sem + (1.0 - beta) * h_struct


class NextVisitPredictor(nn.Module):
    """Multi-label CCS prediction head for the next visit."""

    def __init__(
        self,
        in_dim: int,
        num_labels: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_labels = int(num_labels)
        hidden_size = int(hidden_dim) if hidden_dim is not None else self.in_dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.num_labels),
        )

    def forward(self, h_G: torch.Tensor) -> torch.Tensor:
        return self.net(h_G)

    @staticmethod
    def loss_bce_with_logits(
        logits: torch.Tensor,
        targets: torch.Tensor,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    @staticmethod
    def predict_proba(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)


def _require_pyg_batch() -> None:
    try:
        from torch_geometric.data import Batch  # noqa: F401
    except Exception as e:
        raise ImportError("torch-geometric is required.") from e


class DecoupledMultiGATEncoder(nn.Module):
    """Decoupled semantic and structural multi-GAT encoder."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        gnn_layers: int = 2,
        attn_heads: int = 4,
        dropout: float = 0.3,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.embed = ConceptEmbedding(vocab_size=vocab_size, dim=dim, padding_idx=padding_idx, dropout=0.0)
        self.phi = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim, self.dim),
        )
        self.semantic = SemanticEncoder(dim=self.dim, num_heads=attn_heads, dropout=dropout)
        self.structure = StructureEncoder(
            dim=self.dim,
            num_layers=gnn_layers,
            num_heads=attn_heads,
            dropout=dropout,
        )
        self.fusion = AdaptiveFusion(dim=self.dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _effective_code_mask(code_mask: torch.Tensor, orig_mask: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Prefer observed ICD nodes, falling back per graph for older data."""
        code_mask = code_mask.bool()
        orig_mask = orig_mask.bool()
        out = code_mask.clone()
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        for graph_index in range(batch_size):
            idx = (batch == graph_index).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0 or out[idx].any():
                continue
            fallback = orig_mask[idx]
            if fallback.any():
                out[idx] = fallback
            else:
                out[idx] = True
        return out

    def forward(
        self,
        batch_data: "Batch",
        past_visit_memory: Optional[List[torch.Tensor]] = None,
        soft_offset_per_graph: Optional[torch.Tensor] = None,
        xi: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return pooled visit embeddings and fused node embeddings."""
        _require_pyg_batch()

        node_ids = batch_data.node_ids
        edge_index = batch_data.edge_index
        batch = batch_data.batch
        orig_mask = batch_data.orig_mask
        raw_code_mask = batch_data.code_mask if hasattr(batch_data, "code_mask") else orig_mask
        code_mask = self._effective_code_mask(raw_code_mask, orig_mask, batch)

        x = self.embed(node_ids)

        if soft_offset_per_graph is not None:
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
            if soft_offset_per_graph.shape[0] != batch_size:
                raise ValueError(
                    f"soft_offset_per_graph shape {soft_offset_per_graph.shape}, "
                    f"expected (B,d) with B={batch_size}"
                )
            if soft_offset_per_graph.shape[1] != self.dim:
                raise ValueError(
                    f"soft_offset_per_graph shape {soft_offset_per_graph.shape}, "
                    f"expected second dimension {self.dim}"
                )
            offsets = self.phi(soft_offset_per_graph.to(x.device))
            x = x.clone()
            for graph_index in range(batch_size):
                idx = (batch == graph_index).nonzero(as_tuple=False).view(-1)
                graph_code_mask = code_mask[idx]
                if graph_code_mask.any():
                    x[idx[graph_code_mask]] = x[idx[graph_code_mask]] + float(xi) * offsets[graph_index]

        x = self.dropout(x)
        h_sem = self.semantic(x, batch=batch, code_mask=code_mask)
        h_struct = self.structure(
            x,
            edge_index=edge_index,
            batch=batch,
            past_visit_memory=past_visit_memory,
        )
        z = self.fusion(h_sem, h_struct)

        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        h_G = torch.zeros((batch_size, self.dim), dtype=z.dtype, device=z.device)
        for graph_index in range(batch_size):
            idx = (batch == graph_index).nonzero(as_tuple=False).view(-1)
            graph_code_mask = code_mask[idx]
            if graph_code_mask.any():
                h_G[graph_index] = z[idx[graph_code_mask]].sum(dim=0)

        return h_G, z


def _require_pyg_graph() -> None:
    try:
        from torch_geometric.data import Batch, Data  # noqa: F401
        from torch_geometric.utils import coalesce  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "torch-geometric is required for graph encoding. "
            "Install torch-geometric and its dependencies."
        ) from exc


@dataclass
class TemplateSubgraph:
    """Compact token-space subgraph used for Hard Import."""

    node_ids: List[int]
    edge_index: torch.LongTensor


def build_pyg_data(
    node_ids: List[int],
    edge_index: torch.Tensor,
    orig_mask: Optional[torch.Tensor] = None,
    code_mask: Optional[torch.Tensor] = None,
):
    """Build a PyG visit graph with original-node and observed-code masks."""
    _require_pyg_graph()
    from torch_geometric.data import Data

    node_ids_t = torch.as_tensor(node_ids, dtype=torch.long)
    if node_ids_t.ndim != 1 or node_ids_t.numel() == 0:
        raise ValueError("node_ids must be a non-empty one-dimensional sequence.")
    if int(node_ids_t.min()) < 0:
        raise ValueError("node_ids must be non-negative.")
    if len(set(node_ids_t.tolist())) != node_ids_t.numel():
        raise ValueError("node_ids must be unique.")
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E).")
    if edge_index.numel() > 0:
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= len(node_ids):
            raise ValueError(
                "edge_index contains a non-local node position; regenerate "
                "processed.pt with the current preprocessing pipeline."
            )
    if orig_mask is None:
        orig_mask = torch.ones(len(node_ids), dtype=torch.bool)
    else:
        orig_mask = torch.as_tensor(orig_mask, dtype=torch.bool)
    if code_mask is None:
        code_mask = orig_mask.clone()
    else:
        code_mask = torch.as_tensor(code_mask, dtype=torch.bool)
    if orig_mask.ndim != 1 or orig_mask.numel() != len(node_ids):
        raise ValueError("orig_mask length must match node_ids.")
    if code_mask.ndim != 1 or code_mask.numel() != len(node_ids):
        raise ValueError("code_mask length must match node_ids.")
    return Data(
        node_ids=node_ids_t,
        edge_index=edge_index,
        orig_mask=orig_mask,
        code_mask=code_mask,
        num_nodes=len(node_ids),
    )


def batch_graphs(graphs: List["Data"]) -> "Batch":
    """Combine visit graphs into a PyG batch."""
    _require_pyg_graph()
    from torch_geometric.data import Batch

    return Batch.from_data_list(graphs)


def graft_hard_import(
    base: "Data",
    subgraph: TemplateSubgraph,
    add_self_loops: bool = False,
) -> "Data":
    """Graft a compact knowledge subgraph onto a visit graph."""
    _require_pyg_graph()
    from torch_geometric.data import Data
    from torch_geometric.utils import coalesce

    device = base.edge_index.device
    base_nodes = base.node_ids.tolist()
    base_index = {node_id: index for index, node_id in enumerate(base_nodes)}
    if not subgraph.node_ids:
        return base
    if any(isinstance(node_id, bool) or int(node_id) < 0 for node_id in subgraph.node_ids):
        raise ValueError("template node IDs must be non-negative integers.")
    if len(set(map(int, subgraph.node_ids))) != len(subgraph.node_ids):
        raise ValueError("template node IDs must be unique.")

    new_nodes = list(base_nodes)
    orig_mask = base.orig_mask.clone().to(device)
    code_mask = (
        base.code_mask.clone().to(device)
        if hasattr(base, "code_mask")
        else orig_mask.clone().to(device)
    )

    overlap = {
        local_index
        for local_index, node_id in enumerate(subgraph.node_ids)
        if node_id in base_index
    }
    if not overlap:
        return base

    sub_edges = subgraph.edge_index.to(device)
    if sub_edges.ndim != 2 or sub_edges.shape[0] != 2:
        raise ValueError("template edge_index must have shape (2, E).")
    if sub_edges.numel() > 0:
        if int(sub_edges.min()) < 0 or int(sub_edges.max()) >= len(subgraph.node_ids):
            raise ValueError("template edge_index references an unknown local node.")

    adjacency = {index: set() for index in range(len(subgraph.node_ids))}
    for left, right in sub_edges.t().tolist():
        adjacency[left].add(right)
        adjacency[right].add(left)
    reachable = set(overlap)
    frontier = list(overlap)
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current]:
            if neighbor not in reachable:
                reachable.add(neighbor)
                frontier.append(neighbor)

    sub_local_to_new = {}
    for local_index, node_id in enumerate(subgraph.node_ids):
        if local_index not in reachable:
            continue
        if node_id in base_index:
            sub_local_to_new[local_index] = base_index[node_id]
        else:
            sub_local_to_new[local_index] = len(new_nodes)
            new_nodes.append(int(node_id))
            orig_mask = torch.cat(
                [orig_mask, torch.tensor([False], dtype=torch.bool, device=device)]
            )
            code_mask = torch.cat(
                [code_mask, torch.tensor([False], dtype=torch.bool, device=device)]
            )

    if sub_edges.numel() > 0:
        retained_edges = [
            (left, right)
            for left, right in sub_edges.t().tolist()
            if left in reachable and right in reachable
        ]
        u = torch.tensor(
            [sub_local_to_new[left] for left, _ in retained_edges],
            dtype=torch.long,
            device=device,
        )
        v = torch.tensor(
            [sub_local_to_new[right] for _, right in retained_edges],
            dtype=torch.long,
            device=device,
        )
        remapped_edges = torch.stack([u, v], dim=0)
    else:
        remapped_edges = torch.zeros((2, 0), dtype=torch.long, device=device)

    edge_index = torch.cat([base.edge_index, remapped_edges], dim=1)
    if add_self_loops:
        node_range = torch.arange(len(new_nodes), device=device)
        edge_index = torch.cat(
            [edge_index, torch.stack([node_range, node_range], dim=0)],
            dim=1,
        )

    base_edge_set = {tuple(edge) for edge in base.edge_index.t().tolist()}
    candidate_edge_set = {tuple(edge) for edge in edge_index.t().tolist()}
    if len(new_nodes) == len(base_nodes) and candidate_edge_set <= base_edge_set:
        return base

    edge_index, _ = coalesce(
        edge_index,
        None,
        num_nodes=len(new_nodes),
    )
    return Data(
        node_ids=torch.tensor(new_nodes, dtype=torch.long, device=device),
        edge_index=edge_index,
        orig_mask=orig_mask,
        code_mask=code_mask,
        num_nodes=len(new_nodes),
    )
