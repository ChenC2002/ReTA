"""
Graph utilities.

These helpers are shared across data preprocessing, augmentation, and model encoding.
All graphs are treated as *untyped* adjacency by default.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def to_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    """Make an edge_index undirected by adding reversed edges."""
    if edge_index.numel() == 0:
        return edge_index
    rev = edge_index.flip(0)
    return torch.cat([edge_index, rev], dim=1)


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Add self loops (i,i) for all nodes."""
    device = edge_index.device
    sl = torch.arange(int(num_nodes), device=device)
    sl_ei = torch.stack([sl, sl], dim=0)
    return torch.cat([edge_index, sl_ei], dim=1)


def coalesce_edge_index(edge_index: torch.Tensor, num_nodes: Optional[int] = None) -> torch.Tensor:
    """Remove duplicate edges and sort.

    If torch_geometric is available, use `torch_geometric.utils.coalesce`.
    Otherwise, fall back to a pure torch implementation.
    """
    if edge_index.numel() == 0:
        return edge_index

    try:
        from torch_geometric.utils import coalesce
        n = int(num_nodes) if num_nodes is not None else None
        if n is None:
            n = int(edge_index.max().item()) + 1
        edge_index2, _ = coalesce(edge_index, None, m=n, n=n)
        return edge_index2
    except Exception:
        if num_nodes is None:
            num_nodes = int(edge_index.max().item()) + 1
        key = edge_index[0] * int(num_nodes) + edge_index[1]
        uniq = torch.unique(key, sorted=True)
        row = torch.div(uniq, int(num_nodes), rounding_mode="floor")
        col = uniq % int(num_nodes)
        return torch.stack([row, col], dim=0)


def remap_edge_index(edge_index: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    """Remap edge_index using a node-id mapping.

    Parameters
    ----------
    edge_index: (2,E) with indices in [0, N)
    mapping: (N,) long tensor, new_index = mapping[old_index]

    Returns
    -------
    (2,E) remapped edge_index
    """
    if edge_index.numel() == 0:
        return edge_index
    return torch.stack([mapping[edge_index[0]], mapping[edge_index[1]]], dim=0)


def edge_stats(edge_index: torch.Tensor) -> Tuple[int, int]:
    """Return (#nodes_estimate, #edges)."""
    if edge_index.numel() == 0:
        return 0, 0
    n = int(edge_index.max().item()) + 1
    e = int(edge_index.size(1))
    return n, e
