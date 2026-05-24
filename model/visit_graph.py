"""
Visit graph utilities and Hard Import grafting.

- Convert processed visit dict to PyG Data
- Hard Import: grafting a compact template subgraph onto the visit graph

``orig_mask`` tracks all nodes present before Hard Import; ``code_mask`` tracks
observed ICD visit codes only. The encoder uses ``code_mask`` for Soft Import,
semantic attention, and visit pooling, while CCS ancestors still participate in
structural message passing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


def _require_pyg():
    try:
        from torch_geometric.data import Data, Batch  # noqa: F401
        from torch_geometric.utils import coalesce  # noqa: F401
    except Exception as e:
        raise ImportError(
            "torch-geometric is required for graph encoding. Install torch-geometric and its dependencies."
        ) from e


@dataclass
class TemplateSubgraph:
    """Compact subgraph used for Hard Import.

    node_ids: token ids in the same namespace as visit graph
    edge_index: (2, E) indexing nodes inside this subgraph [0..num_nodes)
    """
    node_ids: List[int]
    edge_index: torch.LongTensor


def build_pyg_data(
    node_ids: List[int],
    edge_index: torch.Tensor,
    orig_mask: Optional[torch.Tensor] = None,
    code_mask: Optional[torch.Tensor] = None,
):
    _require_pyg()
    from torch_geometric.data import Data

    node_ids_t = torch.tensor(node_ids, dtype=torch.long)
    edge_index = edge_index.long()
    if orig_mask is None:
        orig_mask = torch.ones(len(node_ids), dtype=torch.bool)
    if code_mask is None:
        code_mask = orig_mask.clone()
    return Data(node_ids=node_ids_t, edge_index=edge_index, orig_mask=orig_mask, code_mask=code_mask)


def from_sample_dict(sample: Dict) -> "Data":
    """Convert dataset sample dict -> PyG Data."""
    _require_pyg()
    node_ids = sample["node_ids"].tolist() if hasattr(sample["node_ids"], "tolist") else list(sample["node_ids"])
    edge_index = sample["edge_index"]
    orig_mask = torch.ones(len(node_ids), dtype=torch.bool)
    icd_tokens = sample.get("icd_tokens")
    if icd_tokens is None:
        code_mask = orig_mask.clone()
    else:
        code_set = set(int(x) for x in (icd_tokens.tolist() if hasattr(icd_tokens, "tolist") else icd_tokens))
        code_mask = torch.tensor([int(nid) in code_set for nid in node_ids], dtype=torch.bool)
    return build_pyg_data(node_ids, edge_index, orig_mask=orig_mask, code_mask=code_mask)


def batch_graphs(graphs: List["Data"]) -> "Batch":
    _require_pyg()
    from torch_geometric.data import Batch
    return Batch.from_data_list(graphs)


def graft_hard_import(base: "Data", subgraph: TemplateSubgraph, add_self_loops: bool = False) -> "Data":
    """Graft a template subgraph onto base visit graph."""
    _require_pyg()
    from torch_geometric.data import Data
    from torch_geometric.utils import coalesce

    device = base.edge_index.device
    base_nodes = base.node_ids.tolist()
    base_index = {nid: i for i, nid in enumerate(base_nodes)}

    new_nodes = list(base_nodes)
    orig_mask = base.orig_mask.clone().to(device)
    code_mask = base.code_mask.clone().to(device) if hasattr(base, "code_mask") else orig_mask.clone().to(device)

    # map subgraph local idx -> new global idx
    sub_local_to_new = {}
    for j, nid in enumerate(subgraph.node_ids):
        if nid in base_index:
            sub_local_to_new[j] = base_index[nid]
        else:
            sub_local_to_new[j] = len(new_nodes)
            new_nodes.append(int(nid))
            orig_mask = torch.cat([orig_mask, torch.tensor([False], dtype=torch.bool, device=device)], dim=0)
            code_mask = torch.cat([code_mask, torch.tensor([False], dtype=torch.bool, device=device)], dim=0)

    # remap subgraph edges
    se = subgraph.edge_index.to(device)
    if se.numel() > 0:
        u = torch.tensor([sub_local_to_new[int(x)] for x in se[0].tolist()], device=device)
        v = torch.tensor([sub_local_to_new[int(x)] for x in se[1].tolist()], device=device)
        sub_ei = torch.stack([u, v], dim=0)
    else:
        sub_ei = torch.zeros((2, 0), dtype=torch.long, device=device)

    edge_index = torch.cat([base.edge_index, sub_ei], dim=1)

    if add_self_loops:
        n = len(new_nodes)
        sl = torch.arange(n, device=device)
        edge_index = torch.cat([edge_index, torch.stack([sl, sl], dim=0)], dim=1)

    edge_index, _ = coalesce(edge_index, None, m=len(new_nodes), n=len(new_nodes))

    return Data(
        node_ids=torch.tensor(new_nodes, dtype=torch.long, device=device),
        edge_index=edge_index,
        orig_mask=orig_mask,
        code_mask=code_mask,
    )
