"""
Training utilities and shared helpers.

This module centralizes:
- config loading
- seeding
- mapping knowledge templates to token-space subgraphs for Hard Import
- sampling full patient trajectories for RL rollouts

Entry points:
- warmup.py : Stage 1 encoder warm-up
- rl_train.py: Stage 2 REINFORCE policy learning + encoder refinement
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    import yaml
except Exception:
    yaml = None

from reta.model.visit_graph import TemplateSubgraph


@dataclass
class DataConfig:
    processed_path: str = "data/processed/processed.pt"
    batch_size: int = 32
    num_workers: int = 0


@dataclass
class KnowledgeConfig:
    templates_jsonl: str = "knowledge/templates.jsonl"
    retrieval_K: int = 20
    retrieval_alpha: float = 0.2
    # Optional JSON for external KG ids; ICD:/CCS: ids come from processed.pt.
    entity_to_token_json: Optional[str] = None


@dataclass
class ModelConfig:
    dim: int = 256
    gnn_layers: int = 2
    attn_heads: int = 4
    dropout: float = 0.3


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # warmup
    warmup_epochs: int = 30
    exposure_prob: float = 0.3
    soft_xi: float = 0.5

    # RL
    rl_iters: int = 50
    rollout_patients: int = 64
    max_visits_per_patient: int = 50
    encoder_updates_per_iter: int = 1
    policy_lr: float = 1e-5
    gamma: float = 0.95
    baseline_decay: float = 0.99
    policy_entropy_coef: float = 0.0
    reward_lambda1: float = 1.0
    reward_lambda2: float = 0.1
    utility_decay: float = 0.95


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: Optional[str] = None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    if yaml is None:
        raise ImportError("pyyaml is required to load config files.")
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}

    def update(dc, sub):
        if sub is None:
            return
        for k, v in sub.items():
            if hasattr(dc, k):
                setattr(dc, k, v)

    update(cfg.data, d.get("data"))
    update(cfg.knowledge, d.get("knowledge"))
    update(cfg.model, d.get("model"))
    update(cfg.train, d.get("train"))
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EntityTokenMapper:
    """Map canonical entity ids (string) to token ids (int)."""

    def __init__(self, mapping: Optional[Dict[str, int]] = None):
        self.mapping = mapping or {}

    @staticmethod
    def from_json(path: Optional[str]) -> "EntityTokenMapper":
        if path is None:
            return EntityTokenMapper({})
        with open(path, "r", encoding="utf-8") as f:
            mp = json.load(f)
        return EntityTokenMapper({str(k): int(v) for k, v in mp.items()})

    @staticmethod
    def from_sources(
        name_to_token: Optional[Dict[str, int]] = None,
        entity_to_token_json: Optional[str] = None,
    ) -> "EntityTokenMapper":
        """Build a mapper from processed ontology tokens plus optional KG ids."""
        mapping: Dict[str, int] = {}
        if name_to_token is not None:
            mapping.update({str(k): int(v) for k, v in name_to_token.items()})
        if entity_to_token_json is not None:
            with open(entity_to_token_json, "r", encoding="utf-8") as f:
                extra = json.load(f)
            mapping.update({str(k): int(v) for k, v in extra.items()})
        return EntityTokenMapper(mapping)

    def to_token(self, entity_id: str) -> Optional[int]:
        return self.mapping.get(str(entity_id))


def template_to_subgraph(template_medoid: Dict[str, Any], mapper: EntityTokenMapper) -> Optional[TemplateSubgraph]:
    """GroundedTemplate dict -> TemplateSubgraph (token ids)."""
    nodes = template_medoid.get("subgraph_nodes", [])
    edges = template_medoid.get("subgraph_edges", [])

    token_nodes: List[int] = []
    node_index: Dict[int, int] = {}

    for n in nodes:
        tok = n if isinstance(n, int) else mapper.to_token(n)
        if tok is None:
            continue
        tok = int(tok)
        if tok not in node_index:
            node_index[tok] = len(token_nodes)
            token_nodes.append(tok)

    if len(token_nodes) == 0:
        return None

    ei_u: List[int] = []
    ei_v: List[int] = []
    for u, v in edges:
        tu = u if isinstance(u, int) else mapper.to_token(u)
        tv = v if isinstance(v, int) else mapper.to_token(v)
        if tu is None or tv is None:
            continue
        tu, tv = int(tu), int(tv)
        if tu not in node_index or tv not in node_index:
            continue
        iu, iv = node_index[tu], node_index[tv]
        if iu == iv:
            continue
        # undirected
        ei_u += [iu, iv]
        ei_v += [iv, iu]

    edge_index = torch.tensor([ei_u, ei_v], dtype=torch.long) if len(ei_u) else torch.zeros((2, 0), dtype=torch.long)
    return TemplateSubgraph(node_ids=token_nodes, edge_index=edge_index)


def template_vector_tensor(template: Any, dim: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Return a template vector padded/truncated to the encoder dimension."""
    v = torch.tensor(template.vector, dtype=torch.float32, device=device)
    dim = int(dim)
    if v.numel() == dim:
        return v
    if v.numel() > dim:
        return v[:dim]
    pad = torch.zeros(dim - v.numel(), dtype=v.dtype, device=v.device)
    return torch.cat([v, pad], dim=0)


class TrajectorySampler:
    """Sample full patient trajectories from processed.pt for RL rollouts."""

    def __init__(self, processed_path: str):
        data = torch.load(processed_path, map_location="cpu")
        self.trajectories = data["trajectories"]
        self.meta = data["meta"]
        self.vocab = data.get("vocab", {})
        self.patient_ids = list(self.trajectories.keys())

    def sample_patients(self, n: int) -> List[str]:
        n = min(int(n), len(self.patient_ids))
        return random.sample(self.patient_ids, n)

    def get_patient_traj(self, patient_id: str) -> List[Dict[str, Any]]:
        return self.trajectories[str(patient_id)]


# -------------------------
# Orchestration CLI
# -------------------------

def _run_module_main(mod_name: str, argv: list):
    """Run a module's main() with a temporary sys.argv."""
    import importlib
    import sys

    old = sys.argv
    try:
        sys.argv = argv
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, "main"):
            raise AttributeError(f"{mod_name} has no main()")
        mod.main()
    finally:
        sys.argv = old


def main_cli():
    import argparse

    p = argparse.ArgumentParser(description="Training orchestrator for ReTA")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config")
    p.add_argument("--stage", type=str, default="all", choices=["warmup", "rl", "all"], help="Which stage to run")
    p.add_argument("--warmup_ckpt", type=str, default="checkpoints/warmup.pt")
    args = p.parse_args()

    if args.stage in ("warmup", "all"):
        argv = ["warmup"]
        if args.config:
            argv += ["--config", str(args.config)]
        _run_module_main("reta.train.warmup", argv)

    if args.stage in ("rl", "all"):
        argv = ["rl_train"]
        if args.config:
            argv += ["--config", str(args.config)]
        argv += ["--warmup_ckpt", str(args.warmup_ckpt)]
        _run_module_main("reta.train.rl_train", argv)


if __name__ == "__main__":
    main_cli()
