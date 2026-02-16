"""
Paired reward for policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RewardConfig:
    lambda1: float = 1.0
    lambda2: float = 0.1


def compute_paired_reward(
    loss_raw: torch.Tensor,
    loss_edit: torch.Tensor,
    is_hard: bool,
    added_nodes: int = 0,
    base_nodes: int = 1,
    cfg: Optional[RewardConfig] = None,
) -> torch.Tensor:
    if cfg is None:
        cfg = RewardConfig()
    base_nodes = max(int(base_nodes), 1)

    delta = loss_raw - loss_edit
    penalty = 0.0
    if is_hard:
        penalty = cfg.lambda2 * (float(added_nodes) / float(base_nodes))

    r = cfg.lambda1 * delta - penalty
    return r
