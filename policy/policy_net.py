"""
Policy network πθ(a|s) and value head Vϕ(s) for ReTA.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.backbone = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, self.action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if s.dim() == 1:
            s = s.unsqueeze(0)
        h = self.backbone(s)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, s: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, v = self.forward(s)
        dist = torch.distributions.Categorical(logits=logits)
        a = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        logp = dist.log_prob(a)
        return a, logp, v

    def evaluate_actions(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, v = self.forward(s)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(a)
        ent = dist.entropy()
        return logp, ent, v
