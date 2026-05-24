"""Categorical policy network ``pi_theta(a | s)`` for ReTA."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
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

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if s.dim() == 1:
            s = s.unsqueeze(0)
        h = self.backbone(s)
        return self.policy_head(h)

    def distribution(self, s: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.forward(s)
        return torch.distributions.Categorical(logits=logits)

    def act(self, s: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(s)
        dist = torch.distributions.Categorical(logits=logits)
        a = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        logp = dist.log_prob(a)
        ent = dist.entropy()
        return a, logp, ent

    def evaluate_actions(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(s)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(a)
        ent = dist.entropy()
        return logp, ent
