"""
Adaptive cross-view fusion (semantic + structural).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveFusion(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = int(dim)
        self.gate = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim, 1),
        )

    def forward(self, h_sem: torch.Tensor, h_struct: torch.Tensor) -> torch.Tensor:
        beta = torch.sigmoid(self.gate(torch.cat([h_sem, h_struct], dim=1)))  # (N,1)
        return beta * h_sem + (1.0 - beta) * h_struct
