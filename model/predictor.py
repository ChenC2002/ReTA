"""
Next-visit CCS prediction head (multi-label).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class NextVisitPredictor(nn.Module):
    def __init__(self, in_dim: int, num_labels: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_labels = int(num_labels)
        h = int(hidden_dim) if hidden_dim is not None else self.in_dim

        self.net = nn.Sequential(
            nn.Linear(self.in_dim, h),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h, self.num_labels),
        )

    def forward(self, h_G: torch.Tensor) -> torch.Tensor:
        return self.net(h_G)

    @staticmethod
    def loss_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    @staticmethod
    def predict_proba(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)
