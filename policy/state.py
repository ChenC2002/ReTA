"""
State encoder for ReTA policy (GRU history + current visit).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """GRU-based history encoder for policy state."""

    def __init__(self, dim: int = 256, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.gru = nn.GRU(
            input_size=self.dim,
            hidden_size=self.dim,
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self._h: Optional[torch.Tensor] = None  # (num_layers, 1, d)

    def reset(self, device: Optional[torch.device] = None):
        """Reset hidden state for a new patient trajectory."""
        self._h = None if device is None else None

    def step(self, v_t: torch.Tensor) -> torch.Tensor:
        """Update GRU with current visit embedding and return s_t (2d,)."""
        if v_t.dim() == 1:
            v_t = v_t.view(1, 1, -1)  # (1,1,d)
        elif v_t.dim() == 2:
            # expect (1,d)
            v_t = v_t.view(1, 1, -1)
        else:
            raise ValueError("StateEncoder.step expects (d,) or (1,d).")

        # history summary is hidden before consuming v_t
        if self._h is None:
            hist = torch.zeros((1, self.dim), dtype=v_t.dtype, device=v_t.device)
        else:
            hist = self._h[-1, 0].view(1, self.dim)

        _, h = self.gru(v_t, self._h)
        self._h = h.detach()

        cur = v_t.view(1, self.dim)
        s_t = torch.cat([hist, cur], dim=1).view(-1)  # (2d,)
        return s_t

    def forward(self, v_seq: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute states for a batch: v_seq (B,T,d) -> states (B,T,2d)."""
        if v_seq.dim() != 3:
            raise ValueError("v_seq must be (B,T,d).")
        B, T, d = v_seq.shape
        if d != self.dim:
            raise ValueError(f"v_seq dim {d} != expected {self.dim}.")

        if lengths is not None:
            lengths = lengths.to("cpu")
            packed = nn.utils.rnn.pack_padded_sequence(v_seq, lengths=lengths, batch_first=True, enforce_sorted=False)
            out_packed, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=T)
        else:
            out, _ = self.gru(v_seq)

        # out[:,t] = GRU(v_1:t); we need GRU(v_1:t-1)
        hist = torch.zeros((B, T, d), dtype=v_seq.dtype, device=v_seq.device)
        hist[:, 1:, :] = out[:, :-1, :]

        states = torch.cat([hist, v_seq], dim=-1)  # (B,T,2d)
        return states
