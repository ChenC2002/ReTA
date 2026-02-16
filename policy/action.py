"""
Action index in [0,2K):
- template_pos = a_idx % K
- mode = a_idx // K  (0=Soft, 1=Hard)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


SOFT = 0
HARD = 1


@dataclass(frozen=True)
class AugmentAction:
    template_id: int
    mode: int        # 0=Soft, 1=Hard
    template_pos: int
    a_idx: int


def decode_action(a_idx: int, candidates: Sequence[int]) -> AugmentAction:
    K = len(candidates)
    if K == 0:
        raise ValueError("candidates is empty; cannot decode action.")
    if a_idx < 0 or a_idx >= 2 * K:
        raise ValueError(f"a_idx out of range: {a_idx}, expected [0, {2*K}).")

    mode = int(a_idx // K)
    pos = int(a_idx % K)
    template_id = int(candidates[pos])
    return AugmentAction(template_id=template_id, mode=mode, template_pos=pos, a_idx=int(a_idx))


def encode_action(template_pos: int, mode: int, K: int) -> int:
    if mode not in (SOFT, HARD):
        raise ValueError("mode must be 0 (SOFT) or 1 (HARD).")
    if template_pos < 0 or template_pos >= K:
        raise ValueError("template_pos out of range.")
    return int(mode * K + template_pos)


def action_size(K: int) -> int:
    return int(2 * K)
