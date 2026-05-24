"""Skip-aware ReTA action space.

For a retrieved candidate pool of size ``K``:
``A(t) = (P_sub(t) x {Soft, Hard}) union {Skip}``, giving ``2K+1`` actions.

Indexing convention:
- ``0 .. K-1``: Soft Import with candidate ``a_idx``
- ``K .. 2K-1``: Hard Import with candidate ``a_idx - K``
- ``2K``: Skip
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


SOFT = 0
HARD = 1
SKIP = 2


@dataclass(frozen=True)
class AugmentAction:
    template_id: Optional[int]
    mode: int        # 0=Soft, 1=Hard, 2=Skip
    template_pos: Optional[int]
    a_idx: int

    @property
    def is_skip(self) -> bool:
        return self.mode == SKIP


def decode_action(a_idx: int, candidates: Sequence[int]) -> AugmentAction:
    K = len(candidates)
    if K < 0:
        raise ValueError("K cannot be negative.")
    if a_idx < 0 or a_idx > 2 * K:
        raise ValueError(f"a_idx out of range: {a_idx}, expected [0, {2*K}].")

    if a_idx == 2 * K:
        return AugmentAction(template_id=None, mode=SKIP, template_pos=None, a_idx=int(a_idx))

    if K == 0:
        raise ValueError("Only the skip action is valid when candidates is empty.")

    mode = int(a_idx // K)
    pos = int(a_idx % K)
    template_id = int(candidates[pos])
    return AugmentAction(template_id=template_id, mode=mode, template_pos=pos, a_idx=int(a_idx))


def encode_action(template_pos: Optional[int], mode: int, K: int) -> int:
    if mode == SKIP:
        return int(2 * K)
    if mode not in (SOFT, HARD):
        raise ValueError("mode must be 0 (SOFT), 1 (HARD), or 2 (SKIP).")
    if template_pos is None:
        raise ValueError("template_pos is required for Soft/Hard actions.")
    if template_pos < 0 or template_pos >= K:
        raise ValueError("template_pos out of range.")
    return int(mode * K + template_pos)


def action_size(K: int) -> int:
    return int(2 * K + 1)
