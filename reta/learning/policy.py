"""Policy learning components for retrieval-aware trajectory augmentation.

This module keeps the action space, policy state, policy network, paired reward,
and REINFORCE update together because they form one training subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


SOFT = 0
HARD = 1
SKIP = 2


@dataclass(frozen=True)
class AugmentAction:
    """Decoded action from the skip-aware ``2K + 1`` action space."""

    template_id: Optional[int]
    mode: int
    template_pos: Optional[int]
    a_idx: int

    @property
    def is_skip(self) -> bool:
        return self.mode == SKIP


def decode_action(a_idx: int, candidates: Sequence[int]) -> AugmentAction:
    """Decode an action index into a candidate, import mode, or skip."""
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
    """Encode a candidate position and import mode as an action index."""
    if K < 0:
        raise ValueError("K cannot be negative.")
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
    """Return the size of the action space for ``K`` candidates."""
    if K < 0:
        raise ValueError("K cannot be negative.")
    return int(2 * K + 1)


def valid_action_mask(
    candidate_count: int,
    K: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Mask unused candidate slots while always retaining the Skip action."""

    candidate_count = int(candidate_count)
    K = int(K)
    if K < 0 or candidate_count < 0 or candidate_count > K:
        raise ValueError("candidate_count must be in [0, K].")
    mask = torch.zeros(action_size(K), dtype=torch.bool, device=device)
    mask[:candidate_count] = True
    mask[K : K + candidate_count] = True
    mask[2 * K] = True
    return mask


def decode_policy_action(a_idx: int, candidates: Sequence[int], K: int) -> AugmentAction:
    """Decode an action from a fixed ``2K+1`` policy without padding candidates."""

    a_idx = int(a_idx)
    K = int(K)
    if K < 0 or len(candidates) > K:
        raise ValueError("candidate count must be in [0, K].")
    if a_idx < 0 or a_idx > 2 * K:
        raise ValueError(f"a_idx out of range: {a_idx}, expected [0, {2*K}].")
    if a_idx == 2 * K:
        return AugmentAction(None, SKIP, None, a_idx)
    if K == 0:
        raise ValueError("Only the skip action is valid when K is zero.")
    mode = a_idx // K
    pos = a_idx % K
    if pos >= len(candidates):
        raise ValueError(f"action {a_idx} selects unavailable candidate slot {pos}.")
    return AugmentAction(int(candidates[pos]), int(mode), int(pos), a_idx)


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
        self._h: Optional[torch.Tensor] = None

    def reset(self, device: Optional[torch.device] = None):
        """Reset hidden state for a new patient trajectory."""
        self._h = None

    def step(self, v_t: torch.Tensor) -> torch.Tensor:
        """Update the GRU with one visit and return its history/current state."""
        if v_t.dim() == 1:
            v_t = v_t.view(1, 1, -1)
        elif v_t.dim() == 2:
            v_t = v_t.view(1, 1, -1)
        else:
            raise ValueError("StateEncoder.step expects (d,) or (1,d).")

        if self._h is None:
            hist = torch.zeros((1, self.dim), dtype=v_t.dtype, device=v_t.device)
        else:
            hist = self._h[-1, 0].view(1, self.dim)

        _, h = self.gru(v_t, self._h)
        self._h = h

        cur = v_t.view(1, self.dim)
        return torch.cat([hist, cur], dim=1).view(-1)

    def forward(self, v_seq: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute history/current states for a ``(B, T, d)`` visit batch."""
        if v_seq.dim() != 3:
            raise ValueError("v_seq must be (B,T,d).")
        B, T, d = v_seq.shape
        if d != self.dim:
            raise ValueError(f"v_seq dim {d} != expected {self.dim}.")

        if lengths is not None:
            lengths = lengths.to("cpu")
            packed = nn.utils.rnn.pack_padded_sequence(
                v_seq,
                lengths=lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            out_packed, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=T)
        else:
            out, _ = self.gru(v_seq)

        hist = torch.zeros((B, T, d), dtype=v_seq.dtype, device=v_seq.device)
        hist[:, 1:, :] = out[:, :-1, :]
        return torch.cat([hist, v_seq], dim=-1)


class TemplateUtilityTracker:
    """Exponential running reward utility for each template id."""

    def __init__(self, decay: float = 0.95):
        self.decay = float(decay)
        self.values: Dict[int, float] = {}

    def get(self, template_id: int) -> float:
        return float(self.values.get(int(template_id), 0.0))

    def vector(self, candidates: Sequence[int], K: int, device: Optional[torch.device] = None) -> torch.Tensor:
        vals = [self.get(int(tid)) for tid in list(candidates)[: int(K)]]
        if len(vals) < int(K):
            vals.extend([0.0] * (int(K) - len(vals)))
        return torch.tensor(vals, dtype=torch.float32, device=device)

    def update(self, template_id: int, reward: float) -> None:
        tid = int(template_id)
        old = self.values.get(tid, 0.0)
        self.values[tid] = self.decay * old + (1.0 - self.decay) * float(reward)


def build_policy_state(
    history_current_state: torch.Tensor,
    uncertainty: Union[torch.Tensor, float],
    candidate_utilities: torch.Tensor,
) -> torch.Tensor:
    """Join history/current state, uncertainty, and candidate utilities."""
    s = history_current_state.view(-1)
    if not torch.is_tensor(uncertainty):
        uncertainty = torch.tensor(float(uncertainty), dtype=s.dtype, device=s.device)
    u = uncertainty.to(dtype=s.dtype, device=s.device).view(1)
    util = candidate_utilities.to(dtype=s.dtype, device=s.device).view(-1)
    return torch.cat([s, u, util], dim=0)


class PolicyNet(nn.Module):
    """Categorical policy network ``pi_theta(a | s)``."""

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

    def _masked_logits(
        self,
        s: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.forward(s)
        if action_mask is None:
            return logits
        mask = action_mask.to(device=logits.device, dtype=torch.bool)
        if mask.shape[-1] != self.action_dim:
            raise ValueError(
                f"action mask has width {mask.shape[-1]}, expected {self.action_dim}."
            )
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[0] not in (1, logits.shape[0]):
            raise ValueError("action mask batch dimension is incompatible with policy state.")
        if not mask.any(dim=-1).all():
            raise ValueError("each policy state must retain at least one valid action.")
        return logits.masked_fill(~mask, float("-inf"))

    def distribution(
        self,
        s: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Categorical:
        logits = self._masked_logits(s, action_mask)
        return torch.distributions.Categorical(logits=logits)

    def act(
        self,
        s: torch.Tensor,
        deterministic: bool = False,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self._masked_logits(s, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        a = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        logp = dist.log_prob(a)
        ent = dist.entropy()
        return a, logp, ent

    def evaluate_actions(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self._masked_logits(s, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(a)
        ent = dist.entropy()
        return logp, ent


@dataclass
class RewardConfig:
    lambda1: float = 1.0
    lambda2: float = 0.1


def compute_paired_reward(
    loss_raw: torch.Tensor,
    loss_edit: torch.Tensor,
    is_hard: bool,
    is_skip: bool = False,
    added_nodes: int = 0,
    base_nodes: int = 1,
    cfg: Optional[RewardConfig] = None,
) -> torch.Tensor:
    """Reward an edit by its paired loss improvement and size penalty."""
    if cfg is None:
        cfg = RewardConfig()
    if is_skip:
        return torch.zeros_like(loss_raw)
    base_nodes = max(int(base_nodes), 1)

    delta = loss_raw - loss_edit
    penalty = 0.0
    if is_hard:
        penalty = cfg.lambda2 * (float(added_nodes) / float(base_nodes))

    return cfg.lambda1 * delta - penalty


@dataclass
class ReinforceConfig:
    gamma: float = 0.95
    baseline_decay: float = 0.99
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5


class ReinforceBuffer:
    def __init__(self):
        self.trajectories: List[List[Dict[str, torch.Tensor]]] = []

    def clear(self) -> None:
        self.trajectories.clear()

    def add_trajectory(self, steps: Sequence[Dict[str, torch.Tensor]]) -> None:
        if steps:
            self.trajectories.append(list(steps))

    def __len__(self) -> int:
        return sum(len(t) for t in self.trajectories)


def discounted_returns(rewards: Sequence[torch.Tensor], gamma: float) -> List[torch.Tensor]:
    """Compute discounted returns for one trajectory."""
    out: List[torch.Tensor] = []
    running = torch.zeros_like(rewards[-1])
    for reward in reversed(rewards):
        running = reward + float(gamma) * running
        out.append(running)
    out.reverse()
    return out


class RunningMeanBaseline:
    def __init__(self, decay: float = 0.99):
        self.decay = float(decay)
        self.value = 0.0
        self.initialized = False

    def update(self, returns: torch.Tensor) -> float:
        mean = float(returns.detach().mean().item())
        if not self.initialized:
            self.value = mean
            self.initialized = True
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * mean
        return self.value


class ReinforceTrainer:
    def __init__(
        self,
        net: PolicyNet,
        cfg: ReinforceConfig,
        lr: float = 1e-5,
        weight_decay: float = 1e-5,
        extra_parameters: Optional[Iterable[torch.nn.Parameter]] = None,
    ):
        self.net = net
        self.cfg = cfg
        params = list(self.net.parameters())
        if extra_parameters is not None:
            params.extend(list(extra_parameters))
        self.params = params
        self.opt = torch.optim.Adam(self.params, lr=lr, weight_decay=weight_decay)
        self.baseline = RunningMeanBaseline(decay=cfg.baseline_decay)

    def update(self, buffer: ReinforceBuffer) -> Dict[str, float]:
        if len(buffer) == 0:
            return {
                "loss_policy": 0.0,
                "entropy": 0.0,
                "return_mean": 0.0,
                "baseline": self.baseline.value,
            }

        logps: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        returns: List[torch.Tensor] = []

        for trajectory in buffer.trajectories:
            rewards = [step["reward"].float() for step in trajectory]
            trajectory_returns = discounted_returns(rewards, gamma=self.cfg.gamma)
            for step, ret in zip(trajectory, trajectory_returns):
                logps.append(step["logp"])
                entropies.append(step["entropy"])
                returns.append(ret.detach())

        logp_t = torch.stack(logps).view(-1)
        ent_t = torch.stack(entropies).view(-1)
        ret_t = torch.stack(returns).view(-1)

        baseline = self.baseline.update(ret_t)
        adv = ret_t - baseline
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        loss_policy = -(adv.detach() * logp_t).mean()
        loss_entropy = -ent_t.mean()
        loss = loss_policy + float(self.cfg.entropy_coef) * loss_entropy

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.cfg.max_grad_norm)
        self.opt.step()

        return {
            "loss_policy": float(loss_policy.item()),
            "entropy": float(ent_t.mean().item()),
            "return_mean": float(ret_t.mean().item()),
            "baseline": float(baseline),
        }


__all__ = [
    "SOFT",
    "HARD",
    "SKIP",
    "AugmentAction",
    "decode_action",
    "decode_policy_action",
    "encode_action",
    "action_size",
    "valid_action_mask",
    "StateEncoder",
    "TemplateUtilityTracker",
    "build_policy_state",
    "PolicyNet",
    "RewardConfig",
    "compute_paired_reward",
    "ReinforceConfig",
    "ReinforceBuffer",
    "discounted_returns",
    "RunningMeanBaseline",
    "ReinforceTrainer",
]
