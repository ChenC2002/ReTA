"""REINFORCE trainer with a running-mean baseline.

Uses discounted trajectory returns, baseline subtraction for variance reduction,
optional entropy regularization, and gradient clipping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from .policy_net import PolicyNet


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
    out: List[torch.Tensor] = []
    running = torch.zeros_like(rewards[-1])
    for r in reversed(rewards):
        running = r + float(gamma) * running
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
            return {"loss_policy": 0.0, "entropy": 0.0, "return_mean": 0.0, "baseline": self.baseline.value}

        logps: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        returns: List[torch.Tensor] = []

        for traj in buffer.trajectories:
            rewards = [step["reward"].float() for step in traj]
            traj_returns = discounted_returns(rewards, gamma=self.cfg.gamma)
            for step, ret in zip(traj, traj_returns):
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
