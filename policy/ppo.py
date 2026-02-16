"""
PPO + GAE for ReTA policy learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import torch
import torch.nn as nn

from .policy_net import PolicyValueNet


@dataclass
class PPOConfig:
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatches: int = 4


class RolloutBuffer:
    def __init__(self, device: torch.device):
        self.device = device
        self.clear()

    def clear(self):
        self.s: List[torch.Tensor] = []
        self.a: List[torch.Tensor] = []
        self.r: List[torch.Tensor] = []
        self.done: List[torch.Tensor] = []
        self.logp: List[torch.Tensor] = []
        self.v: List[torch.Tensor] = []

    def add(self, s, a, r, done, logp, v):
        self.s.append(s.detach())
        self.a.append(a.detach())
        self.r.append(r.detach())
        self.done.append(done.detach())
        self.logp.append(logp.detach())
        self.v.append(v.detach())

    def stack(self) -> Dict[str, torch.Tensor]:
        return {
            "s": torch.stack(self.s, dim=0).to(self.device),
            "a": torch.stack(self.a, dim=0).to(self.device),
            "r": torch.stack(self.r, dim=0).to(self.device),
            "done": torch.stack(self.done, dim=0).to(self.device),
            "logp": torch.stack(self.logp, dim=0).to(self.device),
            "v": torch.stack(self.v, dim=0).to(self.device),
        }


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, cfg: PPOConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    rewards = rewards.float()
    values = values.float()
    dones = dones.float()

    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[0])

    next_value = torch.zeros_like(values[0])
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + cfg.gamma * next_value * mask - values[t]
        last_gae = delta + cfg.gamma * cfg.gae_lambda * mask * last_gae
        adv[t] = last_gae
        next_value = values[t]

    ret = adv + values
    return adv, ret


def minibatch_indices(n: int, minibatches: int, device: torch.device) -> Iterator[torch.Tensor]:
    idx = torch.randperm(n, device=device)
    mb = max(1, n // minibatches)
    for i in range(minibatches):
        j0 = i * mb
        j1 = n if i == minibatches - 1 else (i + 1) * mb
        yield idx[j0:j1]


class PPOTrainer:
    def __init__(self, net: PolicyValueNet, cfg: PPOConfig, lr: float = 1e-4, weight_decay: float = 1e-5):
        self.net = net
        self.cfg = cfg
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        data = buffer.stack()
        s, a, r, done, logp_old, v_old = data["s"], data["a"], data["r"], data["done"], data["logp"], data["v"]

        # flatten T×B -> N if needed
        if s.dim() == 3:
            T, B, D = s.shape
            s_f = s.reshape(T * B, D)
            a_f = a.reshape(T * B)
            logp_old_f = logp_old.reshape(T * B)
            v_old_f = v_old.reshape(T * B)
        else:
            s_f, a_f, logp_old_f, v_old_f = s, a, logp_old, v_old

        # compute advantages/returns using sequence info when available
        if r.dim() >= 2:
            adv, ret = compute_gae(r, v_old, done, self.cfg)
            adv_f = adv.reshape(-1)
            ret_f = ret.reshape(-1)
        else:
            adv_f = (r - v_old).detach()
            ret_f = (r).detach()

        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        N = s_f.shape[0]
        stats = {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0, "kl": 0.0}

        for _ in range(self.cfg.update_epochs):
            for mb_idx in minibatch_indices(N, self.cfg.minibatches, device=s_f.device):
                mb_s = s_f[mb_idx]
                mb_a = a_f[mb_idx]
                mb_adv = adv_f[mb_idx]
                mb_ret = ret_f[mb_idx]
                mb_logp_old = logp_old_f[mb_idx]

                logp, ent, v = self.net.evaluate_actions(mb_s, mb_a)

                ratio = torch.exp(logp - mb_logp_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps) * mb_adv
                loss_pi = -torch.min(surr1, surr2).mean()

                loss_v = 0.5 * (mb_ret - v).pow(2).mean()
                loss_ent = -ent.mean()

                loss = loss_pi + self.cfg.vf_coef * loss_v + self.cfg.ent_coef * loss_ent

                with torch.no_grad():
                    approx_kl = (mb_logp_old - logp).mean().item()

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

                stats["loss_pi"] += float(loss_pi.item())
                stats["loss_v"] += float(loss_v.item())
                stats["entropy"] += float(ent.mean().item())
                stats["kl"] += float(approx_kl)

        denom = float(self.cfg.update_epochs * self.cfg.minibatches)
        for k in stats:
            stats[k] /= denom
        return stats
