"""
Stage 2: PPO policy learning + encoder refinement.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import List, Optional

import numpy as np
import torch

from reta.knowledge.pool import KnowledgePool
from reta.model.encoder import DecoupledMultiGATEncoder
from reta.model.predictor import NextVisitPredictor
from reta.model.visit_graph import build_pyg_data, graft_hard_import
from reta.model.embeddings import labels_to_multihot_from_ccs_tokens

from reta.policy.action import decode_action, SOFT, HARD
from reta.policy.state import StateEncoder
from reta.policy.policy_net import PolicyValueNet
from reta.policy.reward import RewardConfig, compute_paired_reward
from reta.policy.ppo import PPOConfig, PPOTrainer, RolloutBuffer

from .trainer import load_config, set_seed, EntityTokenMapper, template_to_subgraph, TrajectorySampler


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2 PPO training for ReTA policy.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--warmup_ckpt", type=str, default="checkpoints/warmup.pt")
    return p


def main():
    args = build_argparser().parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)

    sampler = TrajectorySampler(cfg.data.processed_path)
    icd_vocab_size = int(sampler.meta["icd_vocab_size"])
    ccs_vocab_size = int(sampler.meta["ccs_vocab_size"])
    vocab_size = icd_vocab_size + ccs_vocab_size

    pool = KnowledgePool.load_jsonl(cfg.knowledge.templates_jsonl)
    mapper = EntityTokenMapper.from_json(cfg.knowledge.entity_to_token_json)

    encoder = DecoupledMultiGATEncoder(
        vocab_size=vocab_size,
        dim=cfg.model.dim,
        gnn_layers=cfg.model.gnn_layers,
        attn_heads=cfg.model.attn_heads,
        dropout=cfg.model.dropout,
    ).to(device)
    predictor = NextVisitPredictor(in_dim=cfg.model.dim, num_labels=ccs_vocab_size, dropout=cfg.model.dropout).to(device)

    if args.warmup_ckpt and os.path.exists(args.warmup_ckpt):
        ck = torch.load(args.warmup_ckpt, map_location="cpu")
        encoder.load_state_dict(ck["encoder"], strict=False)
        predictor.load_state_dict(ck["predictor"], strict=False)
        print(f"[rl] loaded warmup ckpt: {args.warmup_ckpt}")

    K = cfg.knowledge.retrieval_K
    policy_net = PolicyValueNet(state_dim=2 * cfg.model.dim, action_dim=2 * K, hidden_dim=256, dropout=0.1).to(device)
    ppo = PPOTrainer(policy_net, PPOConfig(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    enc_opt = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    state_enc = StateEncoder(dim=cfg.model.dim).to(device)
    reward_cfg = RewardConfig(lambda1=1.0, lambda2=0.1)

    def make_lookup():
        w = encoder.embed.emb.weight.detach().cpu().numpy().astype(np.float32)
        n = np.linalg.norm(w, axis=1, keepdims=True) + 1e-12
        wn = w / n
        def lookup(tok: int) -> np.ndarray:
            return wn[int(tok)]
        return lookup

    def forward_loss(graph_data, soft_vec: Optional[torch.Tensor], y_tokens: List[int], dropout_off: bool) -> torch.Tensor:
        if dropout_off:
            encoder.eval(); predictor.eval()
        else:
            encoder.train(); predictor.train()

        from torch_geometric.data import Batch
        bg = Batch.from_data_list([graph_data]).to(device)

        soft_offset = soft_vec.view(1, -1).to(device) if soft_vec is not None else None
        with torch.set_grad_enabled(not dropout_off):
            h_G, _ = encoder(bg, past_visit_memory=None, soft_offset_per_graph=soft_offset, xi=cfg.train.soft_xi)
            logits = predictor(h_G)
            y = labels_to_multihot_from_ccs_tokens(y_tokens, ccs_vocab_size, icd_vocab_size, device=device).view(1, -1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        return loss

    for it in range(cfg.train.rl_iters):
        buffer = RolloutBuffer(device=device)
        stored_transitions = []

        patient_ids = sampler.sample_patients(cfg.train.rollout_patients)
        lookup = make_lookup()

        encoder.eval(); predictor.eval()
        with torch.no_grad():
            for pid in patient_ids:
                traj = sampler.get_patient_traj(pid)
                T = min(len(traj), cfg.train.max_visits_per_patient)
                if T < 2:
                    continue

                state_enc.reset()
                for t in range(T - 1):
                    visit = traj[t]
                    y_tokens = traj[t].get("label_ccs")
                    if y_tokens is None:
                        continue

                    node_ids = visit["node_ids"]
                    edge_index = torch.tensor(visit["edge_index"], dtype=torch.long)
                    orig_mask = torch.ones(len(node_ids), dtype=torch.bool)
                    g_raw = build_pyg_data(node_ids, edge_index, orig_mask=orig_mask).to(device)

                    # v_t from raw graph
                    from torch_geometric.data import Batch
                    h_G_raw, _ = encoder(Batch.from_data_list([g_raw]).to(device), past_visit_memory=None)
                    s_t = state_enc.step(h_G_raw[0])

                    # candidates
                    ret = pool.retrieve_topk(visit["icd_tokens"], lookup, K=K)
                    candidates = ret.template_ids
                    if len(candidates) < K:
                        if len(candidates) == 0:
                            continue
                        candidates = candidates + [candidates[-1]] * (K - len(candidates))

                    # policy action
                    a, logp, v = policy_net.act(s_t.to(device), deterministic=False)
                    act = decode_action(int(a.item()), candidates)
                    tmpl = pool.get_template(act.template_id)

                    soft_vec = None
                    g_aug = g_raw
                    added_nodes = 0
                    if act.mode == SOFT:
                        soft_vec = torch.tensor(tmpl.vector, dtype=torch.float32)
                    else:
                        sub = template_to_subgraph(tmpl.medoid.to_dict(), mapper)
                        if sub is not None:
                            before = int(g_raw.node_ids.numel())
                            g_aug = graft_hard_import(g_raw, sub)
                            after = int(g_aug.node_ids.numel())
                            added_nodes = max(0, after - before)

                    # paired reward
                    loss_raw = forward_loss(g_raw, None, y_tokens, dropout_off=True)
                    loss_edit = forward_loss(g_aug, soft_vec, y_tokens, dropout_off=True)
                    r_t = compute_paired_reward(loss_raw, loss_edit, is_hard=(act.mode == HARD),
                                                added_nodes=added_nodes, base_nodes=int(g_raw.node_ids.numel()), cfg=reward_cfg)

                    done = torch.tensor(1.0 if (t == T - 2) else 0.0, device=device)
                    buffer.add(s_t.to(device), a.to(device).view(()), r_t.to(device).view(()),
                               done, logp.to(device).view(()), v.to(device).view(()))

                    stored_transitions.append((g_aug.detach().cpu(), soft_vec, y_tokens))

        stats = ppo.update(buffer)

        # supervised refinement
        if stored_transitions:
            random.shuffle(stored_transitions)
            n_upd = min(len(stored_transitions), cfg.train.encoder_updates_per_iter * 128)
            total_loss = 0.0
            for i in range(n_upd):
                g_aug, soft_vec, y_tokens = stored_transitions[i]
                g_aug = g_aug.to(device)
                loss = forward_loss(g_aug, soft_vec, y_tokens, dropout_off=False)
                enc_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), cfg.train.grad_clip)
                enc_opt.step()
                total_loss += float(loss.item())
            sup_loss = total_loss / max(1, n_upd)
        else:
            sup_loss = float("nan")

        print(f"[rl] iter {it+1}/{cfg.train.rl_iters} "
              f"PPO(loss_pi={stats['loss_pi']:.4f}, loss_v={stats['loss_v']:.4f}, ent={stats['entropy']:.3f}, kl={stats['kl']:.4f}) "
              f"sup_loss={sup_loss:.4f}")

        os.makedirs("checkpoints", exist_ok=True)
        torch.save({"encoder": encoder.state_dict(), "predictor": predictor.state_dict(),
                    "policy": policy_net.state_dict(), "cfg": cfg},
                   f"checkpoints/rl_iter{it+1}.pt")


if __name__ == "__main__":
    main()
