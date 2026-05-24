"""
Stage 1: Encoder warm-up (stochastic augment exposure).
"""

from __future__ import annotations

import argparse
import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch

from reta.knowledge.pool import KnowledgePool
from reta.model.encoder import DecoupledMultiGATEncoder
from reta.model.predictor import NextVisitPredictor
from reta.model.visit_graph import build_pyg_data, batch_graphs, graft_hard_import
from reta.model.embeddings import labels_to_multihot_from_ccs_tokens
from reta.policy.action import decode_action, SOFT, HARD

from .trainer import (
    EntityTokenMapper,
    TrajectorySampler,
    load_config,
    set_seed,
    template_to_subgraph,
    template_vector_tensor,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 1 warm-up training for encoder.")
    p.add_argument("--config", type=str, default=None)
    return p


def main():
    args = build_argparser().parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)

    device = torch.device(cfg.train.device)

    sampler = TrajectorySampler(cfg.data.processed_path)

    pool = KnowledgePool.load_jsonl(cfg.knowledge.templates_jsonl)
    mapper = EntityTokenMapper.from_sources(
        sampler.vocab.get("name_to_token"),
        cfg.knowledge.entity_to_token_json,
    )

    vocab_size = int(sampler.meta["icd_vocab_size"]) + int(sampler.meta["ccs_vocab_size"])
    encoder = DecoupledMultiGATEncoder(
        vocab_size=vocab_size,
        dim=cfg.model.dim,
        gnn_layers=cfg.model.gnn_layers,
        attn_heads=cfg.model.attn_heads,
        dropout=cfg.model.dropout,
    ).to(device)

    predictor = NextVisitPredictor(
        in_dim=cfg.model.dim,
        num_labels=int(sampler.meta["ccs_vocab_size"]),
        dropout=cfg.model.dropout
    ).to(device)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    # retrieval lookup from current embedding weights
    def make_lookup():
        w = encoder.embed.emb.weight.detach().cpu().numpy().astype(np.float32)
        n = np.linalg.norm(w, axis=1, keepdims=True) + 1e-12
        wn = w / n
        def lookup(tok: int) -> np.ndarray:
            return wn[int(tok)]
        return lookup

    def build_visit_graph(visit):
        node_ids = visit["node_ids"]
        edge_index = torch.tensor(visit["edge_index"], dtype=torch.long)
        orig_mask = torch.ones(len(node_ids), dtype=torch.bool)
        code_set = set(int(x) for x in visit["icd_tokens"])
        code_mask = torch.tensor([int(nid) in code_set for nid in node_ids], dtype=torch.bool)
        return build_pyg_data(node_ids, edge_index, orig_mask=orig_mask, code_mask=code_mask)

    def maybe_augment(visit, g, lookup) -> Tuple[object, Optional[torch.Tensor]]:
        if random.random() >= cfg.train.exposure_prob:
            return g, None

        ret = pool.retrieve_topk(visit["icd_tokens"], lookup, K=cfg.knowledge.retrieval_K)
        K = len(ret.template_ids)
        if K == 0:
            return g, None

        act = decode_action(random.randrange(2 * K), ret.template_ids)
        tmpl = pool.get_template(act.template_id)

        if act.mode == SOFT:
            return g, template_vector_tensor(tmpl, cfg.model.dim)
        if act.mode == HARD:
            sub = template_to_subgraph(tmpl.medoid.to_dict(), mapper)
            if sub is not None:
                return graft_hard_import(g, sub), None
        return g, None

    def step_optimizer() -> None:
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), cfg.train.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

    for epoch in range(cfg.train.warmup_epochs):
        encoder.train()
        predictor.train()
        running, steps = 0.0, 0
        accum = 0
        update_every = max(1, int(cfg.data.batch_size))
        opt.zero_grad(set_to_none=True)

        patient_ids = list(sampler.patient_ids)
        random.shuffle(patient_ids)

        for pid in patient_ids:
            traj = sampler.get_patient_traj(pid)
            if len(traj) < 2:
                continue

            past_visit_embs: List[torch.Tensor] = []
            for t in range(len(traj) - 1):
                visit = traj[t]
                y_tokens = visit.get("label_ccs")
                if y_tokens is None:
                    continue

                lookup = make_lookup()
                g_raw = build_visit_graph(visit).to(device)
                g_aug, soft_vec = maybe_augment(visit, g_raw, lookup)
                bg = batch_graphs([g_aug]).to(device)

                mem = (
                    torch.stack(past_visit_embs, dim=0)
                    if past_visit_embs
                    else torch.zeros((0, cfg.model.dim), device=device)
                )
                soft_offset = soft_vec.view(1, -1).to(device) if soft_vec is not None else None

                h_G, _ = encoder(
                    bg,
                    past_visit_memory=[mem],
                    soft_offset_per_graph=soft_offset,
                    xi=cfg.train.soft_xi,
                )
                logits = predictor(h_G)

                y = labels_to_multihot_from_ccs_tokens(
                    y_tokens,
                    num_ccs_labels=int(sampler.meta["ccs_vocab_size"]),
                    icd_vocab_size=int(sampler.meta["icd_vocab_size"]),
                    device=device,
                ).view(1, -1)

                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
                (loss / update_every).backward()
                accum += 1
                past_visit_embs.append(h_G[0].detach())

                if accum >= update_every:
                    step_optimizer()
                    accum = 0

                running += float(loss.item())
                steps += 1

        if accum > 0:
            step_optimizer()

        print(f"[warmup] epoch {epoch+1}/{cfg.train.warmup_epochs} loss={running/max(steps,1):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"encoder": encoder.state_dict(), "predictor": predictor.state_dict(), "cfg": cfg}, "checkpoints/warmup.pt")
    print("[warmup] saved checkpoints/warmup.pt")


if __name__ == "__main__":
    main()
