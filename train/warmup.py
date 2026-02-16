"""
Stage 1: Encoder warm-up (stochastic augment exposure).
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from reta.data.dataset import TrajectoryDataset, collate_transitions
from reta.knowledge.pool import KnowledgePool
from reta.model.encoder import DecoupledMultiGATEncoder
from reta.model.predictor import NextVisitPredictor
from reta.model.visit_graph import from_sample_dict, batch_graphs, graft_hard_import
from reta.model.embeddings import labels_to_multihot_from_ccs_tokens

from .trainer import load_config, set_seed, EntityTokenMapper, template_to_subgraph


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 1 warm-up training for encoder.")
    p.add_argument("--config", type=str, default=None)
    return p


def main():
    args = build_argparser().parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.train.seed)

    device = torch.device(cfg.train.device)

    ds = TrajectoryDataset(cfg.data.processed_path, split="all")
    dl = DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=lambda b: collate_transitions(b, num_labels=None),
    )

    pool = KnowledgePool.load_jsonl(cfg.knowledge.templates_jsonl)
    mapper = EntityTokenMapper.from_json(cfg.knowledge.entity_to_token_json)

    vocab_size = int(ds.meta["icd_vocab_size"]) + int(ds.meta["ccs_vocab_size"])
    encoder = DecoupledMultiGATEncoder(
        vocab_size=vocab_size,
        dim=cfg.model.dim,
        gnn_layers=cfg.model.gnn_layers,
        attn_heads=cfg.model.attn_heads,
        dropout=cfg.model.dropout,
    ).to(device)

    predictor = NextVisitPredictor(
        in_dim=cfg.model.dim,
        num_labels=int(ds.meta["ccs_vocab_size"]),
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

    for epoch in range(cfg.train.warmup_epochs):
        encoder.train()
        predictor.train()
        running, steps = 0.0, 0

        for batch in dl:
            graphs = [from_sample_dict({"node_ids": n, "edge_index": e})
                      for n, e in zip(batch["node_ids"], batch["edge_index"])]

            lookup = make_lookup()
            aug_graphs = []
            soft_offsets = []

            for i, g in enumerate(graphs):
                g = g.to(device)
                if random.random() >= cfg.train.exposure_prob:
                    aug_graphs.append(g)
                    soft_offsets.append(None)
                    continue

                visit_tokens = batch["icd_tokens"][i].tolist()
                ret = pool.retrieve_topk(visit_tokens, lookup, K=cfg.knowledge.retrieval_K)
                K = len(ret.template_ids)
                if K == 0:
                    aug_graphs.append(g); soft_offsets.append(None); continue

                # uniform over 2K
                a_idx = random.randrange(2 * K)
                mode = a_idx // K
                pos = a_idx % K
                tmpl_id = ret.template_ids[pos]
                tmpl = pool.get_template(tmpl_id)

                if mode == 0:
                    soft_offsets.append(torch.tensor(tmpl.vector, dtype=torch.float32))
                    aug_graphs.append(g)
                else:
                    sub = template_to_subgraph(tmpl.medoid.to_dict(), mapper)
                    if sub is None:
                        aug_graphs.append(g); soft_offsets.append(None)
                    else:
                        aug_graphs.append(graft_hard_import(g, sub))
                        soft_offsets.append(None)

            bg = batch_graphs(aug_graphs).to(device)

            B = len(aug_graphs)
            d = cfg.model.dim
            soft_offset_per_graph = None
            if any(x is not None for x in soft_offsets):
                mat = torch.zeros((B, d), dtype=torch.float32)
                for i, off in enumerate(soft_offsets):
                    if off is not None:
                        mat[i] = off
                soft_offset_per_graph = mat.to(device)

            h_G, _ = encoder(bg, past_visit_memory=None, soft_offset_per_graph=soft_offset_per_graph, xi=cfg.train.soft_xi)
            logits = predictor(h_G)

            targets = []
            for y_tokens in batch["labels"]:
                y = labels_to_multihot_from_ccs_tokens(
                    y_tokens,
                    num_ccs_labels=int(ds.meta["ccs_vocab_size"]),
                    icd_vocab_size=int(ds.meta["icd_vocab_size"]),
                    device=device,
                )
                targets.append(y)
            y = torch.stack(targets, dim=0)

            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), cfg.train.grad_clip)
            opt.step()

            running += float(loss.item())
            steps += 1

        print(f"[warmup] epoch {epoch+1}/{cfg.train.warmup_epochs} loss={running/max(steps,1):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"encoder": encoder.state_dict(), "predictor": predictor.state_dict(), "cfg": cfg}, "checkpoints/warmup.pt")
    print("[warmup] saved checkpoints/warmup.pt")


if __name__ == "__main__":
    main()
