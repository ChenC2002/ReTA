"""
Stage 1: Encoder warm-up with stochastic augmentation exposure.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
import random
from typing import List, Optional, Tuple

import torch

from reta.knowledge.pool import KnowledgePool
from .runtime import (
    EntityTokenMapper,
    PoolAlignedTokenIndex,
    TrajectorySampler,
    build_checkpoint_contract,
    encoder_code_embedding_lookup,
    load_config,
    resolve_device,
    setup_logger,
    set_seed,
    template_to_subgraph,
    template_vector_tensor,
    visit_retrieval_tokens,
)
from .model import (
    DecoupledMultiGATEncoder,
    NextVisitPredictor,
    batch_graphs,
    build_pyg_data,
    graft_hard_import,
    labels_to_multihot_from_ccs_tokens,
)
from .policy import HARD, SOFT, decode_action


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 1 warm-up training for encoder.")
    p.add_argument("--config", type=str, default=None)
    return p


def main():
    args = build_argparser().parse_args()
    cfg = load_config(args.config)
    logger = setup_logger(cfg.outputs.logs_dir, name="warmup")
    set_seed(cfg.train.seed)

    device = resolve_device(cfg.train.device)

    training_split = cfg.data.train_split if cfg.data.split_json else "all"
    sampler = TrajectorySampler(
        cfg.data.processed_path,
        split_json=cfg.data.split_json,
        split=training_split,
    )
    logger.info(f"training split={training_split} patients={len(sampler.patient_ids)}")

    pool = KnowledgePool.load_jsonl(cfg.knowledge.templates_jsonl)
    icd_vocab_size = int(sampler.meta["icd_vocab_size"])
    ccs_vocab_size = int(sampler.meta["ccs_vocab_size"])
    ccs_label_vocab_size = int(
        sampler.meta.get("ccs_label_vocab_size", ccs_vocab_size)
    )
    base_vocab_size = icd_vocab_size + ccs_vocab_size
    mapper = EntityTokenMapper.from_sources(
        sampler.vocab.get("name_to_token"),
        cfg.knowledge.entity_to_token_json,
        base_vocab_size=base_vocab_size,
    )
    aligned_index = PoolAlignedTokenIndex(pool, mapper)
    logger.info(
        f"pool token coverage={aligned_index.processed_coverage:.3f} "
        f"({len(aligned_index.processed_tokens)}/{mapper.base_vocab_size})"
    )
    contract = build_checkpoint_contract(
        mapper=mapper,
        processed_path=cfg.data.processed_path,
        templates_path=cfg.knowledge.templates_jsonl,
        split_json=cfg.data.split_json,
        training_split=training_split,
        icd_vocab_size=icd_vocab_size,
        ccs_vocab_size=ccs_vocab_size,
        ccs_label_vocab_size=ccs_label_vocab_size,
        model_dim=cfg.model.dim,
        gnn_layers=cfg.model.gnn_layers,
        attn_heads=cfg.model.attn_heads,
        dropout=cfg.model.dropout,
        pool_dim=aligned_index.dim,
        retrieval_K=cfg.knowledge.retrieval_K,
        retrieval_alpha=cfg.knowledge.retrieval_alpha,
        soft_xi=cfg.train.soft_xi,
    )

    encoder = DecoupledMultiGATEncoder(
        vocab_size=mapper.vocab_size,
        dim=cfg.model.dim,
        gnn_layers=cfg.model.gnn_layers,
        attn_heads=cfg.model.attn_heads,
        dropout=cfg.model.dropout,
    ).to(device)
    initialized_tokens = aligned_index.initialize_embedding_table(
        encoder.embed.emb.weight
    )
    logger.info(f"initialized {initialized_tokens} encoder embeddings from the pool")
    code_embed_lookup = encoder_code_embedding_lookup(encoder)

    predictor = NextVisitPredictor(
        in_dim=cfg.model.dim,
        num_labels=ccs_label_vocab_size,
        dropout=cfg.model.dropout
    ).to(device)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    def build_visit_graph(visit):
        node_ids = visit["node_ids"]
        edge_index = torch.tensor(visit["edge_index"], dtype=torch.long)
        orig_mask = torch.ones(len(node_ids), dtype=torch.bool)
        code_set = set(int(x) for x in visit["icd_tokens"])
        code_mask = torch.tensor([int(nid) in code_set for nid in node_ids], dtype=torch.bool)
        return build_pyg_data(node_ids, edge_index, orig_mask=orig_mask, code_mask=code_mask)

    def maybe_augment(visit, g) -> Tuple[object, Optional[torch.Tensor]]:
        if random.random() >= cfg.train.exposure_prob:
            return g, None

        query_tokens = visit_retrieval_tokens(visit)
        ret = pool.retrieve_topk(
            query_tokens,
            code_embed_lookup,
            K=cfg.knowledge.retrieval_K,
        )
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

    def step_optimizer(gradient_scale: float = 1.0) -> None:
        if gradient_scale != 1.0:
            for parameter in list(encoder.parameters()) + list(predictor.parameters()):
                if parameter.grad is not None:
                    parameter.grad.mul_(float(gradient_scale))
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

                g_raw = build_visit_graph(visit).to(device)
                g_aug, soft_vec = maybe_augment(visit, g_raw)
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
                    num_ccs_labels=ccs_label_vocab_size,
                    icd_vocab_size=icd_vocab_size,
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
            step_optimizer(gradient_scale=update_every / float(accum))

        logger.info(
            f"epoch {epoch+1}/{cfg.train.warmup_epochs} "
            f"loss={running/max(steps,1):.4f}"
        )

    os.makedirs(cfg.outputs.checkpoints_dir, exist_ok=True)
    checkpoint_path = os.path.join(cfg.outputs.checkpoints_dir, "warmup.pt")
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "cfg": asdict(cfg),
            "contract": contract,
        },
        checkpoint_path,
    )
    logger.info(f"saved {checkpoint_path}")


if __name__ == "__main__":
    main()
