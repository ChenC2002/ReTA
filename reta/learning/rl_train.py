"""Stage 2: REINFORCE policy learning and encoder refinement."""

from __future__ import annotations

import argparse
import copy
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
    fold_policy_state_for_retrieval,
    load_checkpoint,
    load_config,
    load_state_dict_strict,
    require_checkpoint_contract,
    resolve_device,
    set_seed,
    setup_logger,
    template_to_subgraph,
    template_vector_tensor,
    visit_retrieval_tokens,
)
from .model import (
    DecoupledMultiGATEncoder,
    NextVisitPredictor,
    build_pyg_data,
    graft_hard_import,
    labels_to_multihot_from_ccs_tokens,
)
from .policy import (
    HARD,
    SOFT,
    PolicyNet,
    ReinforceBuffer,
    ReinforceConfig,
    ReinforceTrainer,
    RewardConfig,
    StateEncoder,
    TemplateUtilityTracker,
    action_size,
    build_policy_state,
    compute_paired_reward,
    decode_policy_action,
    valid_action_mask,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 2 REINFORCE training for ReTA policy.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--warmup_ckpt", type=str, default=None)
    return p


def make_models(vocab_size: int, ccs_vocab_size: int, cfg, device: torch.device) -> Tuple[DecoupledMultiGATEncoder, NextVisitPredictor]:
    encoder = DecoupledMultiGATEncoder(
        vocab_size=vocab_size,
        dim=cfg.model.dim,
        gnn_layers=cfg.model.gnn_layers,
        attn_heads=cfg.model.attn_heads,
        dropout=cfg.model.dropout,
    ).to(device)
    predictor = NextVisitPredictor(in_dim=cfg.model.dim, num_labels=ccs_vocab_size, dropout=cfg.model.dropout).to(device)
    return encoder, predictor


def main():
    args = build_argparser().parse_args()
    cfg = load_config(args.config)
    logger = setup_logger(cfg.outputs.logs_dir, name="rl_train")
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    warmup_checkpoint = args.warmup_ckpt or os.path.join(
        cfg.outputs.checkpoints_dir, "warmup.pt"
    )

    training_split = cfg.data.train_split if cfg.data.split_json else "all"
    sampler = TrajectorySampler(
        cfg.data.processed_path,
        split_json=cfg.data.split_json,
        split=training_split,
    )
    logger.info(f"training split={training_split} patients={len(sampler.patient_ids)}")
    icd_vocab_size = int(sampler.meta["icd_vocab_size"])
    ccs_vocab_size = int(sampler.meta["ccs_vocab_size"])
    ccs_label_vocab_size = int(
        sampler.meta.get("ccs_label_vocab_size", ccs_vocab_size)
    )
    base_vocab_size = icd_vocab_size + ccs_vocab_size

    pool = KnowledgePool.load_jsonl(cfg.knowledge.templates_jsonl)
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

    encoder, predictor = make_models(
        mapper.vocab_size,
        ccs_label_vocab_size,
        cfg,
        device,
    )
    if os.path.exists(warmup_checkpoint):
        ck = load_checkpoint(warmup_checkpoint)
        require_checkpoint_contract(ck, contract)
        load_state_dict_strict(encoder, ck["encoder"], "encoder")
        load_state_dict_strict(predictor, ck["predictor"], "predictor")
        logger.info(f"loaded warmup checkpoint: {warmup_checkpoint}")
    else:
        raise FileNotFoundError(
            f"Warmup checkpoint is required for Stage 2 baseline: {warmup_checkpoint}"
        )
    code_embed_lookup = encoder_code_embedding_lookup(encoder)

    # Frozen Stage-1 baseline for L_CE(G_raw) and uncertainty u_t.
    baseline_encoder = copy.deepcopy(encoder).to(device).eval()
    baseline_predictor = copy.deepcopy(predictor).to(device).eval()
    for p in list(baseline_encoder.parameters()) + list(baseline_predictor.parameters()):
        p.requires_grad_(False)

    K = int(cfg.knowledge.retrieval_K)
    policy_state_dim = 2 * cfg.model.dim + 1 + K
    policy_net = PolicyNet(state_dim=policy_state_dim, action_dim=action_size(K), hidden_dim=256, dropout=0.1).to(device)
    state_enc = StateEncoder(dim=cfg.model.dim).to(device)
    reinforce = ReinforceTrainer(
        policy_net,
        ReinforceConfig(
            gamma=cfg.train.gamma,
            baseline_decay=cfg.train.baseline_decay,
            entropy_coef=cfg.train.policy_entropy_coef,
            max_grad_norm=cfg.train.grad_clip,
        ),
        lr=cfg.train.policy_lr,
        weight_decay=cfg.train.weight_decay,
        extra_parameters=state_enc.parameters(),
    )

    enc_opt = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    utility_tracker = TemplateUtilityTracker(decay=cfg.train.utility_decay)
    reward_cfg = RewardConfig(lambda1=cfg.train.reward_lambda1, lambda2=cfg.train.reward_lambda2)

    def labels(y_tokens: List[int]) -> torch.Tensor:
        return labels_to_multihot_from_ccs_tokens(
            y_tokens,
            ccs_label_vocab_size,
            icd_vocab_size,
            device=device,
        ).view(1, -1)

    def forward_logits_loss(
        enc: DecoupledMultiGATEncoder,
        pred: NextVisitPredictor,
        graph_data,
        soft_vec: Optional[torch.Tensor],
        y_tokens: List[int],
        past_memory: Optional[List[torch.Tensor]],
        train_mode: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from torch_geometric.data import Batch

        enc.train(train_mode)
        pred.train(train_mode)
        bg = Batch.from_data_list([graph_data]).to(device)
        soft_offset = soft_vec.view(1, -1).to(device) if soft_vec is not None else None
        h_G, _ = enc(bg, past_visit_memory=past_memory, soft_offset_per_graph=soft_offset, xi=cfg.train.soft_xi)
        logits = pred(h_G)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels(y_tokens))
        return logits, loss, h_G

    for it in range(cfg.train.rl_iters):
        buffer = ReinforceBuffer()
        stored_transitions = []
        action_counts = {"soft": 0, "hard": 0, "skip": 0}

        patient_ids = sampler.sample_patients(cfg.train.rollout_patients)
        policy_net.train()
        state_enc.train()
        encoder.eval()
        predictor.eval()
        baseline_encoder.eval()
        baseline_predictor.eval()

        for pid in patient_ids:
            traj_steps = []
            traj = sampler.get_patient_traj(pid)
            T = min(len(traj), cfg.train.max_visits_per_patient)
            if T < 2:
                continue

            state_enc.reset()
            past_visit_embs: List[torch.Tensor] = []
            baseline_past_visit_embs: List[torch.Tensor] = []
            for t in range(T - 1):
                visit = traj[t]
                y_tokens = visit.get("label_ccs")
                if y_tokens is None:
                    continue

                node_ids = visit["node_ids"]
                edge_index = torch.tensor(visit["edge_index"], dtype=torch.long)
                orig_mask = torch.ones(len(node_ids), dtype=torch.bool)
                code_set = set(int(x) for x in visit["icd_tokens"])
                code_mask = torch.tensor([int(nid) in code_set for nid in node_ids], dtype=torch.bool)
                g_raw = build_pyg_data(node_ids, edge_index, orig_mask=orig_mask, code_mask=code_mask).to(device)
                mem = [torch.stack(past_visit_embs, dim=0)] if past_visit_embs else [torch.zeros((0, cfg.model.dim), device=device)]
                base_mem = (
                    [torch.stack(baseline_past_visit_embs, dim=0)]
                    if baseline_past_visit_embs
                    else [torch.zeros((0, cfg.model.dim), device=device)]
                )

                with torch.no_grad():
                    _, _, h_G_current_raw = forward_logits_loss(
                        encoder, predictor, g_raw, None, y_tokens, mem, train_mode=False
                    )
                    base_logits, base_loss_raw, h_G_base = forward_logits_loss(
                        baseline_encoder, baseline_predictor, g_raw, None, y_tokens, base_mem, train_mode=False
                    )
                    base_prob = torch.sigmoid(base_logits)
                    uncertainty = 1.0 - base_prob.max()

                s_hist_cur = state_enc.step(h_G_current_raw[0].detach())
                query_tokens = visit_retrieval_tokens(visit)
                retrieval_state = fold_policy_state_for_retrieval(
                    s_hist_cur,
                    cfg.model.dim,
                )

                if query_tokens:
                    ret = pool.retrieve_topk(
                        query_tokens,
                        code_embed_lookup,
                        K=K,
                        state_vector=retrieval_state,
                        alpha=cfg.knowledge.retrieval_alpha,
                    )
                    candidates = ret.template_ids
                else:
                    candidates = []

                util_vec = utility_tracker.vector(candidates, K=K, device=device)
                policy_state = build_policy_state(s_hist_cur.to(device), uncertainty.detach(), util_vec)
                action_mask = valid_action_mask(len(candidates), K, device=device)
                a, logp, entropy = policy_net.act(
                    policy_state,
                    deterministic=False,
                    action_mask=action_mask,
                )
                act = decode_policy_action(int(a.item()), candidates, K)

                soft_vec = None
                g_aug = g_raw
                added_nodes = 0
                template_id = act.template_id
                effective_skip = act.is_skip

                if act.is_skip:
                    action_counts["skip"] += 1
                else:
                    tmpl = pool.get_template(template_id)
                    if act.mode == SOFT:
                        action_counts["soft"] += 1
                        soft_vec = template_vector_tensor(tmpl, cfg.model.dim, device=device)
                    elif act.mode == HARD:
                        sub = template_to_subgraph(tmpl.medoid.to_dict(), mapper)
                        if sub is not None:
                            action_counts["hard"] += 1
                            before = int(g_raw.node_ids.numel())
                            g_aug = graft_hard_import(g_raw, sub)
                            added_nodes = max(0, int(g_aug.node_ids.numel()) - before)
                            if g_aug is g_raw:
                                effective_skip = True
                                template_id = None
                                action_counts["hard"] -= 1
                                action_counts["skip"] += 1
                        else:
                            effective_skip = True
                            template_id = None
                            action_counts["skip"] += 1

                with torch.no_grad():
                    _, loss_edit, h_G_aug = forward_logits_loss(encoder, predictor, g_aug, soft_vec, y_tokens, mem, train_mode=False)
                    reward = compute_paired_reward(
                        base_loss_raw,
                        loss_edit,
                        is_hard=(act.mode == HARD and not effective_skip),
                        is_skip=effective_skip,
                        added_nodes=added_nodes,
                        base_nodes=int(g_raw.node_ids.numel()),
                        cfg=reward_cfg,
                    ).detach()
                    if not torch.isfinite(reward).all():
                        raise RuntimeError("non-finite policy reward encountered.")

                if template_id is not None:
                    utility_tracker.update(template_id, float(reward.item()))

                traj_steps.append({"logp": logp.view(()), "entropy": entropy.view(()), "reward": reward.view(())})
                stored_transitions.append(
                    (
                        g_aug.detach().cpu(),
                        soft_vec.detach().cpu() if soft_vec is not None else None,
                        mem[0].detach().cpu(),
                        y_tokens,
                    )
                )
                past_visit_embs.append(h_G_aug[0].detach())
                baseline_past_visit_embs.append(h_G_base[0].detach())

            buffer.add_trajectory(traj_steps)

        stats = reinforce.update(buffer)

        # Supervised refinement on policy-augmented visits.
        if stored_transitions:
            random.shuffle(stored_transitions)
            total_loss = 0.0
            total_samples = 0
            cursor = 0
            for _ in range(cfg.train.encoder_updates_per_iter):
                enc_opt.zero_grad(set_to_none=True)
                batch = []
                for _ in range(cfg.data.batch_size):
                    if cursor >= len(stored_transitions):
                        random.shuffle(stored_transitions)
                        cursor = 0
                    batch.append(stored_transitions[cursor])
                    cursor += 1

                for g_aug, soft_vec, past_memory, y_tokens in batch:
                    g_aug = g_aug.to(device)
                    soft_vec = soft_vec.to(device) if soft_vec is not None else None
                    past_memory = past_memory.to(device)
                    _, loss, _ = forward_logits_loss(
                        encoder,
                        predictor,
                        g_aug,
                        soft_vec,
                        y_tokens,
                        [past_memory],
                        train_mode=True,
                    )
                    (loss / len(batch)).backward()
                    total_loss += float(loss.item())
                    total_samples += 1

                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()),
                    cfg.train.grad_clip,
                )
                enc_opt.step()
            sup_loss = total_loss / total_samples
        else:
            sup_loss = 0.0

        total_actions = max(1, sum(action_counts.values()))
        logger.info(
            f"iter {it+1}/{cfg.train.rl_iters} "
            f"REINFORCE(loss={stats['loss_policy']:.4f}, ent={stats['entropy']:.3f}, "
            f"return={stats['return_mean']:.4f}, baseline={stats['baseline']:.4f}) "
            f"actions=S:{action_counts['soft']/total_actions:.2f} H:{action_counts['hard']/total_actions:.2f} "
            f"Skip:{action_counts['skip']/total_actions:.2f} sup_loss={sup_loss:.4f}"
        )

        os.makedirs(cfg.outputs.checkpoints_dir, exist_ok=True)
        checkpoint_path = os.path.join(cfg.outputs.checkpoints_dir, f"rl_iter{it+1}.pt")
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "baseline_encoder": baseline_encoder.state_dict(),
                "baseline_predictor": baseline_predictor.state_dict(),
                "policy": policy_net.state_dict(),
                "state_encoder": state_enc.state_dict(),
                "policy_state_dim": policy_state_dim,
                "action_dim": action_size(K),
                "template_utilities": utility_tracker.values,
                "cfg": asdict(cfg),
                "contract": contract,
            },
            checkpoint_path,
        )
        logger.info(f"saved {checkpoint_path}")


if __name__ == "__main__":
    main()
