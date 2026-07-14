"""
Sequential next-visit CCS prediction with visit-level, budget-aware knowledge injection.
It supports:
- Dynamic per-visit action selection via a trained policy (Soft/Hard/Skip)
- Top-K knowledge retrieval from clustered templates
- Sequential processing over a patient trajectory (history state + past-visit memory)

Inputs
------
- processed.pt from `reta/data/preprocess.py`
- templates.jsonl from `reta/knowledge/clustering.py`
- checkpoint produced by `reta.learning.warmup` or `reta.learning.rl_train`
  * warmup checkpoint: {encoder, predictor}; evaluated with Skip/raw visits
  * rl checkpoint: {encoder, predictor, policy}

Hard Import
-----------
ICD:/CCS: template nodes are mapped from `processed.pt`. Generate the external
KG mapping with `reta build-entity-map` and provide it through the checkpoint
configuration or `--entity_to_token_json`.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Mapping, Optional

import numpy as np
import torch

from reta.knowledge.pool import KnowledgePool
from .runtime import (
    EntityTokenMapper,
    PoolAlignedTokenIndex,
    build_checkpoint_contract,
    encoder_code_embedding_lookup,
    fold_policy_state_for_retrieval,
    load_checkpoint,
    load_state_dict_strict,
    load_processed_data,
    require_checkpoint_contract,
    resolve_device,
    select_patient_ids,
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
    StateEncoder,
    TemplateUtilityTracker,
    build_policy_state,
    decode_policy_action,
    valid_action_mask,
)


# -------------------------
# Metrics and helpers
# -------------------------

def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def auprc_micro(y_true, y_score) -> float:
    """Compute micro-averaged area under the precision-recall curve."""
    true = _to_numpy(y_true).astype(np.int32)
    score = _to_numpy(y_score).astype(np.float32)
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(true.reshape(-1), score.reshape(-1)))
    except Exception:
        true_flat = true.reshape(-1)
        score_flat = score.reshape(-1)
        order = np.argsort(-score_flat)
        true_flat = true_flat[order]
        cumulative = np.cumsum(true_flat)
        positives = np.where(true_flat == 1)[0]
        if len(positives) == 0:
            return 0.0
        return float(np.mean(cumulative[positives] / (positives + 1)))


def micro_f1(y_true, y_pred_bin) -> float:
    """Compute micro F1 for binary multi-label predictions."""
    true = _to_numpy(y_true).astype(np.int32)
    predicted = _to_numpy(y_pred_bin).astype(np.int32)
    true_positive = (true * predicted).sum()
    false_positive = ((1 - true) * predicted).sum()
    false_negative = (true * (1 - predicted)).sum()
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator > 0 else 0.0


def acc_at_k(y_true, y_score, k: int = 20) -> float:
    """Return mean per-sample recall normalized by ``min(k, positives)``."""
    true = _to_numpy(y_true).astype(np.int32)
    score = _to_numpy(y_score).astype(np.float32)
    if true.size == 0:
        return 0.0
    k = min(int(k), score.shape[1])
    if k <= 0:
        return 0.0
    topk = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
    correct = np.take_along_axis(true, topk, axis=1).sum(axis=1)
    denominators = np.minimum(k, true.sum(axis=1))
    per_sample = np.divide(
        correct,
        denominators,
        out=np.zeros_like(correct, dtype=np.float64),
        where=denominators > 0,
    )
    return float(per_sample.mean())


def compute_all(
    y_true,
    logits=None,
    probs=None,
    threshold: float = 0.5,
    k: int = 20,
) -> Dict[str, float]:
    """Compute the standard next-visit metric bundle."""
    if (logits is None) == (probs is None):
        raise ValueError("provide exactly one of probs or logits")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    true = _to_numpy(y_true)
    if true.ndim != 2:
        raise ValueError("y_true must be a two-dimensional array.")
    if true.shape[0] == 0 or true.shape[1] == 0:
        raise ValueError("y_true must contain at least one sample and one label.")
    if not np.isfinite(true).all() or not np.isin(true, [0, 1]).all():
        raise ValueError("y_true must contain only finite binary values.")
    true = true.astype(np.int32)

    if probs is None:
        logits_array = _to_numpy(logits).astype(np.float32)
        if not np.isfinite(logits_array).all():
            raise ValueError("logits contain NaN or infinite values.")
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits_array, -80.0, 80.0)))

    score = _to_numpy(probs).astype(np.float32)
    if score.shape != true.shape:
        raise ValueError(
            f"prediction shape {score.shape} does not match y_true shape {true.shape}."
        )
    if not np.isfinite(score).all() or ((score < 0.0) | (score > 1.0)).any():
        raise ValueError("probabilities must be finite and in [0, 1].")
    predicted = (score >= float(threshold)).astype(np.int32)
    return {
        "AUPRC_micro": auprc_micro(true, score),
        f"MicroF1@{threshold:g}": micro_f1(true, predicted),
        f"Acc@{int(k)}": acc_at_k(true, score, k=int(k)),
        "num_samples": int(true.shape[0]),
        "num_labels": int(true.shape[1]) if true.ndim == 2 else 0,
    }


def _config_value(config, section: str, key: str, default):
    """Read one value from a format-v2 checkpoint configuration."""
    section_value = config.get(section, {}) if isinstance(config, dict) else {}
    return section_value.get(key, default) if isinstance(section_value, dict) else default


# -------------------------
# Main inference
# -------------------------

def run_inference(
    checkpoint_path: str,
    processed_path: str,
    templates_jsonl: str,
    entity_to_token_json: Optional[str],
    split_json: Optional[str],
    split: str,
    device: str = "auto",
    retrieval_K: Optional[int] = None,
    soft_xi: Optional[float] = None,
    deterministic: bool = True,
    seed: Optional[int] = None,
    max_patients: Optional[int] = None,
    out_path: Optional[str] = None,
    include_sample_metadata: bool = False,
) -> Dict:
    data = load_processed_data(processed_path)
    trajectories: Dict[str, List[Dict]] = data["trajectories"]
    meta = data["meta"]
    icd_vocab_size = int(meta["icd_vocab_size"])
    ccs_vocab_size = int(meta["ccs_vocab_size"])
    ccs_label_vocab_size = int(meta.get("ccs_label_vocab_size", ccs_vocab_size))
    base_vocab_size = icd_vocab_size + ccs_vocab_size

    pool = KnowledgePool.load_jsonl(templates_jsonl)
    ck = load_checkpoint(checkpoint_path)
    cfg = ck.get("cfg", None)
    if not isinstance(cfg, dict):
        raise ValueError("checkpoint cfg must be a format-v2 configuration mapping.")
    encoder_state = ck.get("encoder")
    predictor_state = ck.get("predictor")
    if not isinstance(encoder_state, Mapping) or not isinstance(predictor_state, Mapping):
        raise ValueError("checkpoint must contain encoder and predictor state dictionaries.")
    embedding_weight = encoder_state.get("embed.emb.weight")
    if embedding_weight is None or embedding_weight.ndim != 2:
        raise ValueError("checkpoint encoder state is missing embed.emb.weight.")
    inferred_dim = int(embedding_weight.shape[1])
    dim = int(_config_value(cfg, "model", "dim", inferred_dim))
    gnn_layers = int(_config_value(cfg, "model", "gnn_layers", 2))
    attn_heads = int(_config_value(cfg, "model", "attn_heads", 4))
    dropout = float(_config_value(cfg, "model", "dropout", 0.3))
    effective_retrieval_K = int(
        retrieval_K
        if retrieval_K is not None
        else _config_value(cfg, "knowledge", "retrieval_K", 20)
    )
    effective_soft_xi = float(
        soft_xi
        if soft_xi is not None
        else _config_value(cfg, "train", "soft_xi", 0.5)
    )
    effective_seed = int(
        seed if seed is not None else _config_value(cfg, "train", "seed", 42)
    )
    effective_entity_mapping = entity_to_token_json
    if effective_entity_mapping is None:
        effective_entity_mapping = _config_value(
            cfg,
            "knowledge",
            "entity_to_token_json",
            None,
        )
    effective_split_json = split_json
    if effective_split_json is None:
        effective_split_json = _config_value(cfg, "data", "split_json", None)
    training_split = (
        str(_config_value(cfg, "data", "train_split", "train"))
        if effective_split_json
        else "all"
    )
    if effective_retrieval_K <= 0:
        raise ValueError("retrieval_K must be positive.")
    if effective_soft_xi < 0.0:
        raise ValueError("soft_xi must be non-negative.")
    if max_patients is not None and int(max_patients) <= 0:
        raise ValueError("max_patients must be positive when provided.")

    pids = select_patient_ids(
        list(trajectories.keys()),
        effective_split_json,
        split,
    )
    if max_patients is not None:
        pids = pids[: int(max_patients)]

    device_t = resolve_device(device)
    set_seed(effective_seed)
    mapper = EntityTokenMapper.from_sources(
        data.get("vocab", {}).get("name_to_token"),
        effective_entity_mapping,
        base_vocab_size=base_vocab_size,
    )
    aligned_index = PoolAlignedTokenIndex(pool, mapper)
    contract = build_checkpoint_contract(
        mapper=mapper,
        processed_path=processed_path,
        templates_path=templates_jsonl,
        split_json=effective_split_json,
        training_split=training_split,
        icd_vocab_size=icd_vocab_size,
        ccs_vocab_size=ccs_vocab_size,
        ccs_label_vocab_size=ccs_label_vocab_size,
        model_dim=dim,
        gnn_layers=gnn_layers,
        attn_heads=attn_heads,
        dropout=dropout,
        pool_dim=aligned_index.dim,
        retrieval_K=int(_config_value(cfg, "knowledge", "retrieval_K", 20)),
        retrieval_alpha=float(
            _config_value(cfg, "knowledge", "retrieval_alpha", 0.3)
        ),
        soft_xi=float(_config_value(cfg, "train", "soft_xi", 0.5)),
    )
    require_checkpoint_contract(ck, contract)

    encoder = DecoupledMultiGATEncoder(
        vocab_size=mapper.vocab_size,
        dim=dim,
        gnn_layers=gnn_layers,
        attn_heads=attn_heads,
        dropout=dropout,
    ).to(device_t)
    predictor = NextVisitPredictor(
        in_dim=dim,
        num_labels=ccs_label_vocab_size,
        dropout=dropout,
    ).to(device_t)

    load_state_dict_strict(encoder, encoder_state, "encoder")
    load_state_dict_strict(predictor, predictor_state, "predictor")

    baseline_encoder = None
    baseline_predictor = None
    if ("baseline_encoder" in ck) != ("baseline_predictor" in ck):
        raise ValueError("checkpoint must contain both baseline components or neither.")
    if "baseline_encoder" in ck and "baseline_predictor" in ck:
        baseline_encoder = DecoupledMultiGATEncoder(
            vocab_size=mapper.vocab_size,
            dim=dim,
            gnn_layers=gnn_layers,
            attn_heads=attn_heads,
            dropout=dropout,
        ).to(device_t)
        baseline_predictor = NextVisitPredictor(
            in_dim=dim,
            num_labels=ccs_label_vocab_size,
            dropout=dropout,
        ).to(device_t)
        load_state_dict_strict(
            baseline_encoder,
            ck["baseline_encoder"],
            "baseline encoder",
        )
        load_state_dict_strict(
            baseline_predictor,
            ck["baseline_predictor"],
            "baseline predictor",
        )

    # policy optional (if rl checkpoint)
    policy_net = None
    state_enc = None
    policy_K = effective_retrieval_K
    utility_tracker = TemplateUtilityTracker()
    utility_values = ck.get("template_utilities", {})
    if not isinstance(utility_values, Mapping):
        raise ValueError("checkpoint template_utilities must be a mapping.")
    parsed_utilities = {}
    known_template_ids = set(pool.id_to_index)
    for raw_template_id, raw_value in utility_values.items():
        template_id = int(raw_template_id)
        value = float(raw_value)
        if template_id not in known_template_ids:
            raise ValueError(
                f"checkpoint utility references unknown template {template_id}."
            )
        if template_id in parsed_utilities:
            raise ValueError(
                f"checkpoint contains duplicate utility key for template {template_id}."
            )
        if not np.isfinite(value):
            raise ValueError(
                f"checkpoint utility for template {template_id} is not finite."
            )
        parsed_utilities[template_id] = value
    utility_tracker.values = parsed_utilities
    retrieval_alpha = float(_config_value(cfg, "knowledge", "retrieval_alpha", 0.3))
    if "policy" in ck:
        if baseline_encoder is None or baseline_predictor is None:
            raise ValueError("policy checkpoint is missing its frozen baseline models.")
        policy_action_dim = int(ck.get("action_dim", 0))
        if policy_action_dim <= 0 or policy_action_dim % 2 != 1:
            raise ValueError("checkpoint policy action_dim must be a positive odd integer.")
        policy_K = (policy_action_dim - 1) // 2
        checkpoint_K = int(_config_value(cfg, "knowledge", "retrieval_K", 0))
        if policy_K != checkpoint_K:
            raise ValueError(
                f"checkpoint action_dim implies K={policy_K}, but config stores K={checkpoint_K}."
            )
        if retrieval_K is not None and int(retrieval_K) != policy_K:
            raise ValueError(
                f"retrieval_K={retrieval_K} conflicts with checkpoint policy K={policy_K}."
            )
        policy_state_dim = int(ck.get("policy_state_dim", 0))
        expected_state_dim = 2 * dim + 1 + policy_K
        if policy_state_dim != expected_state_dim:
            raise ValueError(
                f"checkpoint policy_state_dim={policy_state_dim}, expected {expected_state_dim}."
            )
        policy_net = PolicyNet(state_dim=policy_state_dim, action_dim=policy_action_dim, hidden_dim=256).to(device_t)
        load_state_dict_strict(policy_net, ck["policy"], "policy")
        policy_net.eval()
        state_enc = StateEncoder(dim=dim).to(device_t)
        if "state_encoder" not in ck:
            raise ValueError("policy checkpoint is missing state_encoder.")
        load_state_dict_strict(state_enc, ck["state_encoder"], "state encoder")
        state_enc.eval()

    encoder.eval()
    predictor.eval()
    if baseline_encoder is not None and baseline_predictor is not None:
        baseline_encoder.eval()
        baseline_predictor.eval()

    code_embed_lookup = encoder_code_embedding_lookup(encoder)

    all_logits = []
    all_targets = []
    sample_metadata = []
    patient_index = {patient_id: index for index, patient_id in enumerate(pids)}
    evaluated_patient_indices = set()

    from torch_geometric.data import Batch

    with torch.no_grad():
        for pid in pids:
            traj = trajectories[pid]
            if len(traj) < 2:
                continue

            if state_enc is not None:
                state_enc.reset()

            past_visit_embs: List[torch.Tensor] = []
            baseline_past_visit_embs: List[torch.Tensor] = []

            for t in range(len(traj) - 1):
                visit = traj[t]
                y_tokens = visit.get("label_ccs")
                if y_tokens is None:
                    continue

                # raw graph
                node_ids = visit["node_ids"]
                edge_index = torch.tensor(visit["edge_index"], dtype=torch.long)
                orig_mask = torch.ones(len(node_ids), dtype=torch.bool)
                code_set = set(int(x) for x in visit["icd_tokens"])
                code_mask = torch.tensor([int(nid) in code_set for nid in node_ids], dtype=torch.bool)
                g_raw = build_pyg_data(node_ids, edge_index, orig_mask=orig_mask, code_mask=code_mask).to(device_t)

                soft_vec = None
                g_aug = g_raw
                added_nodes = 0
                action_name = "skip"
                template_id = None
                baseline_h_current = None
                candidates = []
                mem = (
                    torch.stack(past_visit_embs, dim=0)
                    if past_visit_embs
                    else torch.zeros((0, dim), device=device_t)
                )

                if policy_net is not None and state_enc is not None:
                    # Encode the unaugmented visit first so retrieval uses the
                    # same learned history/current state as the policy.
                    bg_raw = Batch.from_data_list([g_raw]).to(device_t)
                    h_G_raw, _ = encoder(bg_raw, past_visit_memory=[mem])
                    if baseline_encoder is not None and baseline_predictor is not None:
                        base_mem = (
                            torch.stack(baseline_past_visit_embs, dim=0)
                            if len(baseline_past_visit_embs) > 0
                            else torch.zeros((0, dim), device=device_t)
                        )
                        h_G_base, _ = baseline_encoder(bg_raw, past_visit_memory=[base_mem])
                        base_logits = baseline_predictor(h_G_base)
                        baseline_h_current = h_G_base[0].detach()
                    else:
                        h_G_base = h_G_raw
                        base_logits = predictor(h_G_raw)
                    uncertainty = 1.0 - torch.sigmoid(base_logits).max()
                    s_hist_cur = state_enc.step(h_G_raw[0])
                    query_tokens = visit_retrieval_tokens(visit)
                    retrieval_state = fold_policy_state_for_retrieval(
                        s_hist_cur,
                        dim,
                    )
                    if query_tokens:
                        ret = pool.retrieve_topk(
                            query_tokens,
                            code_embed_lookup,
                            K=policy_K,
                            state_vector=retrieval_state,
                            alpha=retrieval_alpha,
                        )
                        candidates = ret.template_ids
                    util_vec = utility_tracker.vector(
                        candidates,
                        K=policy_K,
                        device=device_t,
                    )
                    policy_state = build_policy_state(
                        s_hist_cur.to(device_t),
                        uncertainty.detach(),
                        util_vec,
                    )
                    action_mask = valid_action_mask(
                        len(candidates),
                        policy_K,
                        device=device_t,
                    )
                    a, _, _ = policy_net.act(
                        policy_state,
                        deterministic=deterministic,
                        action_mask=action_mask,
                    )
                    act = decode_policy_action(int(a.item()), candidates, policy_K)

                    if act is not None and act.is_skip:
                        action_name = "skip"
                    elif act is not None:
                        tmpl = pool.get_template(act.template_id)
                        template_id = int(act.template_id)

                        if act.mode == SOFT:
                            action_name = "soft"
                            soft_vec = template_vector_tensor(tmpl, dim)
                        elif act.mode == HARD:
                            action_name = "hard"
                            sub = template_to_subgraph(tmpl.medoid.to_dict(), mapper)
                            if sub is not None:
                                before = int(g_raw.node_ids.numel())
                                g_aug = graft_hard_import(g_raw, sub)
                                after = int(g_aug.node_ids.numel())
                                added_nodes = max(0, after - before)
                                if g_aug is g_raw:
                                    action_name = "hard_noop"
                            else:
                                action_name = "hard_noop"

                # encode augmented and predict
                bg = Batch.from_data_list([g_aug]).to(device_t)
                soft_offset = soft_vec.view(1, -1).to(device_t) if soft_vec is not None else None
                h_G, _ = encoder(
                    bg,
                    past_visit_memory=[mem],
                    soft_offset_per_graph=soft_offset,
                    xi=effective_soft_xi,
                )
                logits = predictor(h_G)

                y = labels_to_multihot_from_ccs_tokens(
                    y_tokens,
                    ccs_label_vocab_size,
                    icd_vocab_size,
                    device=device_t,
                ).view(1, -1)
                all_logits.append(logits.detach().cpu())
                all_targets.append(y.detach().cpu())
                local_patient_index = patient_index[pid]
                evaluated_patient_indices.add(local_patient_index)
                if include_sample_metadata:
                    sample_metadata.append({
                        "patient_index": local_patient_index,
                        "t": t,
                        "action": action_name,
                        "template_id": template_id,
                        "added_nodes": int(added_nodes),
                        "used_soft": soft_vec is not None,
                    })

                # update memory with current visit embedding (after augmentation)
                past_visit_embs.append(h_G[0].detach())
                if baseline_h_current is not None:
                    baseline_past_visit_embs.append(baseline_h_current)

    if len(all_logits) == 0:
        raise RuntimeError("No transitions found for inference. Check processed.pt and split settings.")

    logits = torch.cat(all_logits, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy().astype(np.int32)
    metrics = compute_all(y_true, logits=logits, threshold=0.5, k=20)
    metrics["num_patients"] = int(len(evaluated_patient_indices))

    result = {
        "metrics": metrics,
        "meta": {
            "checkpoint": checkpoint_path,
            "processed_path": processed_path,
            "templates_jsonl": templates_jsonl,
            "split": split,
            "retrieval_K": int(policy_K if policy_net is not None else effective_retrieval_K),
            "soft_xi": float(effective_soft_xi),
            "deterministic_policy": bool(deterministic),
            "augmentation_policy": "learned" if policy_net is not None else "skip_raw",
            "seed": int(effective_seed),
            "metric_value_scale": "unit_interval",
            "auprc_averaging": "micro",
            "prediction_threshold": 0.5,
            "accuracy_k": 20,
            "retrieval_space": contract["retrieval_space"],
            "pool_token_coverage": float(aligned_index.processed_coverage),
            "processed_sha256": contract["processed_sha256"],
            "templates_sha256": contract["templates_sha256"],
            "split_manifest_sha256": contract["split_manifest_sha256"],
        },
    }
    if include_sample_metadata:
        result["per_sample_meta"] = sample_metadata

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")

    return result


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inference for ReTA with dynamic augmentation.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--processed_path", type=str, default="data/processed/processed.pt")
    p.add_argument("--templates_jsonl", type=str, default="data/knowledge/templates.jsonl")
    p.add_argument("--entity_to_token_json", type=str, default=None)
    p.add_argument(
        "--split_json",
        type=str,
        default="data/processed/splits.json",
        help="Disjoint patient split manifest generated during preprocessing.",
    )
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--retrieval_K", type=int, default=None)
    p.add_argument("--soft_xi", type=float, default=None)
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use greedy policy actions; pass --no-deterministic for seeded sampling.",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max_patients", type=int, default=None)
    p.add_argument("--out", type=str, default="results/inference.json", help="Path to write JSON results.")
    p.add_argument("--log_dir", type=str, default="logs", help="Directory for inference logs.")
    p.add_argument(
        "--include-sample-metadata",
        action="store_true",
        help="Include action metadata keyed only by a run-local patient index.",
    )
    return p


def main():
    args = build_argparser().parse_args()
    logger = setup_logger(args.log_dir, name="inference")
    res = run_inference(
        checkpoint_path=args.checkpoint,
        processed_path=args.processed_path,
        templates_jsonl=args.templates_jsonl,
        entity_to_token_json=args.entity_to_token_json,
        split_json=args.split_json,
        split=args.split,
        device=args.device,
        retrieval_K=args.retrieval_K,
        soft_xi=args.soft_xi,
        deterministic=args.deterministic,
        seed=args.seed,
        max_patients=args.max_patients,
        out_path=args.out,
        include_sample_metadata=args.include_sample_metadata,
    )
    logger.info(f"metrics: {json.dumps(res['metrics'], sort_keys=True)}")
    if args.out:
        logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
