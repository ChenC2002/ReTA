"""
This script runs sequential next-visit CCS prediction with visit-level, budget-aware knowledge injection.
It supports:
- Dynamic per-visit action selection via a trained policy (Soft/Hard/Skip)
- Top-K knowledge retrieval from clustered templates
- Sequential processing over a patient trajectory (history state + past-visit memory)

Inputs
------
- processed.pt from `data/preprocess.py`
- templates.jsonl from `knowledge/clustering.py`
- checkpoint produced by `train/warmup.py` or `train/rl_train.py`
  * warmup checkpoint: {encoder, predictor}
  * rl checkpoint: {encoder, predictor, policy}

Hard Import
-----------
ICD:/CCS: template nodes are mapped from `processed.pt`. Provide
`--entity_to_token_json` for additional KG ids such as UMLS:/PrimeKG nodes;
unmapped external nodes are skipped during grafting.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from reta.knowledge.pool import KnowledgePool
from reta.model.encoder import DecoupledMultiGATEncoder
from reta.model.predictor import NextVisitPredictor
from reta.model.visit_graph import build_pyg_data, graft_hard_import
from reta.model.embeddings import labels_to_multihot_from_ccs_tokens

from reta.policy.action import HARD, SOFT, action_size, decode_action
from reta.policy.state import StateEncoder
from reta.policy.policy_net import PolicyNet
from reta.policy.state import TemplateUtilityTracker, build_policy_state

from reta.train.trainer import EntityTokenMapper, template_to_subgraph, template_vector_tensor


# -------------------------
# Metrics (lightweight)
# -------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def micro_f1(y_true: np.ndarray, y_pred_bin: np.ndarray) -> float:
    tp = (y_true * y_pred_bin).sum()
    fp = ((1 - y_true) * y_pred_bin).sum()
    fn = (y_true * (1 - y_pred_bin)).sum()
    denom = (2 * tp + fp + fn)
    return float(2 * tp / denom) if denom > 0 else 0.0


def acc_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 20) -> float:
    if y_true.shape[0] == 0 or y_score.shape[1] == 0:
        return 0.0
    k = min(int(k), y_score.shape[1])
    if k <= 0:
        return 0.0
    topk = np.argpartition(-y_score, kth=k - 1, axis=1)[:, :k]
    hit = 0
    for i in range(y_true.shape[0]):
        if y_true[i, topk[i]].sum() > 0:
            hit += 1
    return float(hit / y_true.shape[0]) if y_true.shape[0] > 0 else 0.0


def auprc_micro(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Micro-averaged AUPRC."""
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true.reshape(-1), y_score.reshape(-1)))
    except Exception:
        # fallback: AP via sorted precision at positives
        yt = y_true.reshape(-1)
        ys = y_score.reshape(-1)
        order = np.argsort(-ys)
        yt = yt[order]
        cumsum = np.cumsum(yt)
        idx_pos = np.where(yt == 1)[0]
        if len(idx_pos) == 0:
            return 0.0
        prec = cumsum[idx_pos] / (idx_pos + 1)
        return float(np.mean(prec))


# -------------------------
# Helpers
# -------------------------

def load_split(split_json: Optional[str], split: str) -> Optional[set]:
    """Load patient ids for a split if provided."""
    if split_json is None or split == "all":
        return None
    with open(split_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    ids = d.get(split)
    if ids is None:
        raise ValueError(f"split '{split}' not found in {split_json}. keys={list(d.keys())}")
    return set(map(str, ids))


def make_embed_lookup(encoder: DecoupledMultiGATEncoder) -> callable:
    """Return a numpy lookup(token)->L2-normalized embedding."""
    w = encoder.embed.emb.weight.detach().cpu().numpy().astype(np.float32)
    n = np.linalg.norm(w, axis=1, keepdims=True) + 1e-12
    wn = w / n

    def lookup(tok: int) -> np.ndarray:
        return wn[int(tok)]
    return lookup


def ensure_candidates(ret_ids: List[int], K: int) -> List[int]:
    if len(ret_ids) == 0:
        return []
    if len(ret_ids) >= K:
        return ret_ids[:K]
    return ret_ids + [ret_ids[-1]] * (K - len(ret_ids))


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
    device: str,
    retrieval_K: int = 20,
    soft_xi: float = 0.5,
    deterministic: bool = True,
    max_patients: Optional[int] = None,
    out_path: Optional[str] = None,
) -> Dict:
    device_t = torch.device(device)

    data = torch.load(processed_path, map_location="cpu")
    trajectories: Dict[str, List[Dict]] = data["trajectories"]
    meta = data["meta"]
    icd_vocab_size = int(meta["icd_vocab_size"])
    ccs_vocab_size = int(meta["ccs_vocab_size"])
    vocab_size = icd_vocab_size + ccs_vocab_size

    keep_ids = load_split(split_json, split)
    pids = list(trajectories.keys())
    if keep_ids is not None:
        pids = [p for p in pids if p in keep_ids]
    if max_patients is not None:
        pids = pids[: int(max_patients)]

    pool = KnowledgePool.load_jsonl(templates_jsonl)
    mapper = EntityTokenMapper.from_sources(data.get("vocab", {}).get("name_to_token"), entity_to_token_json)

    ck = torch.load(checkpoint_path, map_location="cpu")

    cfg = ck.get("cfg", None)
    dim = None
    if cfg is not None:
        try:
            dim = int(cfg.model.dim)
        except Exception:
            dim = None
    if dim is None:
        dim = int(next(iter(ck["encoder"].values())).shape[-1]) if "encoder" in ck else 256

    gnn_layers = int(getattr(getattr(cfg, "model", object()), "gnn_layers", 2)) if cfg is not None else 2
    attn_heads = int(getattr(getattr(cfg, "model", object()), "attn_heads", 4)) if cfg is not None else 4
    dropout = float(getattr(getattr(cfg, "model", object()), "dropout", 0.3)) if cfg is not None else 0.3

    encoder = DecoupledMultiGATEncoder(
        vocab_size=vocab_size,
        dim=dim,
        gnn_layers=gnn_layers,
        attn_heads=attn_heads,
        dropout=dropout,
    ).to(device_t)
    predictor = NextVisitPredictor(in_dim=dim, num_labels=ccs_vocab_size, dropout=dropout).to(device_t)

    encoder.load_state_dict(ck["encoder"], strict=False)
    predictor.load_state_dict(ck["predictor"], strict=False)

    baseline_encoder = None
    baseline_predictor = None
    if "baseline_encoder" in ck and "baseline_predictor" in ck:
        baseline_encoder = DecoupledMultiGATEncoder(
            vocab_size=vocab_size,
            dim=dim,
            gnn_layers=gnn_layers,
            attn_heads=attn_heads,
            dropout=dropout,
        ).to(device_t)
        baseline_predictor = NextVisitPredictor(in_dim=dim, num_labels=ccs_vocab_size, dropout=dropout).to(device_t)
        baseline_encoder.load_state_dict(ck["baseline_encoder"], strict=False)
        baseline_predictor.load_state_dict(ck["baseline_predictor"], strict=False)

    # policy optional (if rl checkpoint)
    policy_net = None
    state_enc = None
    policy_K = int(retrieval_K)
    utility_tracker = TemplateUtilityTracker()
    utility_tracker.values = {int(k): float(v) for k, v in ck.get("template_utilities", {}).items()}
    retrieval_alpha = 0.2
    try:
        retrieval_alpha = float(cfg.knowledge.retrieval_alpha) if cfg is not None else retrieval_alpha
    except Exception:
        retrieval_alpha = 0.2
    if "policy" in ck:
        policy_state_dim = int(ck.get("policy_state_dim", 2 * dim + 1 + retrieval_K))
        policy_action_dim = int(ck.get("action_dim", action_size(retrieval_K)))
        policy_K = max(0, (policy_action_dim - 1) // 2)
        policy_net = PolicyNet(state_dim=policy_state_dim, action_dim=policy_action_dim, hidden_dim=256).to(device_t)
        policy_net.load_state_dict(ck["policy"], strict=False)
        policy_net.eval()
        state_enc = StateEncoder(dim=dim).to(device_t)
        if "state_encoder" in ck:
            state_enc.load_state_dict(ck["state_encoder"], strict=False)
        state_enc.eval()

    encoder.eval()
    predictor.eval()
    if baseline_encoder is not None and baseline_predictor is not None:
        baseline_encoder.eval()
        baseline_predictor.eval()

    lookup = make_embed_lookup(encoder)

    all_logits = []
    all_targets = []
    all_meta = []

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

                # retrieve candidates
                active_K = policy_K if policy_net is not None else retrieval_K
                ret = pool.retrieve_topk(visit["icd_tokens"], lookup, K=active_K)
                candidates = ensure_candidates(ret.template_ids, active_K)

                soft_vec = None
                g_aug = g_raw
                added_nodes = 0
                action_name = "skip"
                template_id = None
                baseline_h_current = None

                if policy_net is None or state_enc is None or len(candidates) == 0:
                    # fallback: best similarity + Soft
                    if len(candidates) > 0:
                        tmpl = pool.get_template(candidates[0])
                        soft_vec = template_vector_tensor(tmpl, dim)
                        template_id = int(candidates[0])
                        action_name = "soft_fallback"
                else:
                    # compute v_t from raw, update state
                    bg_raw = Batch.from_data_list([g_raw]).to(device_t)
                    mem = torch.stack(past_visit_embs, dim=0) if len(past_visit_embs) > 0 else torch.zeros((0, dim), device=device_t)
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
                    hist = s_hist_cur[:dim]
                    cur = s_hist_cur[dim : 2 * dim]
                    retrieval_state = (hist + cur).mul(0.5).detach().cpu().numpy()

                    ret = pool.retrieve_topk(
                        visit["icd_tokens"],
                        lookup,
                        K=policy_K,
                        state_vector=retrieval_state,
                        alpha=retrieval_alpha,
                    )
                    candidates = ensure_candidates(ret.template_ids, policy_K)
                    if len(candidates) == 0:
                        act = None
                    else:
                        util_vec = utility_tracker.vector(candidates, K=policy_K, device=device_t)
                        policy_state = build_policy_state(s_hist_cur.to(device_t), uncertainty.detach(), util_vec)
                        a, _, _ = policy_net.act(policy_state, deterministic=deterministic)
                        act = decode_action(int(a.item()), candidates)

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
                            else:
                                action_name = "hard_noop"

                # encode augmented and predict
                bg = Batch.from_data_list([g_aug]).to(device_t)
                mem = torch.stack(past_visit_embs, dim=0) if len(past_visit_embs) > 0 else torch.zeros((0, dim), device=device_t)
                soft_offset = soft_vec.view(1, -1).to(device_t) if soft_vec is not None else None
                h_G, _ = encoder(bg, past_visit_memory=[mem], soft_offset_per_graph=soft_offset, xi=soft_xi)
                logits = predictor(h_G)

                y = labels_to_multihot_from_ccs_tokens(y_tokens, ccs_vocab_size, icd_vocab_size, device=device_t).view(1, -1)
                all_logits.append(logits.detach().cpu())
                all_targets.append(y.detach().cpu())
                all_meta.append({
                    "patient_id": pid,
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
    y_score = _sigmoid(logits)

    y_pred = (y_score >= 0.5).astype(np.int32)
    metrics = {
        "AUPRC_micro": auprc_micro(y_true, y_score),
        "MicroF1@0.5": micro_f1(y_true, y_pred),
        "Acc@20": acc_at_k(y_true, y_score, k=20),
        "num_samples": int(y_true.shape[0]),
        "num_patients": int(len(set([m["patient_id"] for m in all_meta]))),
    }

    result = {
        "metrics": metrics,
        "meta": {
            "checkpoint": checkpoint_path,
            "processed_path": processed_path,
            "templates_jsonl": templates_jsonl,
            "split": split,
            "retrieval_K": int(retrieval_K),
            "soft_xi": float(soft_xi),
            "deterministic_policy": bool(deterministic),
        },
        "per_sample_meta": all_meta,
    }

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inference for ReTA with dynamic augmentation.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--processed_path", type=str, default="data/processed/processed.pt")
    p.add_argument("--templates_jsonl", type=str, default="knowledge/templates.jsonl")
    p.add_argument("--entity_to_token_json", type=str, default=None)
    p.add_argument("--split_json", type=str, default=None, help="Optional JSON: {train:[...], val:[...], test:[...]}")
    p.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--retrieval_K", type=int, default=20)
    p.add_argument("--soft_xi", type=float, default=0.5)
    p.add_argument("--deterministic", action="store_true", help="Use greedy policy action (argmax).")
    p.add_argument("--max_patients", type=int, default=None)
    p.add_argument("--out", type=str, default=None, help="Optional path to write JSON results.")
    return p


def main():
    args = build_argparser().parse_args()
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
        max_patients=args.max_patients,
        out_path=args.out,
    )
    print("[infer] metrics:", res["metrics"])
    if args.out:
        print(f"[infer] wrote {args.out}")


if __name__ == "__main__":
    main()
