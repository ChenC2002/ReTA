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
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch

from .runtime import load_processed_data, select_patient_ids, setup_logger
from .model import labels_to_multihot_from_ccs_tokens
from .prediction import load_prediction_session


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
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(true.reshape(-1), score.reshape(-1)))


def micro_f1(y_true, y_pred_bin) -> float:
    """Compute micro F1 for binary multi-label predictions."""
    from sklearn.metrics import f1_score
    return float(f1_score(_to_numpy(y_true).reshape(-1), _to_numpy(y_pred_bin).reshape(-1),
                          average="binary", zero_division=0))


def auprc_macro(y_true, y_score) -> float:
    """Mean per-label average precision; labels without positives contribute zero."""
    from sklearn.metrics import average_precision_score
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No positive class found in y_true")
        return float(average_precision_score(_to_numpy(y_true), _to_numpy(y_score), average="macro"))


def select_f1_threshold(y_true, probs) -> float:
    """Select a global Micro-F1 threshold on validation predictions only.

    Search observed score thresholds with sklearn; ties choose the highest
    threshold (fewer predicted labels). The caller enforces split isolation.
    """
    from sklearn.metrics import precision_recall_curve
    true, score = _to_numpy(y_true), _to_numpy(probs)
    if (true.ndim != 2 or true.size == 0 or score.shape != true.shape
            or not np.isin(true, [0, 1]).all() or not np.isfinite(score).all()
            or ((score < 0) | (score > 1)).any()):
        raise ValueError("Threshold selection requires aligned binary labels and finite probabilities.")
    if not true.any():
        return 1.0
    precision, recall, thresholds = precision_recall_curve(true.ravel(), score.ravel())
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    best = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[best])


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
    f1 = micro_f1(true, predicted)
    return {
        "AUPRC_macro": auprc_macro(true, score),
        "AUPRC_micro": auprc_micro(true, score),
        "MicroF1": f1,
        f"MicroF1@{threshold:g}": f1,
        f"Acc@{int(k)}": acc_at_k(true, score, k=int(k)),
        "num_samples": int(true.shape[0]),
        "num_labels": int(true.shape[1]) if true.ndim == 2 else 0,
    }


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
    calibrate_threshold: bool = False,
) -> Dict:
    if calibrate_threshold and split != "val":
        raise ValueError("Threshold calibration is allowed only on the validation split.")
    data = load_processed_data(processed_path)
    trajectories: Dict[str, List[Dict]] = data["trajectories"]
    if max_patients is not None and int(max_patients) <= 0:
        raise ValueError("max_patients must be positive when provided.")
    session = load_prediction_session(
        checkpoint_path, processed_path, templates_jsonl, entity_to_token_json,
        split_json, device, retrieval_K, soft_xi, seed, data=data,
    )
    if calibrate_threshold and (not session.effective_split_json or session.training_split != "train"):
        raise ValueError("Threshold calibration requires held-out validation patients and train-only fitting.")
    pids = select_patient_ids(list(trajectories), session.effective_split_json, split)
    if max_patients is not None:
        pids = pids[:int(max_patients)]
    all_logits, all_targets, sample_metadata = [], [], []
    evaluated_patient_indices = set()
    for patient_index, pid in enumerate(pids):
        session.reset()
        for t, visit in enumerate(trajectories[pid][:-1]):
            y_tokens = visit.get("label_ccs")
            if y_tokens is None:
                continue
            logits, details = session.predict_visit(visit, deterministic=deterministic)
            target = labels_to_multihot_from_ccs_tokens(
                y_tokens, session.ccs_label_vocab_size, session.icd_vocab_size,
            ).view(1, -1)
            all_logits.append(logits)
            all_targets.append(target)
            evaluated_patient_indices.add(patient_index)
            if include_sample_metadata:
                sample_metadata.append({"patient_index": patient_index, "t": t, **details})

    ck = session.ck
    contract = session.contract
    policy_K = session.policy_K
    effective_soft_xi = session.effective_soft_xi
    effective_seed = session.effective_seed
    policy_net = session.policy_net
    aligned_index = session.aligned_index
    if len(all_logits) == 0:
        raise RuntimeError("No transitions found for inference. Check processed.pt and split settings.")

    logits = torch.cat(all_logits, dim=0).numpy()
    y_true = torch.cat(all_targets, dim=0).numpy().astype(np.int32)
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    selection = ck.get("selection", {})
    threshold = float(selection.get("prediction_threshold", 0.5))
    threshold_source = "validation" if "prediction_threshold" in selection else "default"
    if calibrate_threshold:
        threshold = select_f1_threshold(y_true, probs)
        threshold_source = "validation"
    metrics = compute_all(y_true, probs=probs, threshold=threshold, k=20)
    metrics["num_patients"] = int(len(evaluated_patient_indices))

    result = {
        "metrics": metrics,
        "meta": {
            "checkpoint": checkpoint_path,
            "processed_path": processed_path,
            "templates_jsonl": templates_jsonl,
            "split": split,
            "retrieval_K": int(policy_K),
            "soft_xi": float(effective_soft_xi),
            "deterministic_policy": bool(deterministic),
            "augmentation_policy": "learned" if policy_net is not None else "skip_raw",
            "seed": int(effective_seed),
            "metric_value_scale": "unit_interval",
            "auprc_averaging": "macro",
            "auprc_estimator": "average_precision",
            "empty_label_ap": 0.0,
            "prediction_threshold": threshold,
            "threshold_source": threshold_source,
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
