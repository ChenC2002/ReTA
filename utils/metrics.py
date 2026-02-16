"""
Metrics: AUPRC / Micro-F1 / Acc@K.

Inputs:
- y_true: (N, L) multi-hot {0,1}
- y_score: (N, L) probabilities in [0,1]
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if "torch" in str(type(x)):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def auprc_micro(y_true, y_score) -> float:
    """Micro-averaged AUPRC."""
    yt = _to_numpy(y_true).astype(np.int32)
    ys = _to_numpy(y_score).astype(np.float32)
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(yt.reshape(-1), ys.reshape(-1)))
    except Exception:
        yt_f = yt.reshape(-1)
        ys_f = ys.reshape(-1)
        order = np.argsort(-ys_f)
        yt_f = yt_f[order]
        cumsum = np.cumsum(yt_f)
        idx_pos = np.where(yt_f == 1)[0]
        if len(idx_pos) == 0:
            return 0.0
        prec = cumsum[idx_pos] / (idx_pos + 1)
        return float(np.mean(prec))


def micro_f1(y_true, y_pred_bin) -> float:
    """Micro F1 for binary multi-label predictions."""
    yt = _to_numpy(y_true).astype(np.int32)
    yp = _to_numpy(y_pred_bin).astype(np.int32)
    tp = (yt * yp).sum()
    fp = ((1 - yt) * yp).sum()
    fn = (yt * (1 - yp)).sum()
    denom = (2 * tp + fp + fn)
    return float(2 * tp / denom) if denom > 0 else 0.0


def acc_at_k(y_true, y_score, k: int = 20) -> float:
    """Acc@K: fraction of samples with at least one true label in top-K predictions."""
    yt = _to_numpy(y_true).astype(np.int32)
    ys = _to_numpy(y_score).astype(np.float32)
    if yt.size == 0:
        return 0.0
    k = min(int(k), ys.shape[1])
    topk = np.argpartition(-ys, kth=k - 1, axis=1)[:, :k]
    hit = 0
    for i in range(yt.shape[0]):
        if yt[i, topk[i]].sum() > 0:
            hit += 1
    return float(hit / yt.shape[0])


def compute_all(y_true, logits=None, probs=None, threshold: float = 0.5, k: int = 20) -> Dict[str, float]:
    """Compute a standard metric bundle. Provide either `probs` or `logits`."""
    if probs is None:
        if logits is None:
            raise ValueError("Provide probs or logits.")
        lg = _to_numpy(logits).astype(np.float32)
        probs = 1.0 / (1.0 + np.exp(-lg))

    yt = _to_numpy(y_true).astype(np.int32)
    ps = _to_numpy(probs).astype(np.float32)
    yp = (ps >= float(threshold)).astype(np.int32)

    return {
        "AUPRC_micro": auprc_micro(yt, ps),
        f"MicroF1@{threshold:.2f}": micro_f1(yt, yp),
        f"Acc@{int(k)}": acc_at_k(yt, ps, k=int(k)),
        "num_samples": int(yt.shape[0]),
        "num_labels": int(yt.shape[1]) if yt.ndim == 2 else 0,
    }
