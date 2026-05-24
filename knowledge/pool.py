"""Knowledge pool and history-aware Top-K retrieval.

The pool stores template vectors ``p_k``. Retrieval combines current-code
similarity with an optional trajectory state similarity:

``(1-alpha) * max_i cos(e_ci, p_k) + alpha * cos(s_t, p_k)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

try:
    from .templates import KnowledgeTemplate
except ImportError:  # allow running as a script
    from templates import KnowledgeTemplate  # type: ignore


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def _coerce_dim(x: np.ndarray, dim: int) -> np.ndarray:
    """Pad or truncate a vector/matrix to ``dim`` for backward-compatible pools."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.shape[-1] == dim:
        return arr
    if arr.shape[-1] > dim:
        return arr[..., :dim]
    pad_width = [(0, 0)] * arr.ndim
    pad_width[-1] = (0, dim - arr.shape[-1])
    return np.pad(arr, pad_width, mode="constant")


@dataclass
class RetrievalResult:
    template_ids: List[int]
    scores: List[float]


class KnowledgePool:
    """A pool of clustered knowledge templates."""

    def __init__(self, templates: Sequence[KnowledgeTemplate]):
        self.templates = list(templates)
        if len(self.templates) == 0:
            raise ValueError("KnowledgePool requires at least one template.")

        vectors = []
        expected_dim = None
        seen_ids = set()
        for t in self.templates:
            if int(t.template_id) in seen_ids:
                raise ValueError(f"Duplicate template_id in KnowledgePool: {t.template_id}")
            seen_ids.add(int(t.template_id))

            v = np.asarray(t.vector, dtype=np.float32)
            if v.ndim != 1 or v.size == 0:
                raise ValueError(f"Template {t.template_id} must have a non-empty 1D vector.")
            if not np.isfinite(v).all():
                raise ValueError(f"Template {t.template_id} vector contains NaN or infinite values.")
            if expected_dim is None:
                expected_dim = int(v.shape[0])
            elif int(v.shape[0]) != expected_dim:
                raise ValueError(
                    f"Template {t.template_id} vector dim {v.shape[0]} != expected {expected_dim}."
                )
            vectors.append(v)

        mat = np.stack(vectors, axis=0).astype(np.float32)
        self.P = l2_normalize(mat, axis=1)  # (M, d)
        self.id_to_index = {int(t.template_id): i for i, t in enumerate(self.templates)}

    @staticmethod
    def load_jsonl(path: str) -> "KnowledgePool":
        import json

        templates = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    templates.append(KnowledgeTemplate.from_dict(json.loads(line)))
        return KnowledgePool(templates)

    def save_jsonl(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as f:
            for t in self.templates:
                f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")

    def retrieve_topk(
        self,
        visit_tokens: Sequence[int],
        code_embed_lookup: Callable[[int], np.ndarray],
        K: int = 20,
        state_vector: Optional[np.ndarray] = None,
        alpha: float = 0.2,
    ) -> RetrievalResult:
        """Retrieve Top-K templates using code- and trajectory-level context."""
        if K <= 0:
            return RetrievalResult([], [])
        alpha = float(alpha)
        if state_vector is None:
            alpha = 0.0
        alpha = min(max(alpha, 0.0), 1.0)

        if len(visit_tokens) == 0:
            k = min(K, len(self.templates))
            return RetrievalResult([t.template_id for t in self.templates[:k]], [0.0] * k)

        dim = self.P.shape[1]
        E = np.stack([_coerce_dim(code_embed_lookup(int(tok)), dim) for tok in visit_tokens], axis=0)
        E = l2_normalize(E, axis=1)  # (n, d)
        sims = E @ self.P.T          # (n, M)
        code_scores = sims.max(axis=0)    # (M,)

        if state_vector is not None and alpha > 0.0:
            s = _coerce_dim(np.asarray(state_vector, dtype=np.float32), dim).reshape(1, -1)
            s = l2_normalize(s, axis=1)
            state_scores = (s @ self.P.T).reshape(-1)
            scores = (1.0 - alpha) * code_scores + alpha * state_scores
        else:
            scores = code_scores

        k = min(K, scores.shape[0])
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        template_ids = [self.templates[i].template_id for i in idx.tolist()]
        return RetrievalResult(template_ids=template_ids, scores=scores[idx].astype(float).tolist())

    def get_template(self, template_id: int) -> KnowledgeTemplate:
        return self.templates[self.id_to_index[int(template_id)]]
