"""
Knowledge pool and Top-K retrieval.

You provide a code embedding lookup function (token -> vector).
The pool stores template vectors p_k (L2-normalized).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np

try:
    from .templates import KnowledgeTemplate
except ImportError:  # allow running as a script
    from templates import KnowledgeTemplate  # type: ignore


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


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
        mat = np.array([t.vector for t in self.templates], dtype=np.float32)
        self.P = l2_normalize(mat, axis=1)  # (M, d)
        self.id_to_index = {t.template_id: i for i, t in enumerate(self.templates)}

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
    ) -> RetrievalResult:
        """Retrieve Top-K templates by max cosine similarity."""
        if len(visit_tokens) == 0:
            k = min(K, len(self.templates))
            return RetrievalResult([t.template_id for t in self.templates[:k]], [0.0] * k)

        E = np.stack([code_embed_lookup(int(tok)).astype(np.float32) for tok in visit_tokens], axis=0)
        E = l2_normalize(E, axis=1)  # (n, d)
        sims = E @ self.P.T          # (n, M)
        scores = sims.max(axis=0)    # (M,)

        k = min(K, scores.shape[0])
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        template_ids = [self.templates[i].template_id for i in idx.tolist()]
        return RetrievalResult(template_ids=template_ids, scores=scores[idx].astype(float).tolist())

    def get_template(self, template_id: int) -> KnowledgeTemplate:
        return self.templates[self.id_to_index[int(template_id)]]
