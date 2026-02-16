"""
Template embedding + clustering.

Input: grounded.jsonl
Output: templates.jsonl (KnowledgeTemplate)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from .templates import GroundedTemplate, KnowledgeTemplate
except ImportError:  # allow running as a script
    from templates import GroundedTemplate, KnowledgeTemplate  # type: ignore


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


class TextEmbedder:
    """Pluggable embedder with safe fallbacks."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.backend = None
        self.vectorizer = None
        self._init_backend()

    def _init_backend(self):
        try:
            from sentence_transformers import SentenceTransformer

            self.backend = SentenceTransformer(self.model_name)
        except Exception:
            self.backend = None

    def fit_if_needed(self, texts: List[str]):
        if self.backend is None:
            self.vectorizer = TfidfVectorizer(max_features=4096, ngram_range=(1, 2))
            self.vectorizer.fit(texts)

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.backend is not None:
            emb = self.backend.encode(texts, show_progress_bar=False, normalize_embeddings=False)
            return np.array(emb, dtype=np.float32)
        assert self.vectorizer is not None, "Call fit_if_needed() first for TF-IDF fallback."
        X = self.vectorizer.transform(texts)
        return X.toarray().astype(np.float32)


def template_text(gt: GroundedTemplate) -> str:
    cascade = "; ".join([e.name for e in gt.cascade_entities])
    return (gt.definition or "").strip() + " " + cascade


def load_grounded_templates(path: str) -> List[GroundedTemplate]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(GroundedTemplate.from_dict(json.loads(line)))
    return out


def save_templates_jsonl(templates: List[KnowledgeTemplate], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in templates:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")


def choose_medoid(cluster_indices: List[int], X: np.ndarray, gts: List[GroundedTemplate]) -> int:
    """Choose representative medoid with deterministic tie-break."""
    centroid = l2_normalize(X[cluster_indices].mean(axis=0, keepdims=True), axis=1)[0]
    sims = (l2_normalize(X[cluster_indices], axis=1) @ centroid.reshape(-1, 1)).reshape(-1)
    best_sim = sims.max()
    candidate_local = np.where(np.isclose(sims, best_sim, atol=1e-6))[0].tolist()

    if len(candidate_local) == 1:
        return cluster_indices[candidate_local[0]]

    def key_func(global_idx: int):
        v, e = gts[global_idx].subgraph_size
        return (e, v, gts[global_idx].root_concept_id)

    candidates = [cluster_indices[i] for i in candidate_local]
    candidates.sort(key=key_func)
    return candidates[0]


def cluster_templates(
    grounded: List[GroundedTemplate],
    tau: float = 0.15,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[KnowledgeTemplate]:
    texts = [template_text(gt) for gt in grounded]
    embedder = TextEmbedder(model_name=model_name)
    embedder.fit_if_needed(texts)
    X = embedder.encode(texts)
    Xn = l2_normalize(X, axis=1)

    # sklearn compatibility: metric vs affinity
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=tau
        )
        labels = clustering.fit_predict(Xn)
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=None, affinity="cosine", linkage="average", distance_threshold=tau
        )
        labels = clustering.fit_predict(Xn)

    templates: List[KnowledgeTemplate] = []
    for tid, lab in enumerate(sorted(set(labels.tolist()))):
        idx = np.where(labels == lab)[0].tolist()
        centroid = l2_normalize(Xn[idx].mean(axis=0, keepdims=True), axis=1)[0]
        medoid_idx = choose_medoid(idx, X, grounded)
        templates.append(
            KnowledgeTemplate(
                template_id=tid,
                vector=centroid.astype(float).tolist(),
                medoid_idx=int(medoid_idx),
                medoid=grounded[int(medoid_idx)],
                member_indices=[int(i) for i in idx],
            )
        )
    return templates


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cluster grounded templates into knowledge pool.")
    p.add_argument("--grounded_jsonl", type=str, required=True)
    p.add_argument("--out_jsonl", type=str, required=True)
    p.add_argument("--tau", type=float, default=0.15)
    p.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    return p


def main():
    args = build_argparser().parse_args()
    grounded = load_grounded_templates(args.grounded_jsonl)
    templates = cluster_templates(grounded, tau=args.tau, model_name=args.model_name)
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    save_templates_jsonl(templates, args.out_jsonl)
    print(f"[clustering] wrote {len(templates)} templates to {args.out_jsonl}")


if __name__ == "__main__":
    main()
