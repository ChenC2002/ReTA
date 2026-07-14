"""
Template embedding + clustering using pinned raw-transformer mean pooling.

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
from sklearn.decomposition import TruncatedSVD

from .pool import (
    GroundedTemplate,
    KnowledgeTemplate,
    strict_json_loads,
    validate_grounded_template,
    validate_template_pool,
)


DEFAULT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
DEFAULT_MODEL_REVISION = "d5892b39a4adaed74b92212a44081509db72f87b"
EMBEDDING_BACKEND = "transformers_mean_pooling"


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    if not np.isfinite(n).all() or np.any(n <= eps):
        raise ValueError("cannot cosine-normalize a zero or non-finite embedding")
    return x / n


def project_to_dim(X: np.ndarray, dim: int) -> np.ndarray:
    """Project cluster centroids into the model embedding space.

    Template vectors ``p_k`` live in ``R^d``. Transformer encoders may produce
    a different dimensionality, so we use a deterministic projection after
    clustering and computing centroids in the original embedding space.
    """
    dim = int(dim)
    if dim <= 0:
        raise ValueError("projection_dim must be positive.")
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("template embeddings must be a non-empty 2D matrix")
    if not np.isfinite(X).all():
        raise ValueError("template embeddings contain NaN or infinite values")
    if X.shape[1] == dim:
        return X
    if X.shape[1] < dim:
        pad = np.zeros((X.shape[0], dim - X.shape[1]), dtype=np.float32)
        return np.concatenate([X, pad], axis=1)

    max_components = min(X.shape[0] - 1, X.shape[1])
    if max_components >= dim and X.shape[0] > 1:
        svd = TruncatedSVD(n_components=dim, random_state=0)
        return svd.fit_transform(X).astype(np.float32)

    # Very tiny pools cannot support SVD to the requested rank.
    return X[:, :dim].astype(np.float32)


class TextEmbedder:
    """Clinical text embedder using one explicit mean-pooling implementation."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        revision: str = DEFAULT_MODEL_REVISION,
        backend: str = EMBEDDING_BACKEND,
    ):
        if backend != EMBEDDING_BACKEND:
            raise ValueError(f"unsupported embedding backend {backend!r}; expected {EMBEDDING_BACKEND!r}")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be non-empty")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("model_revision must be non-empty and immutable")
        self.model_name = model_name.strip()
        self.revision = revision.strip()
        self.backend_kind = backend
        self.backend = None
        self.tokenizer = None
        self._init_backend()

    def _init_backend(self):
        try:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
            self.backend = AutoModel.from_pretrained(self.model_name, revision=self.revision)
            self.backend.eval()
        except Exception as exc:
            raise RuntimeError(
                f"unable to load embedding model {self.model_name!r} at revision {self.revision!r} "
                f"with backend {self.backend_kind!r}"
            ) from exc

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("cannot embed an empty text list")
        import torch

        outs = []
        with torch.no_grad():
            for start in range(0, len(texts), 32):
                batch = texts[start : start + 32]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
                model_device = next(self.backend.parameters()).device
                enc = {k: v.to(model_device) for k, v in enc.items()}
                hidden = self.backend(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                outs.append(pooled.cpu().numpy())
        result = np.concatenate(outs, axis=0).astype(np.float32)
        if result.ndim != 2 or result.shape[0] != len(texts) or not np.isfinite(result).all():
            raise ValueError("embedding backend returned an invalid matrix")
        return result


def template_text(gt: GroundedTemplate) -> str:
    cascade = "; ".join([e.name for e in gt.cascade_entities])
    return (gt.definition or "").strip() + " " + cascade


def load_grounded_templates(path: str) -> List[GroundedTemplate]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    template = GroundedTemplate.from_dict(strict_json_loads(line))
                except Exception as exc:
                    raise ValueError(f"{path}:{line_no}: invalid grounded JSON: {exc}") from exc
                errors = validate_grounded_template(template)
                if errors:
                    raise ValueError(f"{path}:{line_no}: invalid grounded template: {'; '.join(errors)}")
                out.append(template)
    if not out:
        raise ValueError(f"{path}: grounded template JSONL is empty")
    return out


def save_templates_jsonl(templates: List[KnowledgeTemplate], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in templates:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")


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
    tau: float = 0.16,
    model_name: str = DEFAULT_MODEL_NAME,
    projection_dim: int = 256,
    model_revision: str = DEFAULT_MODEL_REVISION,
    backend: str = EMBEDDING_BACKEND,
) -> List[KnowledgeTemplate]:
    if len(grounded) == 0:
        raise ValueError("grounded template list is empty; cannot build a knowledge pool.")
    tau = float(tau)
    if not np.isfinite(tau) or not 0.0 <= tau <= 2.0:
        raise ValueError("tau must be finite and in [0, 2].")
    for index, template in enumerate(grounded):
        errors = validate_grounded_template(template)
        if errors:
            raise ValueError(f"grounded template {index} is invalid: {'; '.join(errors)}")

    texts = [template_text(gt) for gt in grounded]
    embedder = TextEmbedder(model_name=model_name, revision=model_revision, backend=backend)
    original = embedder.encode(texts)
    original_normalized = l2_normalize(original, axis=1)

    if len(grounded) == 1:
        labels = np.zeros(1, dtype=np.int64)
    else:
        # The paper clusters in the original ClinicalBERT embedding space.
        # Projection is deliberately deferred until after cluster centroids and
        # medoids have been computed.
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None, metric="cosine", linkage="average", distance_threshold=tau
            )
            labels = clustering.fit_predict(original_normalized)
        except TypeError:
            clustering = AgglomerativeClustering(
                n_clusters=None, affinity="cosine", linkage="average", distance_threshold=tau
            )
            labels = clustering.fit_predict(original_normalized)

    cluster_records = []
    for lab in sorted(set(labels.tolist())):
        indices = np.where(labels == lab)[0].tolist()
        centroid = l2_normalize(
            original_normalized[indices].mean(axis=0, keepdims=True),
            axis=1,
        )[0]
        medoid_idx = choose_medoid(indices, original_normalized, grounded)
        cluster_records.append((indices, centroid, medoid_idx))

    original_centroids = np.stack(
        [centroid for _, centroid, _ in cluster_records],
        axis=0,
    ).astype(np.float32)
    projected_centroids = l2_normalize(
        project_to_dim(original_centroids, projection_dim),
        axis=1,
    )

    templates: List[KnowledgeTemplate] = []
    for tid, ((indices, _, medoid_idx), vector) in enumerate(
        zip(cluster_records, projected_centroids)
    ):
        templates.append(
            KnowledgeTemplate(
                template_id=tid,
                vector=vector.astype(float).tolist(),
                medoid_idx=int(medoid_idx),
                medoid=grounded[int(medoid_idx)],
                member_indices=[int(i) for i in indices],
            )
        )
    errors = validate_template_pool(templates, expected_dim=projection_dim)
    if errors:
        raise ValueError("invalid clustered template pool: " + "; ".join(errors))
    return templates


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Cluster grounded templates into knowledge pool.")
    p.add_argument("--grounded_jsonl", type=str, required=True)
    p.add_argument("--out_jsonl", type=str, required=True)
    p.add_argument("--tau", type=float, default=0.16)
    p.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    p.add_argument("--model_revision", type=str, default=DEFAULT_MODEL_REVISION)
    p.add_argument("--backend", choices=[EMBEDDING_BACKEND], default=EMBEDDING_BACKEND)
    p.add_argument("--projection_dim", type=int, default=256, help="Template vector dimension d used by the encoder.")
    return p


def main():
    args = build_argparser().parse_args()
    grounded = load_grounded_templates(args.grounded_jsonl)
    templates = cluster_templates(
        grounded,
        tau=args.tau,
        model_name=args.model_name,
        projection_dim=args.projection_dim,
        model_revision=args.model_revision,
        backend=args.backend,
    )
    save_templates_jsonl(templates, args.out_jsonl)
    print(f"[clustering] wrote {len(templates)} templates to {args.out_jsonl}")


if __name__ == "__main__":
    main()
