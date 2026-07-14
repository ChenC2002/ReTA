"""Knowledge artifacts, pool validation, and history-aware Top-K retrieval.

This module owns the complete knowledge-pool contract, from distilled and
grounded artifacts through clustered online templates.  The pool stores
template vectors ``p_k``. Retrieval combines current-code similarity with an
optional trajectory state similarity:

``(1-alpha) * max_i cos(e_ci, p_k) + alpha * cos(s_t, p_k)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ModuleNotFoundError:  # Contract-only workflows do not need retrieval dependencies.
    np = None  # type: ignore[assignment]


def _require_numpy():
    if np is None:
        raise ModuleNotFoundError("NumPy is required for knowledge-pool retrieval.")
    return np


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _reject_json_constant(value)
    return parsed


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r} is not allowed")
        value[key] = item
    return value


def strict_json_loads(value: str) -> Any:
    """Parse strict JSON, rejecting duplicate keys, NaN, and infinity."""

    return json.loads(
        value,
        parse_constant=_reject_json_constant,
        parse_float=_parse_finite_json_float,
        object_pairs_hook=_reject_duplicate_json_keys,
    )


def _is_finite_json(value: Any) -> bool:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


@dataclass
class DistilledArtifact:
    """Raw distilled artifact before grounding.

    Adaptive clinical cascade length: dense KG neighborhoods may use one item
    while sparse neighborhoods may use up to five.
    """

    concept_id: str
    concept_name: str
    definition: str
    cascade: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "concept_name": self.concept_name,
            "definition": self.definition,
            "cascade": list(self.cascade),
            "meta": dict(self.meta),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DistilledArtifact":
        return DistilledArtifact(
            concept_id=d["concept_id"],
            concept_name=d.get("concept_name", ""),
            definition=d.get("definition", ""),
            cascade=list(d.get("cascade", [])),
            meta=dict(d.get("meta", {})),
        )


@dataclass
class GroundedEntity:
    """A grounded entity in an external ontology or knowledge graph."""

    entity_id: str
    name: str
    source: str = "unknown"
    score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "source": self.source,
            "score": float(self.score),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GroundedEntity":
        return GroundedEntity(
            entity_id=d["entity_id"],
            name=d.get("name", ""),
            source=d.get("source", "unknown"),
            score=d.get("score", 1.0),
        )


@dataclass
class GroundedTemplate:
    """A grounded knowledge template for one concept before clustering."""

    root_concept_id: str
    root_name: str
    definition: str
    cascade_entities: List[GroundedEntity]
    subgraph_nodes: List[str] = field(default_factory=list)
    subgraph_edges: List[Tuple[str, str]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_concept_id": self.root_concept_id,
            "root_name": self.root_name,
            "definition": self.definition,
            "cascade_entities": [e.to_dict() for e in self.cascade_entities],
            "subgraph_nodes": list(self.subgraph_nodes),
            "subgraph_edges": [(u, v) for (u, v) in self.subgraph_edges],
            "meta": dict(self.meta),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GroundedTemplate":
        return GroundedTemplate(
            root_concept_id=d["root_concept_id"],
            root_name=d.get("root_name", ""),
            definition=d.get("definition", ""),
            cascade_entities=[GroundedEntity.from_dict(x) for x in d.get("cascade_entities", [])],
            subgraph_nodes=list(d.get("subgraph_nodes", [])),
            subgraph_edges=list(d.get("subgraph_edges", [])),
            meta=dict(d.get("meta", {})),
        )

    @property
    def subgraph_size(self) -> Tuple[int, int]:
        return (len(self.subgraph_nodes), len(self.subgraph_edges))


@dataclass
class KnowledgeTemplate:
    """A clustered template used for online retrieval."""

    template_id: int
    vector: List[float]
    medoid_idx: int
    medoid: GroundedTemplate
    member_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_id": int(self.template_id),
            "vector": list(self.vector),
            "medoid_idx": int(self.medoid_idx),
            "medoid": self.medoid.to_dict(),
            "member_indices": list(self.member_indices),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "KnowledgeTemplate":
        return KnowledgeTemplate(
            template_id=d["template_id"],
            vector=list(d["vector"]),
            medoid_idx=d.get("medoid_idx", -1),
            medoid=GroundedTemplate.from_dict(d["medoid"]),
            member_indices=list(d.get("member_indices", [])),
        )


def validate_grounded_template(template: GroundedTemplate) -> List[str]:
    """Validate the compact subgraph contract used by Hard Import."""

    errors: List[str] = []
    if not isinstance(template.root_concept_id, str) or not template.root_concept_id.strip():
        errors.append("missing root_concept_id")
    if not isinstance(template.root_name, str):
        errors.append(f"{template.root_concept_id}: root_name must be a string")
    if not isinstance(template.definition, str) or not template.definition.strip():
        errors.append(f"{template.root_concept_id}: missing definition")

    valid_nodes: List[str] = []
    for index, node in enumerate(template.subgraph_nodes):
        if not isinstance(node, str) or not node.strip():
            errors.append(f"{template.root_concept_id}: invalid subgraph node at {index}")
        else:
            valid_nodes.append(node)
    node_set = set(valid_nodes)
    if template.root_concept_id and template.root_concept_id not in node_set:
        errors.append(f"{template.root_concept_id}: root_concept_id is absent from subgraph_nodes")
    if len(node_set) != len(valid_nodes):
        errors.append(f"{template.root_concept_id}: duplicate subgraph_nodes")

    seen_edges = set()
    adjacency: Dict[str, List[str]] = {node: [] for node in node_set}
    for edge_index, edge in enumerate(template.subgraph_edges):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            errors.append(f"{template.root_concept_id}: invalid edge at {edge_index}; expected two endpoints")
            continue
        u, v = edge
        if not isinstance(u, str) or not isinstance(v, str) or not u or not v:
            errors.append(f"{template.root_concept_id}: invalid edge endpoint at {edge_index}")
            continue
        if u == v:
            errors.append(f"{template.root_concept_id}: self-loop edge {u}->{v}")
        if u not in node_set:
            errors.append(f"{template.root_concept_id}: edge references unknown node {u}")
        if v not in node_set:
            errors.append(f"{template.root_concept_id}: edge references unknown node {v}")
        key = tuple(sorted((u, v)))
        if key in seen_edges:
            errors.append(f"{template.root_concept_id}: duplicate undirected edge {u}<->{v}")
        seen_edges.add(key)
        if u in node_set and v in node_set and u != v:
            adjacency[u].append(v)
            adjacency[v].append(u)

    seen_entities = set()
    if not template.cascade_entities:
        errors.append(f"{template.root_concept_id}: empty cascade_entities")
    for index, entity in enumerate(template.cascade_entities):
        if not isinstance(entity, GroundedEntity):
            errors.append(f"{template.root_concept_id}: invalid grounded entity at {index}")
            continue
        if not isinstance(entity.entity_id, str) or not entity.entity_id.strip():
            errors.append(f"{template.root_concept_id}: grounded entity {index} has no entity_id")
            continue
        if entity.entity_id in seen_entities:
            errors.append(f"{template.root_concept_id}: duplicate grounded entity {entity.entity_id}")
        seen_entities.add(entity.entity_id)
        if entity.entity_id == template.root_concept_id:
            errors.append(f"{template.root_concept_id}: cascade entity repeats the root")
        if entity.entity_id not in node_set:
            errors.append(f"{template.root_concept_id}: grounded entity {entity.entity_id} is absent from subgraph_nodes")
        if not isinstance(entity.name, str) or not entity.name.strip():
            errors.append(f"{template.root_concept_id}: grounded entity {entity.entity_id} has no name")
        if not isinstance(entity.source, str) or not entity.source.strip():
            errors.append(f"{template.root_concept_id}: grounded entity {entity.entity_id} has no source")
        if isinstance(entity.score, bool) or not isinstance(entity.score, (int, float)):
            errors.append(f"{template.root_concept_id}: grounded entity {entity.entity_id} has a non-numeric score")
        elif not math.isfinite(float(entity.score)) or not -1.0 <= float(entity.score) <= 1.0:
            errors.append(f"{template.root_concept_id}: grounded entity {entity.entity_id} score must be finite in [-1, 1]")

    if not isinstance(template.meta, dict) or not _is_finite_json(template.meta):
        errors.append(f"{template.root_concept_id}: meta must be a finite JSON object")

    if template.root_concept_id in node_set:
        reachable = {template.root_concept_id}
        frontier = [template.root_concept_id]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency.get(current, []):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    frontier.append(neighbor)
        disconnected = sorted(node_set - reachable)
        if disconnected:
            errors.append(
                f"{template.root_concept_id}: subgraph contains root-disconnected nodes: {', '.join(disconnected)}"
            )
    return errors


def validate_knowledge_template(template: KnowledgeTemplate, expected_dim: Optional[int] = None) -> List[str]:
    """Validate one online retrieval template."""

    errors: List[str] = []
    if isinstance(template.template_id, bool) or not isinstance(template.template_id, int) or template.template_id < 0:
        errors.append("template_id must be non-negative")
    if not template.vector:
        errors.append(f"template {template.template_id}: empty vector")
    if expected_dim is not None and len(template.vector) != int(expected_dim):
        errors.append(
            f"template {template.template_id}: vector dim {len(template.vector)} != expected {int(expected_dim)}"
        )
    norm_sq = 0.0
    for j, value in enumerate(template.vector):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"template {template.template_id}: non-numeric vector value at {j}")
            break
        if not math.isfinite(float(value)):
            errors.append(f"template {template.template_id}: non-finite vector value at {j}")
            break
        norm_sq += float(value) * float(value)
    if template.vector and norm_sq <= 0.0:
        errors.append(f"template {template.template_id}: zero vector")
    elif template.vector and abs(math.sqrt(norm_sq) - 1.0) > 1e-3:
        errors.append(f"template {template.template_id}: vector is not L2-normalized")
    if isinstance(template.medoid_idx, bool) or not isinstance(template.medoid_idx, int) or template.medoid_idx < 0:
        errors.append(f"template {template.template_id}: medoid_idx must be non-negative")
    if not template.member_indices:
        errors.append(f"template {template.template_id}: empty member_indices")
    valid_members: List[int] = []
    for member in template.member_indices:
        if isinstance(member, bool) or not isinstance(member, int) or member < 0:
            errors.append(f"template {template.template_id}: member_indices must contain non-negative integers")
            break
        valid_members.append(member)
    if len(set(valid_members)) != len(valid_members):
        errors.append(f"template {template.template_id}: duplicate member_indices")
    if (
        isinstance(template.medoid_idx, int)
        and not isinstance(template.medoid_idx, bool)
        and template.medoid_idx >= 0
        and template.medoid_idx not in template.member_indices
    ):
        errors.append(f"template {template.template_id}: medoid_idx is absent from member_indices")
    if not isinstance(template.medoid, GroundedTemplate):
        errors.append(f"template {template.template_id}: medoid must be a GroundedTemplate")
    else:
        errors.extend(f"template {template.template_id}: {msg}" for msg in validate_grounded_template(template.medoid))
    return errors


def load_templates_jsonl(path: str) -> List[KnowledgeTemplate]:
    templates: List[KnowledgeTemplate] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                templates.append(KnowledgeTemplate.from_dict(strict_json_loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid template JSON: {exc}") from exc
    return templates


def validate_template_pool(templates: Iterable[KnowledgeTemplate], expected_dim: Optional[int] = None) -> List[str]:
    errors: List[str] = []
    seen_ids = set()
    count = 0
    vector_dim: Optional[int] = None
    claimed_members: Dict[int, int] = {}
    for template in templates:
        count += 1
        if isinstance(template.template_id, int) and not isinstance(template.template_id, bool):
            if template.template_id in seen_ids:
                errors.append(f"duplicate template_id {template.template_id}")
            seen_ids.add(template.template_id)
        if vector_dim is None:
            vector_dim = len(template.vector)
        elif len(template.vector) != vector_dim:
            errors.append(
                f"template {template.template_id}: vector dim {len(template.vector)} != pool dim {vector_dim}"
            )
        for member in template.member_indices:
            if isinstance(member, int) and not isinstance(member, bool) and member >= 0:
                previous = claimed_members.get(member)
                if previous is not None and previous != template.template_id:
                    errors.append(
                        f"member index {member} appears in templates {previous} and {template.template_id}"
                    )
                claimed_members[member] = template.template_id
        errors.extend(validate_knowledge_template(template, expected_dim=expected_dim))
    if count == 0:
        errors.append("template pool is empty")
    return errors


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    np = _require_numpy()
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


@dataclass
class RetrievalResult:
    template_ids: List[int]
    scores: List[float]


class KnowledgePool:
    """A pool of clustered knowledge templates."""

    def __init__(self, templates: Sequence[KnowledgeTemplate]):
        np = _require_numpy()
        self.templates = list(templates)
        validation_errors = validate_template_pool(self.templates)
        if validation_errors:
            raise ValueError("invalid knowledge pool:\n" + "\n".join(validation_errors))

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
            if float(np.linalg.norm(v)) <= 0.0:
                raise ValueError(f"Template {t.template_id} vector must be non-zero.")
            vectors.append(v)

        mat = np.stack(vectors, axis=0).astype(np.float32)
        self.P = l2_normalize(mat, axis=1)  # (M, d)
        self.id_to_index = {int(t.template_id): i for i, t in enumerate(self.templates)}

    @staticmethod
    def load_jsonl(path: str) -> "KnowledgePool":
        return KnowledgePool(load_templates_jsonl(path))

    def save_jsonl(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for t in self.templates:
                f.write(json.dumps(t.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")

    def retrieve_topk(
        self,
        visit_tokens: Sequence[int],
        code_embed_lookup: Callable[[int], np.ndarray],
        K: int = 20,
        state_vector: Optional[np.ndarray] = None,
        alpha: float = 0.3,
    ) -> RetrievalResult:
        """Retrieve Top-K templates using code- and trajectory-level context."""
        np = _require_numpy()
        if isinstance(K, bool) or not isinstance(K, int):
            raise TypeError("K must be an integer")
        if K <= 0:
            return RetrievalResult([], [])
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be finite and in [0, 1]")
        if state_vector is None:
            alpha = 0.0

        if len(visit_tokens) == 0:
            k = min(K, len(self.templates))
            template_ids = sorted(int(t.template_id) for t in self.templates)[:k]
            return RetrievalResult(template_ids, [0.0] * k)

        dim = self.P.shape[1]
        vectors = []
        for tok in visit_tokens:
            vector = np.asarray(code_embed_lookup(int(tok)), dtype=np.float32)
            if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
                raise ValueError(f"code embedding for token {tok} must be a finite, non-empty 1D vector")
            if int(vector.shape[0]) != dim:
                raise ValueError(f"code embedding for token {tok} has dim {vector.shape[0]}; expected {dim}")
            vectors.append(vector)
        E = np.stack(vectors, axis=0)
        E = l2_normalize(E, axis=1)  # (n, d)
        sims = E @ self.P.T          # (n, M)
        code_scores = sims.max(axis=0)    # (M,)

        if state_vector is not None and alpha > 0.0:
            state = np.asarray(state_vector, dtype=np.float32)
            if state.ndim != 1 or state.size == 0 or not np.isfinite(state).all():
                raise ValueError("state_vector must be a finite, non-empty 1D vector")
            if int(state.shape[0]) != dim:
                raise ValueError(f"state_vector has dim {state.shape[0]}; expected {dim}")
            s = state.reshape(1, -1)
            s = l2_normalize(s, axis=1)
            state_scores = (s @ self.P.T).reshape(-1)
            scores = (1.0 - alpha) * code_scores + alpha * state_scores
        else:
            scores = code_scores

        k = min(K, scores.shape[0])
        if not np.isfinite(scores).all():
            raise ValueError("retrieval produced non-finite scores")
        template_id_array = np.asarray([int(t.template_id) for t in self.templates], dtype=np.int64)
        idx = np.lexsort((template_id_array, -scores))[:k]
        template_ids = [self.templates[i].template_id for i in idx.tolist()]
        return RetrievalResult(template_ids=template_ids, scores=scores[idx].astype(float).tolist())

    def get_template(self, template_id: int) -> KnowledgeTemplate:
        return self.templates[self.id_to_index[int(template_id)]]
