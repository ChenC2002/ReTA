"""
This package implements the *offline* knowledge-pool construction:
- Each medical concept is distilled into two fields: (1) Definition, (2) adaptive Clinical Cascade.
- Definition parameterizes Soft Import; Cascade materializes a compact subgraph for Hard Import.

We keep the implementation modular so you can plug in:
- LLM distillation backend
- grounding inventory (UMLS / CCS / ICD)
- external support graph

All objects here are serializable to/from dict for JSONL caching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class DistilledArtifact:
    """Raw distilled artifact (before grounding).

    Adaptive clinical cascade length: dense KG neighborhoods may use one item
    while sparse neighborhoods may use up to five.
    """
    concept_id: str
    concept_name: str
    definition: str
    cascade: List[str]                 # 1-5 strings after adaptive distillation
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
    """A grounded entity in an external ontology/KG."""
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
            score=float(d.get("score", 1.0)),
        )


@dataclass
class GroundedTemplate:
    """A grounded knowledge template for one concept (pre-clustering)."""
    root_concept_id: str
    root_name: str
    definition: str
    cascade_entities: List[GroundedEntity]

    # optional compact subgraph for Hard Import
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
            subgraph_edges=[tuple(x) for x in d.get("subgraph_edges", [])],
            meta=dict(d.get("meta", {})),
        )

    @property
    def subgraph_size(self) -> Tuple[int, int]:
        return (len(self.subgraph_nodes), len(self.subgraph_edges))


@dataclass
class KnowledgeTemplate:
    """Clustered template used online."""
    template_id: int
    vector: List[float]            # L2-normalized
    medoid_idx: int                # index in grounded list
    medoid: GroundedTemplate       # representative with compact subgraph
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
            template_id=int(d["template_id"]),
            vector=list(d["vector"]),
            medoid_idx=int(d.get("medoid_idx", -1)),
            medoid=GroundedTemplate.from_dict(d["medoid"]),
            member_indices=list(d.get("member_indices", [])),
        )


def validate_grounded_template(template: GroundedTemplate) -> List[str]:
    """Validate the compact subgraph contract used by Hard Import."""
    errors: List[str] = []
    if not template.root_concept_id:
        errors.append("missing root_concept_id")
    if not template.definition:
        errors.append(f"{template.root_concept_id}: missing definition")

    node_set = set(template.subgraph_nodes)
    if template.root_concept_id and template.root_concept_id not in node_set:
        errors.append(f"{template.root_concept_id}: root_concept_id is absent from subgraph_nodes")
    if len(node_set) != len(template.subgraph_nodes):
        errors.append(f"{template.root_concept_id}: duplicate subgraph_nodes")

    seen_edges = set()
    for u, v in template.subgraph_edges:
        if u == v:
            errors.append(f"{template.root_concept_id}: self-loop edge {u}->{v}")
        if u not in node_set:
            errors.append(f"{template.root_concept_id}: edge references unknown node {u}")
        if v not in node_set:
            errors.append(f"{template.root_concept_id}: edge references unknown node {v}")
        key = (u, v)
        if key in seen_edges:
            errors.append(f"{template.root_concept_id}: duplicate edge {u}->{v}")
        seen_edges.add(key)
    return errors


def validate_knowledge_template(template: KnowledgeTemplate, expected_dim: Optional[int] = None) -> List[str]:
    """Validate one online retrieval template."""
    errors: List[str] = []
    if template.template_id < 0:
        errors.append("template_id must be non-negative")
    if not template.vector:
        errors.append(f"template {template.template_id}: empty vector")
    if expected_dim is not None and len(template.vector) != int(expected_dim):
        errors.append(
            f"template {template.template_id}: vector dim {len(template.vector)} != expected {int(expected_dim)}"
        )
    norm_sq = 0.0
    for j, value in enumerate(template.vector):
        if not isinstance(value, (int, float)):
            errors.append(f"template {template.template_id}: non-numeric vector value at {j}")
            break
        if not math.isfinite(float(value)):
            errors.append(f"template {template.template_id}: non-finite vector value at {j}")
            break
        norm_sq += float(value) * float(value)
    if template.vector and norm_sq <= 0.0:
        errors.append(f"template {template.template_id}: zero vector")
    if template.medoid_idx < 0:
        errors.append(f"template {template.template_id}: medoid_idx must be non-negative")
    if not template.member_indices:
        errors.append(f"template {template.template_id}: empty member_indices")
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
                templates.append(KnowledgeTemplate.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid template JSON: {exc}") from exc
    return templates


def validate_template_pool(templates: Iterable[KnowledgeTemplate], expected_dim: Optional[int] = None) -> List[str]:
    errors: List[str] = []
    seen_ids = set()
    count = 0
    for template in templates:
        count += 1
        if template.template_id in seen_ids:
            errors.append(f"duplicate template_id {template.template_id}")
        seen_ids.add(template.template_id)
        errors.extend(validate_knowledge_template(template, expected_dim=expected_dim))
    if count == 0:
        errors.append("template pool is empty")
    return errors
