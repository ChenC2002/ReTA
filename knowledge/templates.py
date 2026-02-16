"""
This package implements the *offline* knowledge-pool construction:
- Each medical concept is distilled into two fields: (1) Definition, (2) Clinical Cascade (3 items).
- Definition parameterizes Soft Import; Cascade materializes a compact subgraph for Hard Import.

We keep the implementation modular so you can plug in:
- LLM distillation backend
- grounding inventory (UMLS / CCS / ICD)
- external support graph

All objects here are serializable to/from dict for JSONL caching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class DistilledArtifact:
    """Raw distilled artifact (before grounding)."""
    concept_id: str
    concept_name: str
    definition: str
    cascade: List[str]                 # exactly 3 strings
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
