"""Deterministic grounding and external-support filtering for ReTA knowledge pools.

This module does not call an embedding model. ClinicalBERT candidates are
computed separately and supplied as one Top-1 candidate per mention, keeping
the release filter deterministic and the model output auditable.

Input contracts
---------------
``artifacts_jsonl`` follows
``releases/pool_v1/schemas/distilled_artifact.schema.json``.
``inventory_csv`` requires ``entity_id,name`` and may include ``source``.
``primekg_edges_csv`` requires ``u,v`` and may include ``relation``; every row
is treated as one *direct* PrimeKG edge.  ``ccs_edges_csv`` requires
``parent,child`` and is kept directed so sibling paths cannot be mistaken for
ancestor/descendant evidence.  Precomputed candidates are CSV or JSONL records
with ``mention,entity_id,score``.

The filter is fail-closed.  A valid relation is either a direct PrimeKG edge or
a consistently directed CCS ancestor/descendant path of at most two edges.
Actual CCS path edges and intermediate nodes are preserved in the output.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .pool import strict_json_loads


TAU_MAP = 0.90
MAX_CCS_HOPS = 2
FILTER_SCHEMA_VERSION = "pool_v1.filtering.v1"
AUDIT_SCHEMA_VERSION = "pool_v1.filtering.audit.v1"
SUMMARY_SCHEMA_VERSION = "pool_v1.filtering.summary.v1"

_ARTIFACT_FIELDS = {"concept_id", "concept_name", "definition", "cascade", "meta"}
_SOURCE_PRIORITY_DESCRIPTION = "ICD, then CCS, then PrimeKG, then UMLS, then lexical source/id"


def canonicalize_ccs_id(value: Any) -> str:
    """Canonicalize a CCS identifier to exactly one ``CCS:`` prefix."""

    entity_id = _clean_id(value)
    if entity_id.upper().startswith("CCS:"):
        entity_id = entity_id[4:].strip()
    return f"CCS:{entity_id}" if entity_id else ""


def _canonical_entity_id(entity_id: Any, source: Any) -> str:
    cleaned = _clean_id(entity_id)
    if cleaned.upper().startswith("CCS:") or normalize_mention(str(source)).startswith("ccs"):
        return canonicalize_ccs_id(cleaned)
    return cleaned


def normalize_mention(value: str) -> str:
    """Normalize names for deterministic exact matching."""

    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _clean_id(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _source_rank(source: str) -> Tuple[int, str]:
    normalized = normalize_mention(source)
    if normalized.startswith("icd"):
        return (0, normalized)
    if normalized.startswith("ccs"):
        return (1, normalized)
    if normalized.startswith("primekg"):
        return (2, normalized)
    if normalized.startswith("umls"):
        return (3, normalized)
    return (9, normalized)


def _canonical_edge(u: str, v: str) -> Optional[Tuple[str, str]]:
    u, v = _clean_id(u), _clean_id(v)
    if not u or not v or u == v:
        return None
    return (u, v) if u < v else (v, u)


def _entity_dict(entity: "Entity", score: float) -> Dict[str, Any]:
    return {
        "entity_id": entity.entity_id,
        "name": entity.name,
        "source": entity.source,
        "score": float(score),
    }


@dataclass(frozen=True)
class Entity:
    entity_id: str
    name: str
    source: str = "unknown"

    def sort_key(self) -> Tuple[Any, ...]:
        return (_source_rank(self.source), self.entity_id, normalize_mention(self.name))


class Inventory:
    """Canonical entity inventory with deterministic exact-name tie-breaking."""

    def __init__(self, entities: Iterable[Entity]):
        by_id: Dict[str, Entity] = {}
        for raw in entities:
            source = str(raw.source or "unknown").strip()
            entity = Entity(
                _canonical_entity_id(raw.entity_id, source),
                str(raw.name).strip(),
                source,
            )
            if not entity.entity_id or not entity.name:
                raise ValueError("inventory entities require non-empty entity_id and name")
            previous = by_id.get(entity.entity_id)
            if previous is not None and previous != entity:
                raise ValueError(f"conflicting inventory rows for entity_id {entity.entity_id!r}")
            by_id[entity.entity_id] = entity

        if not by_id:
            raise ValueError("inventory requires at least one entity")

        exact: Dict[str, List[Entity]] = defaultdict(list)
        for entity in by_id.values():
            exact[normalize_mention(entity.name)].append(entity)
        self._by_id = by_id
        self._exact = {key: tuple(sorted(values, key=Entity.sort_key)) for key, values in exact.items()}

    @classmethod
    def from_csv(cls, path: str) -> "Inventory":
        rows = _read_csv(path, required=("entity_id", "name"))
        return cls(
            Entity(row["entity_id"], row["name"], row.get("source", "unknown") or "unknown")
            for _, row in rows
        )

    def get(self, entity_id: str) -> Optional[Entity]:
        cleaned = _clean_id(entity_id)
        direct = self._by_id.get(cleaned)
        if direct is not None:
            return direct
        return self._by_id.get(canonicalize_ccs_id(cleaned))

    def exact_matches(self, mention: str) -> Tuple[Entity, ...]:
        return self._exact.get(normalize_mention(mention), ())


@dataclass(frozen=True)
class EmbeddingCandidate:
    mention: str
    entity_id: str
    score: float


class EmbeddingCandidateIndex:
    """One externally computed Top-1 candidate per normalized mention."""

    def __init__(self, candidates: Iterable[EmbeddingCandidate] = ()):
        by_mention: Dict[str, EmbeddingCandidate] = {}
        for raw in candidates:
            normalized = normalize_mention(str(raw.mention))
            entity_id = _clean_id(raw.entity_id)
            score = float(raw.score)
            if not normalized or not entity_id or not math.isfinite(score) or not -1.0 <= score <= 1.0:
                raise ValueError("embedding candidates require mention, entity_id, and finite cosine score in [-1, 1]")
            candidate = EmbeddingCandidate(str(raw.mention).strip(), entity_id, score)
            previous = by_mention.get(normalized)
            if previous is not None and previous != candidate:
                raise ValueError(f"multiple precomputed Top-1 candidates for normalized mention {normalized!r}")
            by_mention[normalized] = candidate
        self._by_mention = by_mention

    @classmethod
    def from_path(cls, path: Optional[str]) -> "EmbeddingCandidateIndex":
        if not path:
            return cls()
        source = Path(path)
        candidates: List[EmbeddingCandidate] = []
        if source.suffix.casefold() == ".jsonl":
            for line_no, value, error in _read_jsonl_entries(path):
                if error is not None or not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_no}: invalid embedding-candidate JSON: {error or 'not an object'}")
                missing = [key for key in ("mention", "entity_id", "score") if key not in value]
                if missing:
                    raise ValueError(f"{path}:{line_no}: missing columns {missing}")
                candidates.append(EmbeddingCandidate(value["mention"], value["entity_id"], value["score"]))
        else:
            for _, row in _read_csv(path, required=("mention", "entity_id", "score")):
                candidates.append(EmbeddingCandidate(row["mention"], row["entity_id"], float(row["score"])))
        return cls(candidates)

    def get(self, mention: str) -> Optional[EmbeddingCandidate]:
        return self._by_mention.get(normalize_mention(mention))


@dataclass(frozen=True)
class MappingDecision:
    accepted: bool
    reason: str
    method: Optional[str]
    entity: Optional[Entity]
    score: Optional[float]
    evidence: Mapping[str, Any]

    def audit_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "method": self.method,
            "selected": _entity_dict(self.entity, self.score) if self.entity is not None and self.score is not None else None,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class SupportDecision:
    accepted: bool
    reason: str
    kind: Optional[str]
    path_nodes: Tuple[str, ...] = ()
    path_edges: Tuple[Tuple[str, str], ...] = ()
    direction: Optional[str] = None
    relations: Tuple[str, ...] = ()

    def audit_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "kind": self.kind,
            "direction": self.direction,
            "hops": len(self.path_edges) if self.path_edges else None,
            "path_nodes": list(self.path_nodes),
            "path_edges": [list(edge) for edge in self.path_edges],
            "relations": list(self.relations),
            "criteria": {
                "direct_primekg_only": True,
                "ccs_ancestor_descendant_max_hops": MAX_CCS_HOPS,
            },
        }


class SupportIndex:
    """Typed PrimeKG and directed CCS support evidence."""

    def __init__(
        self,
        primekg_edges: Iterable[Tuple[str, str, Optional[str]]] = (),
        ccs_edges: Iterable[Tuple[str, str]] = (),
    ):
        primekg: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for u, v, relation in primekg_edges:
            edge = _canonical_edge(u, v)
            if edge is None:
                continue
            primekg.setdefault(edge, set())
            if relation is not None and str(relation).strip():
                primekg[edge].add(str(relation).strip())

        parent_to_children: Dict[str, Set[str]] = defaultdict(set)
        child_to_parents: Dict[str, Set[str]] = defaultdict(set)
        ccs_pairs: Set[Tuple[str, str]] = set()
        for parent, child in ccs_edges:
            parent, child = canonicalize_ccs_id(parent), canonicalize_ccs_id(child)
            if not parent or not child:
                continue
            if parent == child:
                raise ValueError(f"CCS hierarchy contains a self-cycle at {parent!r}")
            pair = (parent, child)
            if pair in ccs_pairs:
                continue
            ccs_pairs.add(pair)
            parent_to_children[parent].add(child)
            child_to_parents[child].add(parent)

        _validate_directed_acyclic(parent_to_children, "CCS hierarchy")

        if not primekg and not ccs_pairs:
            raise ValueError("support index requires at least one usable PrimeKG or CCS edge")

        self._primekg = {edge: tuple(sorted(relations)) for edge, relations in primekg.items()}
        self._parent_to_children = {node: tuple(sorted(values)) for node, values in parent_to_children.items()}
        self._child_to_parents = {node: tuple(sorted(values)) for node, values in child_to_parents.items()}

    @classmethod
    def from_csvs(
        cls,
        primekg_path: Optional[str] = None,
        ccs_path: Optional[str] = None,
    ) -> "SupportIndex":
        primekg: List[Tuple[str, str, Optional[str]]] = []
        ccs: List[Tuple[str, str]] = []
        if primekg_path:
            for _, row in _read_csv(primekg_path, required=("u", "v")):
                primekg.append((row["u"], row["v"], row.get("relation")))
        if ccs_path:
            for _, row in _read_csv(ccs_path, required=("parent", "child")):
                ccs.append((row["parent"], row["child"]))
        return cls(primekg, ccs)

    def find_support(self, root: str, target: str) -> SupportDecision:
        root, target = _clean_id(root), _clean_id(target)
        if not root or not target or root == target:
            return SupportDecision(False, "self_reference_or_missing_endpoint", None)

        direct = _canonical_edge(root, target)
        if direct is not None and direct in self._primekg:
            return SupportDecision(
                True,
                "direct_primekg_edge",
                "primekg_direct",
                (root, target),
                (direct,),
                relations=self._primekg[direct],
            )

        ccs_root = canonicalize_ccs_id(root)
        ccs_target = canonicalize_ccs_id(target)
        if ccs_root == ccs_target:
            return SupportDecision(False, "self_reference_or_missing_endpoint", None)

        paths: List[Tuple[int, Tuple[str, ...], str]] = []
        down = _shortest_directed_path(self._parent_to_children, ccs_root, ccs_target, MAX_CCS_HOPS)
        if down is not None:
            paths.append((len(down) - 1, down, "root_is_ancestor"))
        up = _shortest_directed_path(self._child_to_parents, ccs_root, ccs_target, MAX_CCS_HOPS)
        if up is not None:
            paths.append((len(up) - 1, up, "root_is_descendant"))

        if paths:
            _, path, direction = min(paths, key=lambda item: (item[0], item[1], item[2]))
            edges = tuple(edge for edge in (_canonical_edge(u, v) for u, v in zip(path, path[1:])) if edge is not None)
            return SupportDecision(
                True,
                "ccs_ancestor_descendant_path",
                "ccs_hierarchy",
                path,
                edges,
                direction=direction,
            )

        return SupportDecision(False, "no_direct_primekg_or_ccs_ancestor_descendant_support", None)


def _validate_directed_acyclic(adjacency: Mapping[str, Iterable[str]], label: str) -> None:
    """Reject cycles while allowing a node to have multiple parents."""

    state: Dict[str, int] = {}

    def visit(node: str, path: List[str]) -> None:
        status = state.get(node, 0)
        if status == 2:
            return
        if status == 1:
            start = path.index(node)
            cycle = path[start:] + [node]
            raise ValueError(f"{label} contains a cycle: {' -> '.join(cycle)}")
        state[node] = 1
        path.append(node)
        for neighbor in sorted(adjacency.get(node, ())):
            visit(neighbor, path)
        path.pop()
        state[node] = 2

    nodes = set(adjacency)
    nodes.update(neighbor for values in adjacency.values() for neighbor in values)
    for node in sorted(nodes):
        if state.get(node, 0) == 0:
            visit(node, [])


def _shortest_directed_path(
    adjacency: Mapping[str, Sequence[str]],
    start: str,
    target: str,
    max_hops: int,
) -> Optional[Tuple[str, ...]]:
    if start == target:
        return (start,)
    frontier: List[Tuple[str, ...]] = [(start,)]
    for _ in range(max_hops):
        next_frontier: List[Tuple[str, ...]] = []
        matches: List[Tuple[str, ...]] = []
        for path in sorted(frontier):
            for neighbor in adjacency.get(path[-1], ()):
                if neighbor in path:
                    continue
                candidate = path + (neighbor,)
                if neighbor == target:
                    matches.append(candidate)
                else:
                    next_frontier.append(candidate)
        if matches:
            return min(matches)
        frontier = next_frontier
        if not frontier:
            break
    return None


@dataclass(frozen=True)
class ArtifactRecord:
    concept_id: str
    concept_name: str
    definition: str
    cascade: Tuple[str, ...]
    meta: Mapping[str, Any]


@dataclass
class FilterResult:
    grounded: List[Dict[str, Any]]
    audit: List[Dict[str, Any]]
    summary: Dict[str, Any]


def _validate_artifact(value: Any) -> Tuple[Optional[ArtifactRecord], List[Dict[str, str]]]:
    errors: List[Dict[str, str]] = []
    if not isinstance(value, dict):
        return None, [{"field": "$", "reason": "record_must_be_object"}]

    keys = set(value)
    for field in sorted(_ARTIFACT_FIELDS - keys):
        errors.append({"field": field, "reason": "required_field_missing"})
    for field in sorted(keys - _ARTIFACT_FIELDS):
        errors.append({"field": field, "reason": "additional_property_not_allowed"})

    concept_id = value.get("concept_id")
    concept_name = value.get("concept_name")
    definition = value.get("definition")
    cascade = value.get("cascade")
    meta = value.get("meta")

    if not isinstance(concept_id, str) or not concept_id.strip():
        errors.append({"field": "concept_id", "reason": "non_empty_string_required"})
    if not isinstance(concept_name, str):
        errors.append({"field": "concept_name", "reason": "string_required"})
    if not isinstance(definition, str) or not definition.strip():
        errors.append({"field": "definition", "reason": "non_empty_string_required"})
    if not isinstance(meta, dict):
        errors.append({"field": "meta", "reason": "object_required"})
    else:
        try:
            json.dumps(meta, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            errors.append({"field": "meta", "reason": "finite_json_object_required"})

    clean_cascade: List[str] = []
    if not isinstance(cascade, list):
        errors.append({"field": "cascade", "reason": "array_required"})
    else:
        if not 1 <= len(cascade) <= 5:
            errors.append({"field": "cascade", "reason": "item_count_must_be_between_1_and_5"})
        for index, mention in enumerate(cascade):
            if not isinstance(mention, str) or not mention.strip():
                errors.append({"field": f"cascade[{index}]", "reason": "non_empty_string_required"})
            else:
                clean_cascade.append(mention.strip())
        if len(set(clean_cascade)) != len(clean_cascade):
            errors.append({"field": "cascade", "reason": "items_must_be_unique"})

    if errors:
        return None, errors
    return ArtifactRecord(
        concept_id=concept_id.strip(),
        concept_name=concept_name.strip(),
        definition=definition.strip(),
        cascade=tuple(clean_cascade),
        meta=dict(meta),
    ), []


def _map_mention(
    mention: str,
    inventory: Inventory,
    candidates: EmbeddingCandidateIndex,
) -> MappingDecision:
    exact = inventory.exact_matches(mention)
    if exact:
        selected = exact[0]
        return MappingDecision(
            True,
            "exact_match",
            "exact",
            selected,
            1.0,
            {
                "normalized_mention": normalize_mention(mention),
                "exact_candidate_ids": [entity.entity_id for entity in exact],
                "tie_break": _SOURCE_PRIORITY_DESCRIPTION,
                "embedding_candidate_consulted": False,
            },
        )

    candidate = candidates.get(mention)
    if candidate is None:
        return MappingDecision(
            False,
            "no_exact_or_precomputed_embedding_candidate",
            None,
            None,
            None,
            {"normalized_mention": normalize_mention(mention), "threshold": TAU_MAP, "comparison": "strictly_greater_than"},
        )
    if not candidate.score > TAU_MAP:
        return MappingDecision(
            False,
            "embedding_score_not_strictly_above_threshold",
            "precomputed_top1",
            None,
            candidate.score,
            {
                "normalized_mention": normalize_mention(mention),
                "candidate_entity_id": candidate.entity_id,
                "candidate_score": candidate.score,
                "threshold": TAU_MAP,
                "comparison": "strictly_greater_than",
            },
        )
    entity = inventory.get(candidate.entity_id)
    if entity is None:
        return MappingDecision(
            False,
            "embedding_candidate_entity_not_in_inventory",
            "precomputed_top1",
            None,
            candidate.score,
            {
                "normalized_mention": normalize_mention(mention),
                "candidate_entity_id": candidate.entity_id,
                "candidate_score": candidate.score,
                "threshold": TAU_MAP,
                "comparison": "strictly_greater_than",
            },
        )
    return MappingDecision(
        True,
        "precomputed_top1_above_threshold",
        "precomputed_top1",
        entity,
        candidate.score,
        {
            "normalized_mention": normalize_mention(mention),
            "candidate_entity_id": candidate.entity_id,
            "candidate_score": candidate.score,
            "threshold": TAU_MAP,
            "comparison": "strictly_greater_than",
        },
    )


def _format_failure_audit(line_no: int, reason: str, errors: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "input_line": line_no,
        "concept_id": None,
        "concept_name": None,
        "mention_index": None,
        "mention": None,
        "normalized_mention": None,
        "status": "rejected",
        "first_failure": "format",
        "reason": reason,
        "format_evidence": {"errors": [dict(error) for error in errors]},
        "mapping": {"attempted": False},
        "support": {"attempted": False},
    }


def _mention_audit_base(record: ArtifactRecord, line_no: int, index: int, mention: str) -> Dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "input_line": line_no,
        "concept_id": record.concept_id,
        "concept_name": record.concept_name,
        "mention_index": index,
        "mention": mention,
        "normalized_mention": normalize_mention(mention),
    }


def _connected_component(root: str, edges: Iterable[Tuple[str, str]]) -> Set[str]:
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    for u, v in edges:
        if u == v:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)
    seen = {root}
    frontier = [root]
    while frontier:
        node = frontier.pop()
        for neighbor in sorted(adjacency.get(node, ())):
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return seen


def _filter_valid_artifact(
    record: ArtifactRecord,
    line_no: int,
    inventory: Inventory,
    candidates: EmbeddingCandidateIndex,
    support: SupportIndex,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    audit: List[Dict[str, Any]] = []
    root_entity = inventory.get(record.concept_id)
    if root_entity is None:
        for index, mention in enumerate(record.cascade):
            item = _mention_audit_base(record, line_no, index, mention)
            item.update(
                {
                    "status": "rejected",
                    "first_failure": "mapping",
                    "reason": "root_concept_not_in_inventory",
                    "mapping": {
                        "attempted": True,
                        "accepted": False,
                        "reason": "root_concept_not_in_inventory",
                        "method": None,
                        "selected": None,
                        "evidence": {"root_concept_id": record.concept_id},
                    },
                    "support": {"attempted": False},
                }
            )
            audit.append(item)
        return None, audit

    root_id = root_entity.entity_id
    accepted_by_entity: Dict[str, Tuple[Entity, float, str, str]] = {}
    graph_nodes: Set[str] = {root_id}
    graph_edges: Set[Tuple[str, str]] = set()

    for index, mention in enumerate(record.cascade):
        item = _mention_audit_base(record, line_no, index, mention)
        mapping = _map_mention(mention, inventory, candidates)
        item["mapping"] = {"attempted": True, **mapping.audit_dict()}
        if not mapping.accepted or mapping.entity is None or mapping.score is None:
            item.update(
                {
                    "status": "rejected",
                    "first_failure": "mapping",
                    "reason": mapping.reason,
                    "support": {"attempted": False},
                }
            )
            audit.append(item)
            continue

        support_decision = support.find_support(root_id, mapping.entity.entity_id)
        item["support"] = {"attempted": True, **support_decision.audit_dict()}
        if not support_decision.accepted:
            item.update(
                {
                    "status": "rejected",
                    "first_failure": "support",
                    "reason": support_decision.reason,
                }
            )
            audit.append(item)
            continue

        item.update({"status": "accepted", "first_failure": None, "reason": support_decision.reason})
        audit.append(item)
        graph_nodes.update(support_decision.path_nodes)
        graph_edges.update(support_decision.path_edges)
        previous = accepted_by_entity.get(mapping.entity.entity_id)
        candidate_value = (mapping.entity, mapping.score, mention, mapping.method or "unknown")
        if previous is None or (-mapping.score, mention, mapping.method or "") < (-previous[1], previous[2], previous[3]):
            accepted_by_entity[mapping.entity.entity_id] = candidate_value

    if not accepted_by_entity:
        return None, audit

    reachable = _connected_component(root_id, graph_edges)
    nodes = [root_id] + sorted((graph_nodes & reachable) - {root_id})
    edges = sorted(
        edge
        for edge in graph_edges
        if edge[0] in reachable and edge[1] in reachable and edge[0] != edge[1]
    )
    cascade_entities = []
    # Python dictionaries retain first-mention order, preserving the clinical
    # cascade while the entity-id key still deduplicates repeated mentions.
    for entity_id in accepted_by_entity:
        if entity_id not in reachable:
            continue
        entity, score, mention, method = accepted_by_entity[entity_id]
        entity_value = _entity_dict(entity, score)
        entity_value.update({"mention": mention, "mapping_method": method})
        cascade_entities.append(entity_value)

    if not cascade_entities or not edges:
        # Defensive fail-closed guard: no template may leave this stage unless
        # at least one grounded cascade entity is connected to the root.
        return None, audit

    meta = dict(record.meta)
    meta["pool_v1_filtering"] = {
        "schema_version": FILTER_SCHEMA_VERSION,
        "tau_map": TAU_MAP,
        "threshold_comparison": "greater_than",
        "ccs_hierarchy_max_hops": MAX_CCS_HOPS,
        "accepted_mentions": sum(1 for item in audit if item["status"] == "accepted"),
        "rejected_mentions": sum(1 for item in audit if item["status"] == "rejected"),
    }
    grounded = {
        "root_concept_id": root_id,
        "root_name": record.concept_name,
        "definition": record.definition,
        "cascade_entities": cascade_entities,
        "subgraph_nodes": nodes,
        "subgraph_edges": [list(edge) for edge in edges],
        "meta": meta,
    }
    return grounded, audit


def _build_summary(
    input_records: int,
    format_passed: int,
    grounded: Sequence[Mapping[str, Any]],
    audit: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    format_failures = sum(1 for item in audit if item.get("first_failure") == "format")
    mention_audit = [item for item in audit if item.get("mention_index") is not None]
    failed_mapping = sum(1 for item in mention_audit if item.get("first_failure") == "mapping")
    low_support = sum(1 for item in mention_audit if item.get("first_failure") == "support")
    supported = sum(1 for item in mention_audit if item.get("status") == "accepted")
    mapping_success = sum(
        1 for item in mention_audit if isinstance(item.get("mapping"), dict) and item["mapping"].get("accepted") is True
    )
    exact_mapped = sum(1 for item in mention_audit if item.get("mapping", {}).get("method") == "exact")
    embedding_mapped = sum(
        1
        for item in mention_audit
        if item.get("mapping", {}).get("method") == "precomputed_top1"
        and item.get("mapping", {}).get("accepted") is True
    )
    grounded_lines = {item.get("input_line") for item in audit if item.get("status") == "accepted"}

    def rate(numerator: int, denominator: int) -> Optional[float]:
        return float(numerator) / float(denominator) if denominator else None

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "configuration": {
            "tau_map": TAU_MAP,
            "threshold_comparison": "greater_than",
            "exact_match_first": True,
            "embedding_input": "supplied_precomputed_top1_only",
            "ccs_hierarchy_max_hops": MAX_CCS_HOPS,
            "fail_closed": True,
        },
        "counts": {
            "input_records": input_records,
            "format_passed_records": format_passed,
            "format_violation_records": format_failures,
            "valid_mentions": len(mention_audit),
            "ontology_mapping_successes": mapping_success,
            "exact_mapping_successes": exact_mapped,
            "embedding_mapping_successes": embedding_mapped,
            "externally_supported_mentions": supported,
            "grounded_templates": len(grounded),
            "rejected_templates": input_records - len(grounded),
            "audit_records": len(audit),
        },
        "first_failure_counts": {
            "format_violation": format_failures,
            "failed_mapping_or_missing_concept": failed_mapping,
            "low_external_support": low_support,
        },
        "rates": {
            "format_pass_rate": rate(format_passed, input_records),
            "ontology_mapping_success_rate": rate(mapping_success, len(mention_audit)),
            "externally_supported_link_rate": rate(supported, mapping_success),
        },
        "rate_denominators": {
            "format_pass_rate": "input_records",
            "ontology_mapping_success_rate": "valid_mentions",
            "externally_supported_link_rate": "ontology_mapping_successes",
        },
        "grounded_input_lines": sorted(line for line in grounded_lines if isinstance(line, int)),
    }


def _filter_entries(
    entries: Iterable[Tuple[int, Any, Optional[str]]],
    inventory: Inventory,
    candidates: EmbeddingCandidateIndex,
    support: SupportIndex,
) -> FilterResult:
    grounded: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    input_records = 0
    format_passed = 0
    for line_no, value, parse_error in entries:
        input_records += 1
        if parse_error is not None:
            audit.append(
                _format_failure_audit(
                    line_no,
                    "invalid_json",
                    ({"field": "$", "reason": parse_error},),
                )
            )
            continue
        record, schema_errors = _validate_artifact(value)
        if record is None:
            audit.append(_format_failure_audit(line_no, "schema_violation", schema_errors))
            continue
        format_passed += 1
        output, record_audit = _filter_valid_artifact(record, line_no, inventory, candidates, support)
        audit.extend(record_audit)
        if output is not None:
            grounded.append(output)

    summary = _build_summary(input_records, format_passed, grounded, audit)
    return FilterResult(grounded=grounded, audit=audit, summary=summary)


def filter_artifact_objects(
    artifacts: Iterable[Any],
    inventory: Inventory,
    support: SupportIndex,
    candidates: Optional[EmbeddingCandidateIndex] = None,
) -> FilterResult:
    """Filter already parsed artifact objects, assigning 1-based input lines."""

    entries = ((line_no, value, None) for line_no, value in enumerate(artifacts, start=1))
    return _filter_entries(entries, inventory, candidates or EmbeddingCandidateIndex(), support)


def filter_artifacts_jsonl(
    path: str,
    inventory: Inventory,
    support: SupportIndex,
    candidates: Optional[EmbeddingCandidateIndex] = None,
) -> FilterResult:
    return _filter_entries(_read_jsonl_entries(path), inventory, candidates or EmbeddingCandidateIndex(), support)


def _read_csv(path: str, required: Sequence[str]) -> List[Tuple[int, Dict[str, str]]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = [field for field in required if field not in fields]
        if missing:
            raise ValueError(f"{path}: missing required CSV columns {missing}")
        rows: List[Tuple[int, Dict[str, str]]] = []
        for row_no, row in enumerate(reader, start=2):
            cleaned = {str(key): (value.strip() if isinstance(value, str) else value) for key, value in row.items()}
            for field in required:
                if not cleaned.get(field):
                    raise ValueError(f"{path}:{row_no}: required field {field!r} is empty")
            rows.append((row_no, cleaned))
        return rows


def _read_jsonl_entries(path: str) -> Iterable[Tuple[int, Any, Optional[str]]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                yield line_no, strict_json_loads(raw), None
            except json.JSONDecodeError as exc:
                yield line_no, None, f"json_decode_error_at_column_{exc.colno}"
            except ValueError as exc:
                yield line_no, None, str(exc)


def _write_jsonl(path: str, values: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            handle.write("\n")


def _write_summary(path: str, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministically ground and externally support-filter pool_v1 distilled artifacts."
    )
    parser.add_argument("--artifacts-jsonl", required=True, help="Distilled artifact JSONL.")
    parser.add_argument("--inventory-csv", required=True, help="Canonical inventory: entity_id,name[,source].")
    parser.add_argument("--embedding-candidates", default=None, help="Optional CSV/JSONL: mention,entity_id,score.")
    parser.add_argument("--primekg-edges-csv", default=None, help="Direct PrimeKG edges: u,v[,relation].")
    parser.add_argument("--ccs-edges-csv", default=None, help="Directed CCS hierarchy: parent,child.")
    parser.add_argument("--out-grounded-jsonl", required=True)
    parser.add_argument("--out-audit-jsonl", required=True)
    parser.add_argument("--out-summary-json", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.primekg_edges_csv and not args.ccs_edges_csv:
        parser.error("at least one typed support source is required: --primekg-edges-csv or --ccs-edges-csv")

    inventory = Inventory.from_csv(args.inventory_csv)
    candidates = EmbeddingCandidateIndex.from_path(args.embedding_candidates)
    support = SupportIndex.from_csvs(args.primekg_edges_csv, args.ccs_edges_csv)
    result = filter_artifacts_jsonl(args.artifacts_jsonl, inventory, support, candidates)
    _write_jsonl(args.out_grounded_jsonl, result.grounded)
    _write_jsonl(args.out_audit_jsonl, result.audit)
    _write_summary(args.out_summary_json, result.summary)
    print(
        json.dumps(
            {
                "grounded_templates": len(result.grounded),
                "audit_records": len(result.audit),
                "summary": args.out_summary_json,
            },
            sort_keys=True,
        )
    )
    return 0 if result.grounded else 2


if __name__ == "__main__":
    raise SystemExit(main())
