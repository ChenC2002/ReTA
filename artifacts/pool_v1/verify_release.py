"""Integrity and semantic validation for the frozen ``pool_v1`` release."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


EXPERIMENTAL_POOLS = {
    "mimic_iii_primekg": 920,
    "mimic_iv_primekg": 1180,
    "mimic_iii_umls": 874,
    "mimic_iv_umls": 1092,
}


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    message: str


@dataclass
class ValidationReport:
    release_dir: str
    issues: List[Issue] = field(default_factory=list)
    checked_files: int = 0
    checked_template_records: int = 0

    def add(self, severity: str, code: str, message: str) -> None:
        self.issues.append(Issue(severity, code, message))

    def ok(self, allow_incomplete: bool = False) -> bool:
        for issue in self.issues:
            if issue.severity == "error":
                return False
            if issue.severity == "incomplete" and not allow_incomplete:
                return False
        return True

    def to_dict(self, allow_incomplete: bool = False) -> Dict[str, Any]:
        return {
            "release_dir": self.release_dir,
            "ok": self.ok(allow_incomplete),
            "allow_incomplete": bool(allow_incomplete),
            "checked_files": self.checked_files,
            "checked_template_records": self.checked_template_records,
            "issues": [asdict(issue) for issue in self.issues],
        }

    def format_text(self, allow_incomplete: bool = False) -> str:
        state = "valid" if self.ok(allow_incomplete) else "invalid"
        lines = [
            f"pool release {state}: {self.release_dir}",
            f"checked files={self.checked_files}, templates={self.checked_template_records}",
        ]
        lines.extend(f"[{item.severity}] {item.code}: {item.message}" for item in self.issues)
        return "\n".join(lines)


def _load_json(path: Path, report: ValidationReport, code: str) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.add("error", code, f"{path}: {exc}")
        return None


def _load_config(path: Path, report: ValidationReport) -> Optional[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml

            value = yaml.safe_load(text)
        except Exception:
            try:
                value = _parse_simple_yaml(text)
            except Exception as exc:
                report.add("error", "config_parse_error", f"{path}: {exc}")
                return None
    if not isinstance(value, dict):
        report.add("error", "config_type", "config.yaml must contain a mapping")
        return None
    return value


def _yaml_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith('"') and value.endswith('"'):
        return json.loads(value)
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?", value):
        return float(value)
    return value


def _parse_simple_yaml(text: str) -> Any:
    """Parse the release's mapping/list/scalar YAML subset without PyYAML."""
    tokens: List[Tuple[int, str]] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise ValueError(f"line {line_no}: tabs are unsupported")
        indent = len(raw) - len(raw.lstrip(" "))
        content = raw.strip()
        if " #" in content:
            content = content.split(" #", 1)[0].rstrip()
        tokens.append((indent, content))
    if not tokens:
        return {}

    def parse_block(position: int, indent: int) -> Tuple[Any, int]:
        is_list = tokens[position][1].startswith("- ")
        container: Any = [] if is_list else {}
        while position < len(tokens):
            current_indent, content = tokens[position]
            if current_indent < indent:
                break
            if current_indent != indent:
                raise ValueError(f"unexpected indentation near {content!r}")
            if is_list:
                if not content.startswith("- "):
                    raise ValueError("cannot mix YAML list and mapping entries")
                container.append(_yaml_scalar(content[2:]))
                position += 1
                continue
            if content.startswith("- ") or ":" not in content:
                raise ValueError(f"expected mapping entry near {content!r}")
            key, raw_value = content.split(":", 1)
            key = key.strip()
            raw_value = raw_value.strip()
            position += 1
            if raw_value:
                container[key] = _yaml_scalar(raw_value)
            elif position < len(tokens) and tokens[position][0] > indent:
                child, position = parse_block(position, tokens[position][0])
                container[key] = child
            else:
                container[key] = None
        return container, position

    parsed, end = parse_block(0, tokens[0][0])
    if end != len(tokens):
        raise ValueError("trailing YAML content")
    return parsed


def _safe_file(root: Path, relative: str, report: ValidationReport) -> Optional[Path]:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        report.add("error", "unsafe_path", f"manifest path is not release-relative: {relative!r}")
        return None
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        report.add("error", "unsafe_path", f"manifest path escapes release: {relative!r}")
        return None
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path, report: ValidationReport) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except Exception as exc:
                    report.add("error", "jsonl_parse_error", f"{path}:{line_no}: {exc}")
                    continue
                if not isinstance(item, dict):
                    report.add("error", "jsonl_record_type", f"{path}:{line_no}: record is not an object")
                    continue
                records.append(item)
    except OSError as exc:
        report.add("error", "file_read_error", f"{path}: {exc}")
    return records


def _validate_config(config: Dict[str, Any], manifest: Dict[str, Any], report: ValidationReport) -> None:
    release = config.get("release", {})
    if release.get("id") != manifest.get("release_id"):
        report.add("error", "config_release_id", "config and manifest release IDs differ")
    if release.get("status") != manifest.get("status"):
        report.add("error", "config_release_status", "config and manifest statuses differ")

    paper = config.get("distillation", {}).get("paper_reported", {})
    expected = {"temperature": 0.2, "top_p": 0.9, "max_tokens": 256}
    for key, value in expected.items():
        if paper.get(key) != value:
            report.add("error", "paper_config_mismatch", f"distillation.paper_reported.{key} must be {value}")
    if paper.get("model_snapshot") is not None or paper.get("access_date") is not None:
        report.add("error", "unsupported_paper_provenance", "paper-reported snapshot/access date must remain null")

    post_hoc = config.get("distillation", {}).get("thread_declared_post_hoc", {})
    if post_hoc.get("verification") != "unverified" or post_hoc.get("reported_by_paper") is not False:
        report.add("error", "post_hoc_provenance", "post-hoc GPT metadata must remain explicitly unverified")

    grounding = config.get("grounding", {}).get("paper_reported", {})
    if grounding.get("top_1_similarity_threshold") != 0.90:
        report.add("error", "mapping_threshold", "tau_map must be 0.90")
    if grounding.get("threshold_comparison") != "greater_than":
        report.add("error", "mapping_comparator", "embedding similarity must strictly exceed tau_map")
    if config.get("filtering", {}).get("paper_reported", {}).get("ccs_hierarchy_max_levels") != 2:
        report.add("error", "ccs_support_depth", "CCS support depth must be two levels")
    if config.get("clustering", {}).get("paper_reported", {}).get("projection_dim") != 256:
        report.add("error", "projection_dim", "experimental template vectors must use d=256")

    resolved = config.get("clustering", {}).get("release_resolved", {})
    if resolved and resolved.get("distance_cut_tau") != 0.16:
        report.add("error", "release_tau", "pool_v1 release-resolved clustering tau must be 0.16")


def _validate_prompts(root: Path, file_paths: set[str], report: ValidationReport) -> None:
    expected = ("prompts/figure8_system.txt", "prompts/figure8_user.txt")
    for path in expected:
        if path not in file_paths:
            report.add("error", "prompt_not_manifested", f"{path} is not covered by the manifest")
    try:
        combined = "\n".join((root / path).read_text(encoding="utf-8") for path in expected)
    except OSError as exc:
        report.add("error", "prompt_missing", str(exc))
        return
    tokens = [
        "expert clinical pathologist",
        "<concept description>",
        "sparse | moderate | dense",
        "Definition",
        "Clinical Cascade",
    ]
    for token in tokens:
        if token not in combined:
            report.add("error", "prompt_content", f"frozen prompt is missing {token!r}")
    if "{1–5}" not in combined and "{1-5}" not in combined:
        report.add("error", "prompt_content", "frozen prompt is missing the 1-5 cascade bound")


def _edge_key(u: str, v: str) -> Tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def _validate_template(record: Dict[str, Any], where: str, expected_dim: Optional[int], report: ValidationReport) -> None:
    template_id = record.get("template_id")
    vector = record.get("vector")
    if not isinstance(template_id, int) or template_id < 0:
        report.add("error", "template_id", f"{where}: template_id must be a non-negative integer")
    if not isinstance(vector, list) or not vector:
        report.add("error", "template_vector", f"{where}: vector must be a non-empty list")
    else:
        numeric = all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in vector)
        if not numeric:
            report.add("error", "template_vector", f"{where}: vector contains a non-finite/non-numeric value")
        else:
            norm = math.sqrt(sum(float(value) ** 2 for value in vector))
            if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
                report.add("error", "template_vector_norm", f"{where}: vector L2 norm is {norm:.8g}, expected 1")
        if expected_dim is not None and len(vector) != expected_dim:
            report.add("error", "template_vector_dim", f"{where}: vector dim {len(vector)} != {expected_dim}")

    medoid_idx = record.get("medoid_idx")
    members = record.get("member_indices")
    if not isinstance(members, list) or not members or any(not isinstance(i, int) or i < 0 for i in members):
        report.add("error", "member_indices", f"{where}: member_indices must be non-empty non-negative integers")
        members = []
    elif len(members) != len(set(members)):
        report.add("error", "member_indices_duplicate", f"{where}: duplicate member_indices")
    if not isinstance(medoid_idx, int) or medoid_idx < 0 or medoid_idx not in members:
        report.add("error", "medoid_not_member", f"{where}: medoid_idx must belong to member_indices")

    medoid = record.get("medoid")
    if not isinstance(medoid, dict):
        report.add("error", "medoid", f"{where}: medoid must be an object")
        return
    root = medoid.get("root_concept_id")
    if not isinstance(root, str) or not root:
        report.add("error", "medoid_root", f"{where}: missing medoid root_concept_id")
    if not isinstance(medoid.get("definition"), str) or not medoid.get("definition", "").strip():
        report.add("error", "medoid_definition", f"{where}: missing medoid definition")

    entities = medoid.get("cascade_entities")
    if not isinstance(entities, list) or not 1 <= len(entities) <= 5:
        report.add("error", "cascade_bound", f"{where}: cascade_entities must contain 1-5 records")
        entities = []
    entity_ids = [entity.get("entity_id") for entity in entities if isinstance(entity, dict)]
    if len(entity_ids) != len(set(entity_ids)):
        report.add("error", "cascade_duplicate", f"{where}: duplicate cascade entity IDs")

    nodes = medoid.get("subgraph_nodes")
    edges = medoid.get("subgraph_edges")
    if not isinstance(nodes, list) or not nodes or any(not isinstance(node, str) or not node for node in nodes):
        report.add("error", "subgraph_nodes", f"{where}: subgraph_nodes must be non-empty strings")
        return
    if len(nodes) != len(set(nodes)):
        report.add("error", "subgraph_node_duplicate", f"{where}: duplicate subgraph nodes")
    node_set = set(nodes)
    if root not in node_set:
        report.add("error", "subgraph_root", f"{where}: root is absent from subgraph nodes")
    for entity_id in entity_ids:
        if entity_id not in node_set:
            report.add("error", "cascade_node_missing", f"{where}: cascade entity {entity_id!r} absent from graph")

    if not isinstance(edges, list):
        report.add("error", "subgraph_edges", f"{where}: subgraph_edges must be a list")
        return
    seen: set[Tuple[str, str]] = set()
    adjacency: Dict[str, List[str]] = {node: [] for node in nodes}
    for edge_idx, edge in enumerate(edges):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2 or not all(isinstance(x, str) for x in edge):
            report.add("error", "subgraph_edge", f"{where}: invalid edge at index {edge_idx}")
            continue
        u, v = edge
        if u == v:
            report.add("error", "subgraph_self_loop", f"{where}: self-loop {u!r}")
        if u not in node_set or v not in node_set:
            report.add("error", "subgraph_unknown_endpoint", f"{where}: edge {u!r}-{v!r} references unknown node")
            continue
        key = _edge_key(u, v)
        if key in seen:
            report.add("error", "subgraph_duplicate_edge", f"{where}: duplicate undirected edge {key}")
        seen.add(key)
        if u != v:
            adjacency[u].append(v)
            adjacency[v].append(u)
    if root in node_set:
        reachable = {root}
        frontier = [root]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency.get(current, []):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    frontier.append(neighbor)
        disconnected = sorted(node_set - reachable)
        if disconnected:
            report.add("error", "subgraph_disconnected", f"{where}: root-disconnected nodes {disconnected}")


def _validate_pool(path: Path, pool: Dict[str, Any], report: ValidationReport) -> int:
    records = _read_jsonl(path, report)
    pool_id = str(pool.get("id", "unknown"))
    expected_dim = pool.get("expected_dim")
    if expected_dim is None:
        is_demo = (
            pool_id == "demo"
            or str(pool.get("path", "")).startswith("demo/")
            or pool.get("paper_reported") is False
        )
        expected_dim = None if is_demo else 256
    ids: List[int] = []
    seen_members: set[int] = set()
    dims: set[int] = set()
    for index, record in enumerate(records):
        where = f"{path}:{index + 1}"
        _validate_template(record, where, int(expected_dim) if expected_dim is not None else None, report)
        if isinstance(record.get("template_id"), int):
            ids.append(record["template_id"])
        if isinstance(record.get("vector"), list):
            dims.add(len(record["vector"]))
        for member in record.get("member_indices", []) if isinstance(record.get("member_indices"), list) else []:
            if member in seen_members:
                report.add("error", "cluster_member_overlap", f"{where}: member index {member} appears in multiple clusters")
            seen_members.add(member)
    if len(dims) > 1:
        report.add("error", "pool_vector_dims", f"{pool_id}: pool contains mixed vector dimensions {sorted(dims)}")
    if sorted(ids) != list(range(len(records))):
        report.add("error", "template_ids_not_contiguous", f"{pool_id}: template IDs must be contiguous from zero")
    expected_records = pool.get("expected_templates")
    if expected_records is not None and len(records) != int(expected_records):
        report.add("error", "pool_record_count", f"{pool_id}: {len(records)} templates != expected {expected_records}")
    report.checked_template_records += len(records)
    return len(records)


def verify_release(release_dir: str | Path) -> ValidationReport:
    root = Path(release_dir).resolve()
    report = ValidationReport(str(root))
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        report.add("error", "manifest_missing", f"missing {manifest_path}")
        return report
    manifest = _load_json(manifest_path, report, "manifest_parse_error")
    if not isinstance(manifest, dict):
        return report

    if not isinstance(manifest.get("schema_version"), str):
        report.add("error", "manifest_schema", "manifest.schema_version must be a string")
    if manifest.get("release_id") != "pool_v1":
        report.add("error", "manifest_release_id", "manifest.release_id must be pool_v1")
    status = manifest.get("status")
    if status not in {"complete", "incomplete"}:
        report.add("error", "manifest_status", "manifest.status must be complete or incomplete")
    elif status == "incomplete":
        report.add("incomplete", "release_incomplete", "experimental pool payloads are not all included")

    files = manifest.get("files")
    if not isinstance(files, list):
        report.add("error", "manifest_files", "manifest.files must be a list")
        files = []
    manifested_paths: set[str] = set()
    records_by_path: Dict[str, int] = {}
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            report.add("error", "manifest_file_entry", f"files[{index}] must be an object")
            continue
        relative = entry.get("path")
        if not isinstance(relative, str):
            report.add("error", "manifest_file_path", f"files[{index}].path must be a string")
            continue
        if relative in manifested_paths:
            report.add("error", "manifest_duplicate_path", f"duplicate manifest path {relative}")
            continue
        manifested_paths.add(relative)
        path = _safe_file(root, relative, report)
        if path is None:
            continue
        if not path.is_file():
            severity = "error" if entry.get("required", True) else "warning"
            report.add(severity, "manifested_file_missing", f"missing {relative}")
            continue
        report.checked_files += 1
        actual_bytes = path.stat().st_size
        if entry.get("bytes") != actual_bytes:
            report.add("error", "size_mismatch", f"{relative}: {actual_bytes} bytes != manifest {entry.get('bytes')}")
        digest = _sha256(path)
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
            report.add("error", "invalid_manifest_sha256", f"{relative}: invalid manifest SHA-256")
        elif entry.get("sha256") != digest:
            report.add("error", "checksum_mismatch", f"{relative}: SHA-256 does not match manifest")
        if relative.endswith(".json"):
            _load_json(path, report, "json_parse_error")
        if relative.endswith(".jsonl"):
            records = _read_jsonl(path, report)
            records_by_path[relative] = len(records)
            if entry.get("records") != len(records):
                report.add("error", "record_count_mismatch", f"{relative}: {len(records)} records != manifest {entry.get('records')}")

    config_path = root / "config.yaml"
    if not config_path.is_file():
        report.add("error", "config_missing", "config.yaml is required")
    else:
        config = _load_config(config_path, report)
        if config is not None:
            _validate_config(config, manifest, report)
    _validate_prompts(root, manifested_paths, report)

    pools_value = manifest.get("pools")
    if isinstance(pools_value, dict):
        pools = [dict(value, id=key) for key, value in pools_value.items() if isinstance(value, dict)]
    elif isinstance(pools_value, list):
        pools = pools_value
    else:
        report.add("error", "manifest_pools", "manifest.pools must be a list")
        pools = []
    pools_by_id = {pool.get("id"): pool for pool in pools if isinstance(pool, dict)}
    missing_required = manifest.get("missing_required")
    if not isinstance(missing_required, list):
        report.add("error", "manifest_missing_required", "manifest.missing_required must be a list")
        missing_required = []
    missing_ids = {
        item if isinstance(item, str) else item.get("id")
        for item in missing_required
        if isinstance(item, (str, dict))
    }

    for pool_id, paper_count in EXPERIMENTAL_POOLS.items():
        pool = pools_by_id.get(pool_id)
        if not isinstance(pool, dict):
            report.add("error", "required_pool_undeclared", f"missing pool declaration {pool_id}")
            continue
        declared_count = pool.get("expected_templates", pool.get("paper_reported_templates"))
        if declared_count != paper_count:
            report.add("error", "paper_pool_count", f"{pool_id}: expected paper count {paper_count}")
        if not pool.get("included", False):
            if pool_id not in missing_ids and pool.get("path") not in missing_ids:
                report.add("error", "missing_pool_not_attested", f"{pool_id} is absent but not in missing_required")
            severity = "incomplete" if status == "incomplete" else "error"
            report.add(severity, "required_pool_missing", f"{pool_id} ({paper_count} templates) is not included")

    for pool in pools:
        if not isinstance(pool, dict) or not pool.get("included", False):
            continue
        relative = pool.get("path")
        if not isinstance(relative, str):
            report.add("error", "pool_path", f"{pool.get('id')}: included pool lacks a path")
            continue
        if relative not in manifested_paths:
            report.add("error", "pool_not_manifested", f"{pool.get('id')}: {relative} is not in files[]")
        path = _safe_file(root, relative, report)
        if path is not None and path.is_file():
            _validate_pool(path, pool, report)

    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the pool_v1 release, hashes, and template invariants.")
    parser.add_argument("--release-dir", "--release_dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--allow-incomplete", "--allow_incomplete", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    report = verify_release(args.release_dir)
    if args.as_json:
        print(json.dumps(report.to_dict(args.allow_incomplete), indent=2, sort_keys=True))
    else:
        print(report.format_text(args.allow_incomplete))
    return 0 if report.ok(args.allow_incomplete) else 2


if __name__ == "__main__":
    raise SystemExit(main())
