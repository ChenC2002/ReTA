"""Integrity and semantic validation for versioned ReTA knowledge releases."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


REQUIRED_RELEASE_FILES = frozenset(
    {
        "README.md",
        "config.yaml",
        "system_prompt.txt",
        "user_prompt.txt",
        "schemas/distilled_artifact.schema.json",
        "schemas/llm_response.schema.json",
    }
)
EMBEDDING_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
EMBEDDING_REVISION = "d5892b39a4adaed74b92212a44081509db72f87b"
EMBEDDING_BACKEND = "transformers_mean_pooling"


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

    def add(self, severity: str, code: str, message: str) -> None:
        self.issues.append(Issue(severity, code, message))

    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "release_dir": self.release_dir,
            "ok": self.ok(),
            "checked_files": self.checked_files,
            "issues": [asdict(issue) for issue in self.issues],
        }

    def format_text(self) -> str:
        state = "valid" if self.ok() else "invalid"
        lines = [
            f"reference release {state}: {self.release_dir}",
            f"checked files={self.checked_files}",
        ]
        lines.extend(f"[{item.severity}] {item.code}: {item.message}" for item in self.issues)
        return "\n".join(lines)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not permitted")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        _reject_json_constant(value)
    return parsed


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r} is not permitted")
        value[key] = item
    return value


def _strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        parse_constant=_reject_json_constant,
        parse_float=_parse_finite_json_float,
        object_pairs_hook=_reject_duplicate_json_keys,
    )


def _load_json(path: Path, report: ValidationReport, code: str) -> Optional[Any]:

    try:
        return _strict_json_loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.add("error", code, f"{path}: {exc}")
        return None


def _load_config(path: Path, report: ValidationReport) -> Optional[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        report.add("error", "config_read_error", f"{path}: {exc}")
        return None
    try:
        value = _strict_json_loads(text)
    except json.JSONDecodeError:
        try:
            import yaml

            class UniqueKeySafeLoader(yaml.SafeLoader):
                def construct_mapping(self, node: Any, deep: bool = False) -> Dict[Any, Any]:
                    self.flatten_mapping(node)
                    mapping: Dict[Any, Any] = {}
                    for key_node, value_node in node.value:
                        key = self.construct_object(key_node, deep=deep)
                        if key in mapping:
                            raise ValueError(f"duplicate YAML mapping key {key!r} is not permitted")
                        mapping[key] = self.construct_object(value_node, deep=deep)
                    return mapping

            value = yaml.load(text, Loader=UniqueKeySafeLoader)
        except ModuleNotFoundError:
            try:
                value = _parse_simple_yaml(text)
            except Exception as exc:
                report.add("error", "config_parse_error", f"{path}: {exc}")
                return None
        except Exception as exc:
            report.add("error", "config_parse_error", f"{path}: {exc}")
            return None
    except ValueError as exc:
        report.add("error", "config_parse_error", f"{path}: {exc}")
        return None
    if not isinstance(value, dict):
        report.add("error", "config_type", "config.yaml must contain a mapping")
        return None
    if _contains_nonfinite_number(value):
        report.add("error", "config_nonfinite", "config.yaml contains a non-finite number")
        return None
    return value


def _contains_nonfinite_number(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_contains_nonfinite_number(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite_number(item) for item in value)
    return False


def _yaml_scalar(value: str) -> Any:
    value = value.strip()
    lowered = value.lower()
    if lowered in {".nan", ".inf", "+.inf", "-.inf"}:
        return float(lowered.replace(".", "", 1))
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
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
            if key in container:
                raise ValueError(f"duplicate YAML mapping key {key!r} is not permitted")
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
    if (
        not relative
        or rel.is_absolute()
        or ".." in rel.parts
        or relative != rel.as_posix()
        or "\\" in relative
    ):
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


def _config_section(
    config: Dict[str, Any],
    name: str,
    report: ValidationReport,
) -> Dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        report.add("error", "config_section_type", f"config.{name} must be a mapping")
        return {}
    return value


def _validate_config(config: Dict[str, Any], manifest: Dict[str, Any], report: ValidationReport) -> None:
    release = _config_section(config, "release", report)
    if release.get("id") != manifest.get("release_id"):
        report.add("error", "config_release_id", "config and manifest release IDs differ")
    if release.get("version") != manifest.get("version"):
        report.add("error", "config_release_version", "config and manifest versions differ")
    if release.get("status") != manifest.get("status"):
        report.add("error", "config_release_status", "config and manifest statuses differ")

    distillation = _config_section(config, "distillation", report)
    expected = {"temperature": 0.2, "top_p": 0.9, "max_tokens": 256}
    for key, value in expected.items():
        if distillation.get(key) != value:
            report.add("error", "distillation_config", f"distillation.{key} must be {value}")
    expected_paths = {
        "system_prompt": "system_prompt.txt",
        "user_prompt": "user_prompt.txt",
        "response_schema": "schemas/llm_response.schema.json",
        "artifact_schema": "schemas/distilled_artifact.schema.json",
    }
    for key, expected_path in expected_paths.items():
        if distillation.get(key) != expected_path:
            report.add(
                "error",
                "distillation_asset",
                f"distillation.{key} must be {expected_path!r}",
            )
    if distillation.get("cascade_min_items") != 1:
        report.add("error", "cascade_bounds", "distillation.cascade_min_items must be 1")
    if distillation.get("cascade_max_items") != 5:
        report.add("error", "cascade_bounds", "distillation.cascade_max_items must be 5")

    grounding = _config_section(config, "grounding", report)
    expected_candidate_embedding = {
        "candidate_embedding_model": EMBEDDING_MODEL,
        "candidate_embedding_revision": EMBEDDING_REVISION,
        "candidate_embedding_backend": EMBEDDING_BACKEND,
    }
    if grounding.get("fallback_mode") != "precomputed_top1_candidate":
        report.add(
            "error",
            "grounding_fallback_mode",
            "grounding.fallback_mode must be 'precomputed_top1_candidate'",
        )
    for key, value in expected_candidate_embedding.items():
        if grounding.get(key) != value:
            report.add("error", "grounding_embedding", f"grounding.{key} must be {value!r}")
    if grounding.get("top_1_similarity_threshold") != 0.90:
        report.add("error", "mapping_threshold", "tau_map must be 0.90")
    if grounding.get("threshold_comparison") != "greater_than":
        report.add("error", "mapping_comparator", "embedding similarity must strictly exceed tau_map")
    filtering = _config_section(config, "filtering", report)
    if filtering.get("ccs_hierarchy_max_levels") != 2:
        report.add("error", "ccs_support_depth", "CCS support depth must be two levels")
    clustering = _config_section(config, "clustering", report)
    expected_embedding = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": EMBEDDING_REVISION,
        "embedding_backend": EMBEDDING_BACKEND,
    }
    for key, value in expected_embedding.items():
        if clustering.get(key) != value:
            report.add("error", "clustering_embedding", f"clustering.{key} must be {value!r}")
    if clustering.get("embedding_max_length") != 256:
        report.add("error", "embedding_max_length", "clustering.embedding_max_length must be 256")
    if clustering.get("projection_dim") != 256:
        report.add("error", "projection_dim", "template vectors must use d=256")
    if clustering.get("distance_cut_tau") != 0.16:
        report.add("error", "release_tau", "clustering distance_cut_tau must be 0.16")
    pool_output = _config_section(config, "pool_output", report)
    if pool_output.get("path") != "data/knowledge/templates.jsonl":
        report.add("error", "pool_output_path", "pool_output.path must target data/knowledge/templates.jsonl")
    if pool_output.get("projection_dim") != clustering.get("projection_dim"):
        report.add("error", "pool_output_dim", "pool output and clustering dimensions differ")

    model_stack = manifest.get("model_stack")
    expected_stack = {
        "model": EMBEDDING_MODEL,
        "revision": EMBEDDING_REVISION,
        "backend": EMBEDDING_BACKEND,
    }
    if not isinstance(model_stack, dict) or model_stack.get("grounding_and_embeddings") != expected_stack:
        report.add(
            "error",
            "manifest_embedding_stack",
            "manifest.model_stack.grounding_and_embeddings must match the pinned embedding stack",
        )


def _validate_prompts(root: Path, file_paths: set[str], report: ValidationReport) -> None:
    expected = ("system_prompt.txt", "user_prompt.txt")
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
            report.add("error", "prompt_content", f"release prompt is missing {token!r}")
    if "{1–5}" not in combined and "{1-5}" not in combined:
        report.add("error", "prompt_content", "release prompt is missing the 1-5 cascade bound")


def _validate_schema_contract(
    root: Path,
    relative: str,
    expected_fields: set[str],
    report: ValidationReport,
) -> None:
    schema = _load_json(root / relative, report, "schema_parse_error")
    if not isinstance(schema, dict):
        return
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        report.add(
            "error",
            "schema_object_contract",
            f"{relative} must define a closed object schema",
        )
    required = schema.get("required")
    properties = schema.get("properties")
    required_is_exact = (
        isinstance(required, list)
        and len(required) == len(expected_fields)
        and all(isinstance(field, str) for field in required)
        and set(required) == expected_fields
    )
    if not required_is_exact:
        report.add(
            "error",
            "schema_required_fields",
            f"{relative} must require exactly {sorted(expected_fields)}",
        )
    if not isinstance(properties, dict) or set(properties) != expected_fields:
        report.add(
            "error",
            "schema_properties",
            f"{relative} must define exactly {sorted(expected_fields)}",
        )
        return
    cascade_name = "clinical_cascade" if "clinical_cascade" in properties else "cascade"
    cascade = properties.get(cascade_name)
    if not isinstance(cascade, dict) or any(
        (
            cascade.get("type") != "array",
            cascade.get("minItems") != 1,
            cascade.get("maxItems") != 5,
            cascade.get("uniqueItems") is not True,
        )
    ):
        report.add(
            "error",
            "schema_cascade_contract",
            f"{relative} must enforce a unique 1-5 item cascade",
        )


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

    if manifest.get("schema_version") != "1.0":
        report.add("error", "manifest_schema", "manifest.schema_version must be '1.0'")
    if manifest.get("release_id") != "pool_v1":
        report.add("error", "manifest_release_id", "manifest.release_id must be pool_v1")
    status = manifest.get("status")
    if status != "released":
        report.add("error", "manifest_status", "manifest.status must be released")

    files = manifest.get("files")
    if not isinstance(files, list):
        report.add("error", "manifest_files", "manifest.files must be a list")
        files = []
    manifested_paths: set[str] = set()
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
        if relative in REQUIRED_RELEASE_FILES and entry.get("required") is not True:
            report.add(
                "error",
                "required_file_optional",
                f"required release file must be marked required: {relative}",
            )
        if not isinstance(entry.get("role"), str) or not entry["role"].strip():
            report.add("error", "manifest_file_role", f"{relative}: role must be a non-empty string")
        declared_bytes = entry.get("bytes")
        if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int) or declared_bytes < 0:
            report.add("error", "manifest_file_bytes", f"{relative}: bytes must be a non-negative integer")
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

    missing_manifest_entries = sorted(REQUIRED_RELEASE_FILES - manifested_paths)
    extra_manifest_entries = sorted(manifested_paths - REQUIRED_RELEASE_FILES)
    for relative in missing_manifest_entries:
        report.add("error", "required_file_not_manifested", f"required release file is absent from files[]: {relative}")
    for relative in extra_manifest_entries:
        report.add("error", "unexpected_manifest_file", f"unexpected release file in files[]: {relative}")

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != manifest_path
    }
    for relative in sorted(actual_paths - REQUIRED_RELEASE_FILES):
        report.add("error", "unmanifested_release_file", f"unexpected unmanifested release file: {relative}")

    config_path = root / "config.yaml"
    if not config_path.is_file():
        report.add("error", "config_missing", "config.yaml is required")
    else:
        config = _load_config(config_path, report)
        if config is not None:
            _validate_config(config, manifest, report)
    _validate_prompts(root, manifested_paths, report)
    _validate_schema_contract(
        root,
        "schemas/llm_response.schema.json",
        {"definition", "clinical_cascade"},
        report,
    )
    _validate_schema_contract(
        root,
        "schemas/distilled_artifact.schema.json",
        {"concept_id", "concept_name", "definition", "cascade", "meta"},
        report,
    )

    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a ReTA knowledge release and its hashes.")
    parser.add_argument(
        "--release-dir",
        "--release_dir",
        default=str(Path(__file__).resolve().parent / "releases" / "pool_v1"),
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    report = verify_release(args.release_dir)
    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.format_text())
    return 0 if report.ok() else 2


if __name__ == "__main__":
    raise SystemExit(main())
