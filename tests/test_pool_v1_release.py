"""Release-level integrity tests for the public pool_v1 package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reta.knowledge.release import verify_release
from reta.cli import main as reta_cli_main


RELEASE = ROOT / "reta" / "knowledge" / "releases" / "pool_v1"


class PoolV1ReleaseTests(unittest.TestCase):
    def test_checked_in_release_is_valid(self) -> None:
        report = verify_release(RELEASE)
        self.assertTrue(report.ok(), report.format_text())
        self.assertFalse(report.issues)

    def test_cli_validates_public_release_by_default(self) -> None:
        self.assertEqual(reta_cli_main(["validate-pool-release"]), 0)

    def test_release_status_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pool_v1"
            shutil.copytree(RELEASE, copy)
            manifest_path = copy / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = "draft"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            report = verify_release(copy)
            self.assertFalse(report.ok())
            self.assertIn("manifest_status", {issue.code for issue in report.issues})

    def test_nonfinite_manifest_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pool_v1"
            shutil.copytree(RELEASE, copy)
            manifest_path = copy / "manifest.json"
            payload = manifest_path.read_text(encoding="utf-8").replace(
                '"version": "1.0.0"',
                '"version": NaN',
            )
            manifest_path.write_text(payload, encoding="utf-8")

            report = verify_release(copy)
            self.assertFalse(report.ok())
            self.assertIn("manifest_parse_error", {issue.code for issue in report.issues})

    def test_overflowed_manifest_json_float_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pool_v1"
            shutil.copytree(RELEASE, copy)
            manifest_path = copy / "manifest.json"
            payload = manifest_path.read_text(encoding="utf-8").replace(
                '"version": "1.0.0"',
                '"version": 1e999',
                1,
            )
            manifest_path.write_text(payload, encoding="utf-8")

            report = verify_release(copy)
            self.assertFalse(report.ok())
            self.assertIn("manifest_parse_error", {issue.code for issue in report.issues})

    def test_duplicate_manifest_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pool_v1"
            shutil.copytree(RELEASE, copy)
            manifest_path = copy / "manifest.json"
            payload = manifest_path.read_text(encoding="utf-8").replace(
                '"status": "released"',
                '"status": "released",\n  "status": "draft"',
                1,
            )
            manifest_path.write_text(payload, encoding="utf-8")

            report = verify_release(copy)
            self.assertFalse(report.ok())
            self.assertIn("manifest_parse_error", {issue.code for issue in report.issues})

    def test_duplicate_schema_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pool_v1"
            shutil.copytree(RELEASE, copy)
            schema_path = copy / "schemas" / "llm_response.schema.json"
            payload = schema_path.read_text(encoding="utf-8").replace(
                '"type": "object"',
                '"type": "object",\n  "type": "array"',
                1,
            )
            schema_path.write_text(payload, encoding="utf-8")

            report = verify_release(copy)
            self.assertFalse(report.ok())
            codes = {issue.code for issue in report.issues}
            self.assertIn("json_parse_error", codes)
            self.assertIn("schema_parse_error", codes)

    def test_checksum_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pool_v1"
            shutil.copytree(RELEASE, copy)
            prompt = copy / "system_prompt.txt"
            prompt.write_text(prompt.read_text(encoding="utf-8") + " changed", encoding="utf-8")
            report = verify_release(copy)
            self.assertFalse(report.ok())
            self.assertIn("checksum_mismatch", {issue.code for issue in report.issues})

    def test_manifest_must_cover_complete_release(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pool_v1"
            shutil.copytree(RELEASE, copy)
            manifest_path = copy / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"] = [
                entry
                for entry in manifest["files"]
                if entry["path"] != "schemas/llm_response.schema.json"
            ]
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

            report = verify_release(copy)
            self.assertFalse(report.ok())
            self.assertIn("required_file_not_manifested", {issue.code for issue in report.issues})

    def test_unmanifested_release_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pool_v1"
            shutil.copytree(RELEASE, copy)
            (copy / "extra.txt").write_text("unexpected\n", encoding="utf-8")

            report = verify_release(copy)
            self.assertFalse(report.ok())
            self.assertIn("unmanifested_release_file", {issue.code for issue in report.issues})

    def test_hidden_unmanifested_release_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pool_v1"
            shutil.copytree(RELEASE, copy)
            (copy / ".DS_Store").write_bytes(b"metadata")

            report = verify_release(copy)
            self.assertFalse(report.ok())
            self.assertIn("unmanifested_release_file", {issue.code for issue in report.issues})

    def test_schema_contract_is_enforced_after_manifest_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pool_v1"
            shutil.copytree(RELEASE, copy)
            schema_path = copy / "schemas" / "llm_response.schema.json"
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            schema["properties"]["clinical_cascade"]["maxItems"] = 10
            schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")

            manifest_path = copy / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in manifest["files"]:
                if entry["path"] == "schemas/llm_response.schema.json":
                    payload = schema_path.read_bytes()
                    entry["bytes"] = len(payload)
                    entry["sha256"] = hashlib.sha256(payload).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

            report = verify_release(copy)
            self.assertFalse(report.ok())
            self.assertIn("schema_cascade_contract", {issue.code for issue in report.issues})

    def test_embedding_revision_is_pinned_semantically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pool_v1"
            shutil.copytree(RELEASE, copy)
            config_path = copy / "config.yaml"
            payload = config_path.read_text(encoding="utf-8").replace(
                "d5892b39a4adaed74b92212a44081509db72f87b",
                "main",
            )
            config_path.write_text(payload, encoding="utf-8")

            manifest_path = copy / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            entry = next(item for item in manifest["files"] if item["path"] == "config.yaml")
            config_bytes = config_path.read_bytes()
            entry["bytes"] = len(config_bytes)
            entry["sha256"] = hashlib.sha256(config_bytes).hexdigest()
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

            report = verify_release(copy)
            self.assertFalse(report.ok())
            codes = {issue.code for issue in report.issues}
            self.assertIn("grounding_embedding", codes)
            self.assertIn("clustering_embedding", codes)

    def test_malformed_config_section_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pool_v1"
            shutil.copytree(RELEASE, copy)
            config_path = copy / "config.yaml"
            config_path.write_text("release: invalid\n", encoding="utf-8")

            report = verify_release(copy)
            self.assertFalse(report.ok())
            self.assertIn("config_section_type", {issue.code for issue in report.issues})

    def test_nonfinite_release_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pool_v1"
            shutil.copytree(RELEASE, copy)
            config_path = copy / "config.yaml"
            payload = config_path.read_text(encoding="utf-8").replace(
                "temperature: 0.2",
                "temperature: .nan",
            )
            config_path.write_text(payload, encoding="utf-8")

            report = verify_release(copy)
            self.assertFalse(report.ok())
            self.assertIn("config_nonfinite", {issue.code for issue in report.issues})

    def test_duplicate_release_config_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pool_v1"
            shutil.copytree(RELEASE, copy)
            config_path = copy / "config.yaml"
            config_path.write_text(
                config_path.read_text(encoding="utf-8") + "release:\n  id: shadowed\n",
                encoding="utf-8",
            )

            report = verify_release(copy)
            self.assertFalse(report.ok())
            self.assertIn("config_parse_error", {issue.code for issue in report.issues})

if __name__ == "__main__":
    unittest.main()
