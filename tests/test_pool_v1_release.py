"""Release-level integrity tests for the frozen pool_v1 package."""

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

from artifacts.pool_v1.verify_release import verify_release
from reta.cli import main as reta_cli_main


RELEASE = ROOT / "artifacts" / "pool_v1"


def _rewrite_demo(release: Path, mutate) -> None:
    pool_path = release / "demo" / "templates.jsonl"
    record = json.loads(pool_path.read_text(encoding="utf-8"))
    mutate(record)
    pool_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(item for item in manifest["files"] if item["path"] == "demo/templates.jsonl")
    payload = pool_path.read_bytes()
    entry["bytes"] = len(payload)
    entry["sha256"] = hashlib.sha256(payload).hexdigest()
    entry["records"] = 1
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


class PoolV1ReleaseTests(unittest.TestCase):
    def test_checked_in_release_is_valid_only_when_incompleteness_is_allowed(self) -> None:
        report = verify_release(RELEASE)
        self.assertFalse(report.ok())
        self.assertTrue(report.ok(allow_incomplete=True), report.format_text(True))
        self.assertFalse(any(issue.severity == "error" for issue in report.issues))
        missing = {issue.message.split(" ", 1)[0] for issue in report.issues if issue.code == "required_pool_missing"}
        self.assertEqual(
            missing,
            {"mimic_iii_primekg", "mimic_iv_primekg", "mimic_iii_umls", "mimic_iv_umls"},
        )
        self.assertEqual(report.checked_template_records, 1)

    def test_cli_is_strict_by_default(self) -> None:
        self.assertEqual(reta_cli_main(["validate-pool-release", "--release-dir", str(RELEASE)]), 2)
        self.assertEqual(
            reta_cli_main(
                ["validate-pool-release", "--release-dir", str(RELEASE), "--allow-incomplete"]
            ),
            0,
        )

    def test_semantic_corruption_is_not_waived(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pool_v1"
            shutil.copytree(RELEASE, copy)
            _rewrite_demo(copy, lambda record: record.__setitem__("vector", [2.0, 0.0, 0.0, 0.0]))
            report = verify_release(copy)
            self.assertFalse(report.ok(allow_incomplete=True))
            self.assertIn("template_vector_norm", {issue.code for issue in report.issues})

    def test_checksum_corruption_is_not_waived(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "pool_v1"
            shutil.copytree(RELEASE, copy)
            prompt = copy / "prompts" / "figure8_system.txt"
            prompt.write_text(prompt.read_text(encoding="utf-8") + " changed", encoding="utf-8")
            report = verify_release(copy)
            self.assertFalse(report.ok(allow_incomplete=True))
            self.assertIn("checksum_mismatch", {issue.code for issue in report.issues})


if __name__ == "__main__":
    unittest.main()
