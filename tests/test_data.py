"""Data-contract tests that run when NumPy and pandas are available."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


HAS_DATA_DEPS = all(importlib.util.find_spec(name) is not None for name in ("numpy", "pandas"))


@unittest.skipUnless(HAS_DATA_DEPS, "NumPy and pandas are not installed")
class DataContractTests(unittest.TestCase):
    def _files(self, root: Path, mapping: str, hierarchy: str):
        mapping_path = root / "mapping.csv"
        hierarchy_path = root / "hierarchy.csv"
        mapping_path.write_text(mapping, encoding="utf-8")
        hierarchy_path.write_text(hierarchy, encoding="utf-8")
        return mapping_path, hierarchy_path

    def test_codes_stay_strings_and_direct_labels_are_first(self) -> None:
        from reta.data.ontology import CCSOntology

        with tempfile.TemporaryDirectory() as tmp:
            mapping, hierarchy = self._files(
                Path(tmp),
                "icd,ccs\n001.0,LEAF\n002.0,OTHER\n",
                "child,parent\nLEAF,PARENT\nOTHER,PARENT\n",
            )
            ontology = CCSOntology.from_files(str(mapping), str(hierarchy))

        self.assertIn("0010", ontology.icd_to_ccs)
        self.assertEqual(ontology.ccs_label_vocab_size, 2)
        label_tokens = sorted(ontology._ccs_label_tokens)
        self.assertEqual(label_tokens, list(range(ontology.icd_vocab_size, ontology.icd_vocab_size + 2)))
        self.assertGreater(ontology.ccs_to_token("PARENT"), label_tokens[-1])

    def test_hierarchy_cycle_is_rejected(self) -> None:
        from reta.data.ontology import CCSOntology

        with tempfile.TemporaryDirectory() as tmp:
            mapping, hierarchy = self._files(
                Path(tmp),
                "icd,ccs\nA,ONE\n",
                "child,parent\nONE,TWO\nTWO,ONE\n",
            )
            with self.assertRaisesRegex(ValueError, "cycle"):
                CCSOntology.from_files(str(mapping), str(hierarchy))

    def test_versioned_mapping_keeps_colliding_codes_distinct(self) -> None:
        from reta.data.ontology import CCSOntology

        with tempfile.TemporaryDirectory() as tmp:
            mapping, hierarchy = self._files(
                Path(tmp),
                "icd,version,ccs\n001,9,NINE\n001,10,TEN\n",
                "child,parent\nNINE,ROOT\nTEN,ROOT\n",
            )
            ontology = CCSOntology.from_files(
                str(mapping),
                str(hierarchy),
                icd_version_col="version",
            )

        token9 = ontology.icd_to_token("001", 9)
        token10 = ontology.icd_to_token("001", 10)
        self.assertNotEqual(token9, token10)
        self.assertEqual(ontology.token_to_name[token9], "ICD:9:001")
        self.assertEqual(ontology.token_to_name[token10], "ICD:10:001")
        with self.assertRaisesRegex(ValueError, "requires an ICD version"):
            ontology.icd_to_token("001")

    def test_preprocess_drops_bad_timestamps_and_writes_splits(self) -> None:
        from reta.data.ontology import CCSOntology
        from reta.data import preprocess as preprocessing

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mapping, hierarchy = self._files(
                root,
                "icd,ccs\n001.0,LEAF\n",
                "child,parent\nLEAF,PARENT\n",
            )
            events = root / "events.csv"
            events.write_text(
                "patient_id,timestamp,icd_code\n"
                "p1,2020-01-01T00:00:00Z,001.0\n"
                "p1,2020-01-03T00:00:00Z,001.0\n"
                "p1,not-a-time,001.0\n",
                encoding="utf-8",
            )
            ontology = CCSOntology.from_files(str(mapping), str(hierarchy))
            fake_torch = SimpleNamespace(save=lambda _value, path: Path(path).write_bytes(b"test"))
            with patch.object(preprocessing, "torch", fake_torch):
                dataset = preprocessing.preprocess(
                    str(events),
                    ontology,
                    str(root / "processed"),
                )

            self.assertEqual(dataset["meta"]["mapped_events"], 2)
            self.assertEqual(dataset["meta"]["dropped_invalid_timestamp"], 1)
            self.assertEqual(dataset["meta"]["ccs_label_vocab_size"], 1)
            splits = json.loads((root / "processed" / "splits.json").read_text(encoding="utf-8"))
            self.assertEqual(splits, {"test": [], "train": ["p1"], "val": []})

    def test_visit_bins_use_fixed_24_hour_windows(self) -> None:
        import pandas as pd

        from reta.data.preprocess import make_24h_visit_bins

        events = pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p1"],
                "timestamp": [
                    "2020-01-01T00:00:00Z",
                    "2020-01-01T23:00:00Z",
                    "2020-01-02T22:00:00Z",
                ],
            }
        )

        bins = make_24h_visit_bins(events, "patient_id", "timestamp")

        self.assertEqual(bins.tolist(), [0, 0, 1])

    def test_filtered_ccs_ids_match_ontology_token_keys(self) -> None:
        from reta.data.ontology import CCSOntology
        from reta.knowledge.filtering import Entity, Inventory, SupportIndex, filter_artifact_objects

        with tempfile.TemporaryDirectory() as tmp:
            mapping, hierarchy = self._files(
                Path(tmp),
                "icd,ccs\n001,49\n002,50\n",
                "child,parent\n50,49\n",
            )
            ontology = CCSOntology.from_files(str(mapping), str(hierarchy))
            inventory = Inventory([Entity("49", "Root", "CCS"), Entity("50", "Target", "CCS")])
            result = filter_artifact_objects(
                [
                    {
                        "concept_id": "CCS:49",
                        "concept_name": "Root",
                        "definition": "Definition.",
                        "cascade": ["Target"],
                        "meta": {},
                    }
                ],
                inventory,
                SupportIndex(ccs_edges=[("49", "50")]),
            )

        self.assertEqual(result.grounded[0]["subgraph_nodes"], ["CCS:49", "CCS:50"])
        self.assertTrue(all(node in ontology.name_to_token for node in result.grounded[0]["subgraph_nodes"]))


if __name__ == "__main__":
    unittest.main()
