"""Portable bundle integrity and public launcher behavior."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reta.demo.__main__ import main
from reta.demo.bundle import FILES, load_bundle
from reta.learning.runtime import file_fingerprint, require_checkpoint_contract


class DemoBundleTests(unittest.TestCase):
    def test_no_open_does_not_launch_browser(self):
        with patch('webbrowser.open') as opened:
            self.assertEqual(main(['--no-open']), 0)
            opened.assert_not_called()

    def test_corrupt_bundle_rejected_before_loading_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in FILES:
                (root/name).write_text('example')
            manifest = {'format_version': 1, 'files': {
                name: {'bytes': (root/name).stat().st_size, 'sha256': file_fingerprint(root/name)}
                for name in FILES}}
            (root/'manifest.json').write_text(json.dumps(manifest))
            (root/'checkpoint.pt').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch: checkpoint.pt'):
                load_bundle(root)

    def test_prediction_relaxes_only_private_cohort_identity(self):
        stored = {'processed_sha256': 'private', 'templates_sha256': 'pool',
                  'base_mapping_sha256': 'vocab', 'split_manifest_sha256': 'split',
                  'training_split': 'train', 'training_split_sha256': 'train-hash'}
        current = dict(stored, processed_sha256='public', split_manifest_sha256=None,
                       training_split='all', training_split_sha256='all-hash')
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            require_checkpoint_contract({'contract': stored}, current)
        require_checkpoint_contract({'contract': stored}, current, vocabulary_only=True)
        for key in ('templates_sha256', 'base_mapping_sha256'):
            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                require_checkpoint_contract({'contract': stored}, dict(current, **{key: 'other'}),
                                            vocabulary_only=True)
