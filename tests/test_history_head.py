"""Causal history, compatible artifacts, and fixed-branch import comparisons."""
import unittest
from types import SimpleNamespace

import torch

from reta.learning.history import LinearHistoryHead, history_features, HISTORY_CONTRACT_KEYS
from reta.learning.prediction import PredictionSession
from reta.learning.model import NextVisitPredictor
from reta.learning.checkpointing import validate_resume


class HistoryHeadTests(unittest.TestCase):
    def test_binary_features_exclude_hierarchy_and_retain_only_observed_history(self):
        columns, history = history_features([0, 0, 2, 4, 7], {1}, 4)
        self.assertEqual(columns, [0, 2, 4, 5, 6])
        self.assertEqual(history, {0, 1, 2})
        columns, history = history_features([3], history, 4)
        self.assertEqual(columns, [3, 4, 5, 6, 7])

    def test_linear_scores_and_import_comparison_use_the_same_history(self):
        head = LinearHistoryHead(torch.arange(16).reshape(2, 8), torch.tensor([1., -1.]), 4, .25)
        linear, history = head([0, 2, 2, 7], {1})
        torch.testing.assert_close(linear, torch.tensor([[18., 56.]]))
        raw, edited = torch.tensor([[1., 2.]]), torch.tensor([[3., -2.]])
        torch.testing.assert_close(head.combine(edited, linear) - head.combine(raw, linear), .25 * (edited - raw))
        self.assertEqual(list(head.parameters()), [])

    def test_session_reset_clears_cross_patient_history(self):
        session = SimpleNamespace(observed_history={2}, past_visit_embs=[torch.ones(2)],
                                  baseline_past_visit_embs=[], state_enc=None)
        PredictionSession.reset(session)
        self.assertEqual(session.observed_history, set())
        self.assertEqual(session.past_visit_embs, [])

    def test_reject_invalid_shapes_weights_and_training_provenance(self):
        contract = {key: key for key in HISTORY_CONTRACT_KEYS}
        checkpoint = {'contract': contract, 'history_head': {'kind': 'logit_blend_v1',
            'graph_weight': .25, 'data_contract': dict(contract)},
            'predictor': {'history_head.weight': torch.zeros(2, 8), 'history_head.bias': torch.zeros(2)}}
        LinearHistoryHead.from_checkpoint(checkpoint, 4, 2)
        checkpoint['history_head']['data_contract']['processed_sha256'] = 'another cohort'
        with self.assertRaisesRegex(ValueError, 'training data'):
            LinearHistoryHead.from_checkpoint(checkpoint, 4, 2)
        for weight, bias, alpha in [(torch.zeros(2, 6), torch.zeros(2), .25),
                                     (torch.full((2, 8), float('nan')), torch.zeros(2), .25),
                                     (torch.zeros(2, 8), torch.zeros(2), float('nan'))]:
            with self.assertRaises(ValueError):
                LinearHistoryHead(weight, bias, 4, alpha)

    def test_old_loaders_and_training_cannot_silently_drop_the_history_head(self):
        predictor = NextVisitPredictor(4, 2)
        state = dict(predictor.state_dict(), **{'history_head.weight': torch.zeros(2, 8)})
        with self.assertRaises(RuntimeError):
            predictor.load_state_dict(state, strict=True)
        with self.assertRaisesRegex(ValueError, 'graph-only source'):
            validate_resume({'history_head': {}}, None, 'rl')
