"""Experiment cohort identity, admission selection and independent prediction branches."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from reta.experiments.mimiciii import cohort_events, cohort_fingerprint, check_cohort, experiment_predictions
from reta.learning.history import LinearHistoryHead


class ExperimentTests(unittest.TestCase):
    def data(self):
        data = {'vocab': {'name_to_token': {'ICD:1': 0, 'CCS:1': 1}},
                'meta': {'icd_vocab_size': 1, 'ccs_vocab_size': 1, 'ccs_label_vocab_size': 1, 'h_anc': 2},
                'trajectories': {pid: [{'node_ids': [0, 1], 'edge_index': np.array([[0, 1], [1, 0]]),
                                       'icd_tokens': [0], 'label_ccs': [1]},
                                      {'node_ids': [0, 1], 'label_ccs': None}] for pid in 'abc'}}
        return data, {'train': ['a'], 'val': ['b'], 'test': ['c']}

    def test_fingerprint_ignores_serialization_but_detects_semantic_changes(self):
        data, splits = self.data()
        expected = cohort_fingerprint(data, splits)
        equivalent = copy.deepcopy(data)
        for visits in equivalent['trajectories'].values():
            visits[0]['edge_index'] = visits[0]['edge_index'].tolist()
        self.assertEqual(expected, cohort_fingerprint(equivalent, splits))
        for field, value in [('node_ids', [1]), ('label_ccs', []), ('edge_index', [[], []])]:
            altered = copy.deepcopy(data)
            altered['trajectories']['a'][0][field] = value
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                check_cohort(altered, splits, {'cohort_sha256': expected})
        changed = dict(splits, train=['b'], val=['a'])
        self.assertNotEqual(expected, cohort_fingerprint(data, changed))
        data['meta']['ccs_label_vocab_size'] = 2
        self.assertNotEqual(expected, cohort_fingerprint(data, splits))

    def test_split_overlap_rejected_before_comparing_hash(self):
        data, splits = self.data()
        splits['val'] = ['a']
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            check_cohort(data, splits, {})

    def test_admission_cohort_keeps_whole_histories_excludes_overlap(self):
        admissions = pd.DataFrame([
            ['a','1','2100-01-01','2100-01-03'], ['a','2','2100-02-01','2100-02-03'],
            ['b','3','2100-01-01','2100-01-05'], ['b','4','2100-01-04','2100-01-06'],
            ['c','5','2100-01-01','2100-01-02']],
            columns=['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME'])
        diagnoses = admissions[['SUBJECT_ID','HADM_ID']].assign(ICD9_CODE='250.00')
        events = cohort_events(admissions, diagnoses, {'25000'})
        self.assertEqual(events.patient_id.tolist(), ['a','a'])
        self.assertEqual(events.visit_id.tolist(), ['1','2'])
        self.assertEqual(events.icd_code.tolist(), ['25000','25000'])

    def test_prediction_branches_reset_patient_history_and_restore_head(self):
        class Session:
            def __init__(self):
                self.predictor = SimpleNamespace()
                self.memory = 0
            def reset(self):
                self.memory = 0
            def predict_visit(self, visit):
                self.memory += 1
                return torch.tensor([[float(self.memory)]]), {}
        graph, encoder = Session(), Session()
        head = LinearHistoryHead(torch.tensor([[1., 10.]]), torch.zeros(1), 1, .25)
        graph.predictor.history_head = head
        data = {'trajectories': {'a': [{'node_ids': [0]}, {'node_ids': []}, {}],
                                 'b': [{'node_ids': []}, {}]}}
        scores = experiment_predictions(data, ['a','b'], graph, encoder)
        np.testing.assert_allclose(scores['linear_history'].ravel(), torch.sigmoid(torch.tensor([11.,10.,0.])).numpy())
        np.testing.assert_allclose(scores['stage2'].ravel(), torch.sigmoid(torch.tensor([1.,2.,1.])).numpy())
        np.testing.assert_allclose(scores['fusion'].ravel(), torch.sigmoid(torch.tensor([8.5,8.,.25])).numpy())
        self.assertIs(graph.predictor.history_head, head)
        self.assertEqual(graph.memory, 0)
