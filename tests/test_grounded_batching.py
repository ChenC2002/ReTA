import unittest
from pathlib import Path
from types import SimpleNamespace
import torch

from reta.knowledge.pool import KnowledgePool
from reta.learning.runtime import grounded_candidate_ids, training_positive_weights, encoder_code_embedding_lookup
from reta.learning.warmup import iter_visit_batches
from reta.learning.model import DecoupledMultiGATEncoder, build_pyg_data, batch_graphs


class GroundedBatchingTests(unittest.TestCase):
    def test_bulk_retrieval_uses_live_projection_and_preserves_token_order(self):
        torch.manual_seed(4)
        model = DecoupledMultiGATEncoder(5, dim=4, attn_heads=2, gnn_layers=1,
                                        text_features=torch.randn(5,6), text_projection=torch.randn(4,6))
        lookup = encoder_code_embedding_lookup(model)
        tokens = [4,1,4,0]
        for _ in range(2):
            expected = torch.stack([torch.from_numpy(lookup(t)) for t in tokens])
            torch.testing.assert_close(torch.from_numpy(lookup.batch_lookup(tokens)), expected)
            with torch.no_grad():
                model.embed.projection.weight.add_(0.1)
        with self.assertRaises(ValueError):
            lookup.batch_lookup([5])

    def test_class_weights_use_selected_training_patients_and_binary_targets(self):
        trajectories = {'train': [{'label_ccs':[5,5]}, {'label_ccs':[]}, {}],
                        'heldout': [{'label_ccs':[6]}, {}]}
        sampler = SimpleNamespace(patient_ids=['train'], get_patient_traj=trajectories.__getitem__)
        weights = training_positive_weights(sampler, 5, 2, power=0.5)
        torch.testing.assert_close(weights, torch.tensor([1.0, 3.0**0.5]))
        self.assertIsNone(training_positive_weights(sampler, 5, 2))

    def test_filter_never_uses_targets_or_falls_back_to_unrelated_knowledge(self):
        pool = KnowledgePool.load_jsonl(str(Path(__file__).resolve().parents[1] / 'examples/tiny_templates.jsonl'))
        mapper = SimpleNamespace(base_mapping={'CCS:49': 1}, external_mapping={})
        self.assertEqual(grounded_candidate_ids(pool, mapper, {'node_ids':[1]}, True), [0])
        ids = grounded_candidate_ids(pool, mapper, {'node_ids':[2], 'label_ccs':[1]}, True)
        self.assertEqual(ids, [])
        result = pool.retrieve_topk([2], lambda _: (_ for _ in ()).throw(AssertionError()), candidate_ids=ids)
        self.assertEqual(result.template_ids, [])
        self.assertEqual(grounded_candidate_ids(pool, mapper, {}, False), None)

    def test_batches_preserve_patient_order_and_cover_every_transition(self):
        trajectories = {p: [dict(label_ccs=[1], t=t) for t in range(n)]
                        for p,n in [('a',4),('b',2),('c',3)]}
        sampler = SimpleNamespace(get_patient_traj=trajectories.__getitem__)
        batches = list(iter_visit_batches(sampler, list(trajectories), 2))
        seen = {p:[] for p in trajectories}
        for batch in batches:
            self.assertLessEqual(len(batch), 2)
            self.assertEqual(len({p for p,_ in batch}), len(batch))
            for p,v in batch:
                seen[p].append(v['t'])
        self.assertEqual(seen, {'a':[0,1,2], 'b':[0], 'c':[0,1]})

    def test_batched_encoder_matches_separate_patients_with_soft_mask(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        model = DecoupledMultiGATEncoder(8, dim=4, attn_heads=2, gnn_layers=1, dropout=0).eval()
        graphs = [build_pyg_data([0,1], torch.tensor([[0,1],[1,0]])),
                  build_pyg_data([2,3,4], torch.tensor([[0,1,1,2],[1,0,2,1]]))]
        memory = [torch.randn(2,4), torch.empty(0,4)]
        offsets = torch.randn(2,4)
        with torch.no_grad():
            actual,_ = model(batch_graphs(graphs), past_visit_memory=memory,
                             soft_offset_per_graph=offsets, soft_import_mask=torch.tensor([True,False]))
            expected = torch.cat([model(batch_graphs([g]), past_visit_memory=[memory[i]],
                                       soft_offset_per_graph=offsets[i:i+1] if i==0 else None)[0]
                                  for i,g in enumerate(graphs)])
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
