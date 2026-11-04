"""Optional frozen diagnosis-history head for the demo's inference ensemble."""
from __future__ import annotations

import torch
from torch import nn


HISTORY_CONTRACT_KEYS = ('processed_sha256', 'training_split_sha256', 'base_mapping_sha256')


def history_features(node_ids, history, vocab_size):
    """Binary current codes and causal history; exclude hierarchy/external nodes."""
    observed = {int(token) for token in node_ids if 0 <= int(token) < vocab_size}
    updated = set(history) | observed
    return sorted(observed | {vocab_size + token for token in updated}), updated


class LinearHistoryHead(nn.Module):
    def __init__(self, weight, bias, vocab_size, graph_weight):
        super().__init__()
        if weight.ndim != 2 or weight.shape[1] != 2 * vocab_size or bias.shape != (weight.shape[0],):
            raise ValueError('History-head dimensions do not match the diagnosis vocabulary.')
        if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
            raise ValueError('History-head weights must be finite.')
        if not 0 <= float(graph_weight) <= 1:
            raise ValueError('History-head graph_weight must be in [0, 1].')
        self.vocab_size = int(vocab_size)
        self.graph_weight = float(graph_weight)
        self.register_buffer('weight', weight.detach().float().clone())
        self.register_buffer('bias', bias.detach().float().clone())

    def forward(self, node_ids, history):
        columns, updated = history_features(node_ids, history, self.vocab_size)
        logits = self.bias + self.weight[:, columns].sum(dim=1)
        return logits.unsqueeze(0), updated

    def combine(self, graph_logits, history_logits):
        return self.graph_weight * graph_logits + (1 - self.graph_weight) * history_logits

    @classmethod
    def from_checkpoint(cls, checkpoint, vocab_size, num_labels):
        meta = checkpoint['history_head']
        if not isinstance(meta, dict) or meta.get('kind') != 'logit_blend_v1':
            raise ValueError('Unsupported history-head format.')
        if meta.get('data_contract') != {key: checkpoint['contract'][key] for key in HISTORY_CONTRACT_KEYS}:
            raise ValueError('History-head training data or vocabulary does not match the graph checkpoint.')
        state = checkpoint['predictor']
        weight, bias = state.get('history_head.weight'), state.get('history_head.bias')
        if not torch.is_tensor(weight) or not torch.is_tensor(bias) or weight.ndim != 2 or weight.shape[0] != num_labels:
            raise ValueError('Checkpoint is missing compatible history-head weights.')
        return cls(weight, bias, vocab_size, meta.get('graph_weight', float('nan')))
