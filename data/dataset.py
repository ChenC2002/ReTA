"""
This dataset expects `processed.pt` created by `data/preprocess.py`.

Each sample corresponds to a *transition* (visit t -> visit t+1):
- inputs: visit graph at time t (node_ids, edge_index) and visit ICD tokens
- targets: multi-label CCS categories at time t+1

Graph format is PyG-compatible edge_index (2, E).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class TransitionSample:
    patient_id: str
    t: int
    icd_tokens: torch.LongTensor          # (n_icd,)
    node_ids: torch.LongTensor            # (n_nodes,)
    edge_index: torch.LongTensor          # (2, E)
    label_ccs: torch.LongTensor           # (n_labels,) as index list (multi-label)


class TrajectoryDataset(Dataset):
    """A dataset of visit transitions."""

    def __init__(self, processed_path: str, split: str = 'all', split_info: Optional[Dict]=None):
        """Create dataset.

        Parameters
        ----------
        processed_path: path to processed.pt
        split: 'train'|'val'|'test'|'all'
        split_info: optional dict {split: [patient_ids]} for patient-level splits.
        """
        super().__init__()
        self.data = torch.load(processed_path, map_location='cpu')
        self.meta = self.data['meta']
        self.trajectories = self.data['trajectories']

        # Split handling
        if split != 'all':
            if split_info is None or split not in split_info:
                raise ValueError("split_info must be provided with patient lists when split != 'all'.")
            keep = set(map(str, split_info[split]))
            self.patient_ids = [pid for pid in self.trajectories.keys() if pid in keep]
        else:
            self.patient_ids = list(self.trajectories.keys())

        self.samples: List[Tuple[str, int]] = []
        for pid in self.patient_ids:
            traj = self.trajectories[pid]
            # last visit has no label
            for t in range(len(traj) - 1):
                if traj[t].get('label_ccs') is None:
                    continue
                self.samples.append((pid, t))

        # label space size if needed for multi-hot conversion
        self.ccs_vocab_size = int(self.meta.get('ccs_vocab_size', 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pid, t = self.samples[idx]
        visit = self.trajectories[pid][t]

        icd_tokens = torch.tensor(visit['icd_tokens'], dtype=torch.long)
        node_ids = torch.tensor(visit['node_ids'], dtype=torch.long)
        edge_index = torch.tensor(visit['edge_index'], dtype=torch.long)
        label_ccs = torch.tensor(visit['label_ccs'], dtype=torch.long)

        return {
            'patient_id': pid,
            't': t,
            'icd_tokens': icd_tokens,
            'node_ids': node_ids,
            'edge_index': edge_index,
            'label_ccs': label_ccs,
        }


def labels_to_multihot(label_idx: torch.LongTensor, num_labels: int) -> torch.FloatTensor:
    """Convert index-list labels to multi-hot vector."""
    y = torch.zeros(num_labels, dtype=torch.float32)
    if label_idx.numel() > 0:
        y[label_idx] = 1.0
    return y


def collate_transitions(batch: List[Dict[str, Any]], num_labels: Optional[int] = None) -> Dict[str, Any]:
    """Collate function.

    Returns padded tensors for ICD token lists and graph nodes.

    Notes
    -----
    For GNNs (PyG), a common approach is to build a big batch graph with node offsets.
    Here we provide a lightweight collate that returns per-sample graphs as lists;
    you can later convert to PyG Batch if you prefer.
    """
    patient_ids = [b['patient_id'] for b in batch]
    ts = torch.tensor([b['t'] for b in batch], dtype=torch.long)

    icd_tokens = [b['icd_tokens'] for b in batch]
    node_ids = [b['node_ids'] for b in batch]
    edge_index = [b['edge_index'] for b in batch]

    label_idx = [b['label_ccs'] for b in batch]

    if num_labels is None:
        # keep as indices
        labels = label_idx
    else:
        labels = torch.stack([labels_to_multihot(li, num_labels) for li in label_idx], dim=0)

    return {
        'patient_id': patient_ids,
        't': ts,
        'icd_tokens': icd_tokens,
        'node_ids': node_ids,
        'edge_index': edge_index,
        'labels': labels,
    }
