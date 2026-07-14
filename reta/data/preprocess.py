"""
- Patient trajectories are sequences of visits.
- Each visit aggregates diagnosis codes within a 24-hour window.
- Inputs are ICD codes; targets are CCS categories at the *next* visit.
- Visit graphs contain ICD nodes plus CCS ancestors within <=h levels, with CCS hierarchy edges.

Column names can be overridden to support MIMIC-III/IV exports or other EHR datasets.

Expected input (CSV) at the event level (one row per diagnosis event):
- patient_id: patient identifier
- timestamp: event timestamp (datetime or string)
- icd_code: ICD-9/10 diagnosis code
- icd_version: optional version (9 or 10), required with a versioned mapping

You can override column names via CLI flags.

Outputs:
- processed.pt: a torch-serialized dict with patient trajectories and metadata.
- splits.json: deterministic, disjoint train/validation/test patient IDs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
try:
    import torch
except ModuleNotFoundError:  # Pure preprocessing helpers remain importable without PyTorch.
    torch = None  # type: ignore[assignment]

from .ontology import CCSOntology, normalize_icd_code, normalize_icd_version


# -------------------------
# Utilities
# -------------------------

def _ensure_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors='coerce', utc=True)


def normalize_icd(code: str) -> str:
    """Normalize ICD code string.

    - Strip spaces
    - Uppercase
    - Remove dots for canonicalization (optional)

    Note: Some pipelines keep dots. We remove dots to improve matching for many mapping files.
    """
    if code is None or pd.isna(code):
        return ''
    return normalize_icd_code(code)


def make_24h_visit_bins(df: pd.DataFrame, patient_col: str, time_col: str) -> pd.Series:
    """Assign fixed 24-hour visit windows for each patient.

    Windows are anchored at the patient's first valid event rather than at the
    preceding event. This prevents a chain of events spaced less than 24 hours
    apart from growing into a visit that spans multiple days. Returns an
    integer series aligned with ``df``.
    """
    df = df[[patient_col, time_col]].copy()
    df[time_col] = _ensure_datetime(df[time_col])
    if df[patient_col].isna().any() or df[time_col].isna().any():
        raise ValueError("visit binning requires non-empty patient IDs and valid timestamps")
    # stable ordering for groupby
    df['_row'] = np.arange(len(df))
    df = df.sort_values([patient_col, time_col, '_row'])

    bins = np.zeros(len(df), dtype=np.int64)
    last_pid = None
    anchor_time = None

    for i, (pid, ts) in enumerate(zip(df[patient_col].values, df[time_col].values)):
        if last_pid is None or pid != last_pid:
            last_pid = pid
            anchor_time = ts
        elapsed_h = (ts - anchor_time) / np.timedelta64(1, 'h')
        bins[i] = int(elapsed_h // 24.0)

    df['visit_bin'] = bins
    # revert to original row order
    df = df.sort_values('_row')
    return df['visit_bin']


def build_patient_splits(
    patient_ids: Sequence[str],
    seed: int = 42,
    fractions: Sequence[float] = (0.7, 0.1, 0.2),
) -> Dict[str, list[str]]:
    """Create deterministic, disjoint patient splits using largest remainders."""

    if len(fractions) != 3:
        raise ValueError("split fractions must contain train, validation, and test values")
    values = [float(value) for value in fractions]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("split fractions must be finite and non-negative")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to 1.0")
    if values[0] <= 0.0:
        raise ValueError("the training split fraction must be positive")

    unique_ids = sorted({str(patient_id) for patient_id in patient_ids})
    if len(unique_ids) != len(patient_ids):
        raise ValueError("patient IDs must be unique before splitting")
    if not unique_ids:
        raise ValueError("cannot split an empty patient set")

    rng = np.random.default_rng(int(seed))
    shuffled = [str(value) for value in rng.permutation(np.asarray(unique_ids, dtype=object)).tolist()]
    quotas = [len(shuffled) * value for value in values]
    counts = [math.floor(quota) for quota in quotas]
    remaining = len(shuffled) - sum(counts)
    order = sorted(range(3), key=lambda index: (-(quotas[index] - counts[index]), index))
    for index in order[:remaining]:
        counts[index] += 1
    if counts[0] == 0:
        donor = max((1, 2), key=lambda index: (counts[index], -index))
        if counts[donor] <= 0:
            raise ValueError("unable to allocate a non-empty training split")
        counts[donor] -= 1
        counts[0] += 1

    train_end = counts[0]
    val_end = train_end + counts[1]
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


# -------------------------
# Main preprocessing
# -------------------------

def preprocess(
    events_path: str,
    ontology: CCSOntology,
    out_dir: str,
    patient_col: str = 'patient_id',
    time_col: str = 'timestamp',
    icd_col: str = 'icd_code',
    h_anc: int = 2,
    min_visits: int = 2,
    max_patients: Optional[int] = None,
    icd_version_col: Optional[str] = None,
    split_seed: int = 42,
    train_fraction: float = 0.7,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
) -> Dict:
    """Run preprocessing and serialize outputs.

    Returns the dataset dict (also saved to disk).
    """
    if h_anc < 0:
        raise ValueError("h_anc must be non-negative")
    if min_visits < 2:
        raise ValueError("min_visits must be at least 2 for next-visit prediction")
    if max_patients is not None and max_patients <= 0:
        raise ValueError("max_patients must be positive when supplied")
    split_fractions = (train_fraction, val_fraction, test_fraction)

    # Load events
    # Identifiers must remain strings. Numeric inference irreversibly changes
    # codes such as 001.0 and 250.00 before normalization.
    df = pd.read_csv(events_path, dtype=str, keep_default_na=False)

    required_columns = [patient_col, time_col, icd_col]
    if icd_version_col is not None:
        required_columns.append(icd_version_col)
    for c in required_columns:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in events file. Available: {list(df.columns)[:20]}")

    if ontology.icd_versioned and icd_version_col is None:
        raise ValueError("a versioned ICD mapping requires icd_version_col in the event data")
    if not ontology.icd_versioned and icd_version_col is not None:
        raise ValueError("icd_version_col requires a versioned ICD-to-CCS mapping")

    input_events = int(len(df))
    if input_events == 0:
        raise ValueError("events file contains no records")

    selected_columns = [patient_col, time_col, icd_col]
    if icd_version_col is not None:
        selected_columns.append(icd_version_col)
    df = df[selected_columns].copy()
    df[patient_col] = df[patient_col].astype("string").str.strip()
    missing_patient = df[patient_col].isna() | df[patient_col].eq("")
    dropped_missing_patient = int(missing_patient.sum())
    df = df.loc[~missing_patient].copy()

    df[icd_col] = df[icd_col].map(normalize_icd)
    empty_icd = df[icd_col].str.len().eq(0)
    dropped_empty_icd = int(empty_icd.sum())
    df = df.loc[~empty_icd].copy()

    df[time_col] = _ensure_datetime(df[time_col])
    invalid_timestamp = df[time_col].isna()
    dropped_invalid_timestamp = int(invalid_timestamp.sum())
    df = df.loc[~invalid_timestamp].copy()

    if icd_version_col is not None:
        normalized_versions = []
        for row_index, value in df[icd_version_col].items():
            try:
                normalized_versions.append(normalize_icd_version(value))
            except ValueError as exc:
                raise ValueError(f"invalid ICD version at event row {row_index}: {exc}") from exc
        df[icd_version_col] = normalized_versions

    # Visit bins (24-hour aggregation)
    df['visit_bin'] = make_24h_visit_bins(df, patient_col, time_col)

    # Map ICD->token ids and ICD->CCS
    if icd_version_col is None:
        df['icd_tok'] = df[icd_col].map(ontology.icd_to_token)
    else:
        df['icd_tok'] = [
            ontology.icd_to_token(code, version)
            for code, version in zip(df[icd_col], df[icd_version_col])
        ]
    unmapped_icd = df['icd_tok'].isna()
    dropped_unmapped_icd = int(unmapped_icd.sum())
    df = df.loc[~unmapped_icd].copy()
    if df.empty:
        raise ValueError("no valid, timestamped diagnosis events map to the supplied ICD-to-CCS ontology")
    df['icd_tok'] = df['icd_tok'].astype(int)

    # group to visits
    visits = (df.groupby([patient_col, 'visit_bin'])['icd_tok']
                .apply(lambda s: sorted(set(map(int, s.values.tolist()))))
                .reset_index(name='icd_tokens'))

    # sort visits by bin
    visits = visits.sort_values([patient_col, 'visit_bin'])

    # Build trajectories
    trajectories = {}
    patients = visits[patient_col].unique().tolist()
    if max_patients is not None:
        patients = patients[:max_patients]

    for pid in patients:
        vdf = visits[visits[patient_col] == pid]
        icd_token_lists = vdf['icd_tokens'].tolist()
        if len(icd_token_lists) < min_visits:
            continue

        traj = []
        # Precompute next-visit CCS labels (multi-label ids)
        next_labels = []
        for t in range(len(icd_token_lists) - 1):
            next_icd = icd_token_lists[t + 1]
            next_ccs = sorted(set(ontology.icd_tokens_to_ccs_tokens(next_icd)))
            next_labels.append(next_ccs)
        next_labels.append(None)  # last visit has no next label

        # Build visit graphs
        for t, icd_tokens in enumerate(icd_token_lists):
            ccs_nodes = sorted(set(ontology.icd_tokens_to_ccs_tokens(icd_tokens)))
            anc = sorted(set(ontology.ancestors_within_h(ccs_nodes, h=h_anc)))
            # Visit graph nodes: ICD + CCS ancestors
            node_ids = sorted(set(icd_tokens + anc))
            edge_index = ontology.edges_among_nodes(node_ids)

            traj.append({
                'icd_tokens': icd_tokens,
                'ccs_ancestors': anc,
                'node_ids': node_ids,
                'edge_index': edge_index.astype(np.int64),
                'label_ccs': next_labels[t]
            })

        trajectories[str(pid)] = traj

    if not trajectories:
        raise ValueError(
            f"no patient has the required {min_visits} mapped visits; no trajectory artifact was written"
        )

    splits = build_patient_splits(
        list(trajectories),
        seed=split_seed,
        fractions=split_fractions,
    )
    split_counts = {name: len(patient_ids) for name, patient_ids in splits.items()}

    # metadata
    dataset = {
        'trajectories': trajectories,
        'meta': {
            'patient_col': patient_col,
            'time_col': time_col,
            'icd_col': icd_col,
            'h_anc': h_anc,
            'min_visits': min_visits,
            'num_patients': len(trajectories),
            'num_visits': int(sum(len(v) for v in trajectories.values())),
            'input_events': input_events,
            'mapped_events': int(len(df)),
            'dropped_missing_patient': dropped_missing_patient,
            'dropped_empty_icd': dropped_empty_icd,
            'dropped_invalid_timestamp': dropped_invalid_timestamp,
            'dropped_unmapped_icd': dropped_unmapped_icd,
            'icd_vocab_size': ontology.icd_vocab_size,
            'ccs_vocab_size': ontology.ccs_vocab_size,
            'ccs_label_vocab_size': ontology.ccs_label_vocab_size,
            'token_namespace': 'shared',
            'icd_mapping_versioned': ontology.icd_versioned,
            'icd_version_col': icd_version_col,
            'split_seed': int(split_seed),
            'split_fractions': {
                'train': float(train_fraction),
                'val': float(val_fraction),
                'test': float(test_fraction),
            },
            'split_counts': split_counts,
            'splits_file': 'splits.json',
        },
        'vocab': {
            'token_to_name': ontology.token_to_name,
            'name_to_token': ontology.name_to_token,
            'icd_to_ccs': ontology.icd_to_ccs,
        }
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'processed.pt')
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to serialize processed.pt")
    torch.save(dataset, out_path)
    splits_path = os.path.join(out_dir, 'splits.json')
    with open(splits_path, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(splits, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write('\n')
    return dataset


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Preprocess EHR events into visit trajectories and visit graphs.')
    p.add_argument('--events', type=str, required=True, help='Path to event-level diagnoses CSV.')
    p.add_argument('--icd2ccs', type=str, required=True, help='Path to ICD->CCS mapping CSV.')
    p.add_argument('--ccs_hierarchy', type=str, required=True, help='Path to CCS hierarchy edges CSV.')
    p.add_argument('--out_dir', type=str, required=True, help='Output directory.')
    p.add_argument('--patient_col', type=str, default='patient_id')
    p.add_argument('--time_col', type=str, default='timestamp')
    p.add_argument('--icd_col', type=str, default='icd_code')
    p.add_argument('--h_anc', type=int, default=2)
    p.add_argument('--min_visits', type=int, default=2)
    p.add_argument('--max_patients', type=int, default=None)
    p.add_argument('--icd_version_col', type=str, default=None)
    p.add_argument('--mapping_icd_version_col', type=str, default=None)
    p.add_argument('--split_seed', type=int, default=42)
    p.add_argument('--train_fraction', type=float, default=0.7)
    p.add_argument('--val_fraction', type=float, default=0.1)
    p.add_argument('--test_fraction', type=float, default=0.2)
    return p


def main():
    args = build_argparser().parse_args()
    ontology = CCSOntology.from_files(
        args.icd2ccs,
        args.ccs_hierarchy,
        icd_version_col=args.mapping_icd_version_col,
    )
    preprocess(
        events_path=args.events,
        ontology=ontology,
        out_dir=args.out_dir,
        patient_col=args.patient_col,
        time_col=args.time_col,
        icd_col=args.icd_col,
        h_anc=args.h_anc,
        min_visits=args.min_visits,
        max_patients=args.max_patients,
        icd_version_col=args.icd_version_col,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )


if __name__ == '__main__':
    main()
