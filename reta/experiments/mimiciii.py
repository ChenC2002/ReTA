"""Prepare the published MIMIC-III cohort and recompute all five demo rows."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml

from reta.data.ontology import CCSOntology, normalize_icd_code
from reta.data.preprocess import preprocess
from reta.demo.bundle import validate_bundle
from reta.learning.inference import compute_all, select_f1_threshold
from reta.learning.prediction import load_prediction_session
from reta.learning.runtime import file_fingerprint, load_config, load_processed_data
from reta.learning.select_checkpoint import select_checkpoint

METRICS = ('AUPRC_macro', 'AUPRC_micro', 'MicroF1', 'Acc@20')
ROWS = ('training_prevalence', 'linear_history', 'stage1', 'stage2', 'fusion')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def cohort_fingerprint(data, splits):
    """Hash logical content, independent of Torch/pickle serialization versions.

    Identifiers remain local. Only the digest of the complete cohort is public.
    Visit order, targets, graphs, vocabulary and all split membership are covered.
    """
    digest = hashlib.sha256()
    def add(value):
        digest.update(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                 default=lambda x: x.tolist(), allow_nan=False).encode())
        digest.update(b'\n')
    add({'format': 1, 'vocabulary': data['vocab']['name_to_token'], 'splits': splits,
         'dimensions': {key: data['meta'].get(key) for key in
                        ('icd_vocab_size', 'ccs_vocab_size', 'ccs_label_vocab_size', 'h_anc')}})
    for pid in sorted(data['trajectories']):
        add([pid, data['trajectories'][pid]])
    return digest.hexdigest()


def check_cohort(data, splits, reference):
    for split in ('train', 'val', 'test'):
        if not splits.get(split):
            raise ValueError(f'Empty {split} split.')
    patients = [pid for values in splits.values() for pid in values]
    if len(patients) != len(set(patients)) or set(patients) != set(data['trajectories']):
        raise ValueError('Splits must be disjoint and cover the complete cohort.')
    actual = cohort_fingerprint(data, splits)
    if actual != reference['cohort_sha256']:
        raise ValueError('Cohort fingerprint differs from the released experiment. Check MIMIC-III version and preparation inputs.')
    return actual


def read_mimic(root, name, columns):
    path = root / (name + '.csv')
    if not path.exists():
        path = root / (name + '.csv.gz')
    return pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False)


def experiment_bundle(args):
    bundle = validate_bundle(args.bundle)
    reference = json.loads(Path(args.reference).read_text())
    if file_fingerprint(bundle/'checkpoint.pt') != reference['checkpoint_sha256']:
        raise ValueError('Use the complete V2.0 demo model specified by the experiment reference.')
    return bundle, reference


def cohort_events(admissions, diagnoses, mapped_codes):
    """Original eligibility rule: >=2 mapped visits; exclude overlapping stays."""
    if admissions.duplicated(['SUBJECT_ID', 'HADM_ID']).any():
        raise ValueError('Admissions must have unique patient/admission pairs.')
    diagnoses = diagnoses.copy()
    diagnoses['icd_code'] = diagnoses.ICD9_CODE.map(normalize_icd_code)
    mapped = diagnoses[diagnoses.icd_code.isin(mapped_codes)]
    visits = mapped[['SUBJECT_ID', 'HADM_ID']].drop_duplicates().merge(
        admissions, on=['SUBJECT_ID', 'HADM_ID'], validate='one_to_one')
    visits['admit'] = pd.to_datetime(visits.ADMITTIME, errors='coerce')
    visits['discharge'] = pd.to_datetime(visits.DISCHTIME, errors='coerce')
    visits = visits.dropna(subset=['admit', 'discharge']).sort_values(['SUBJECT_ID', 'admit', 'HADM_ID'])
    previous = visits.groupby('SUBJECT_ID').discharge.shift()
    overlapping = set(visits.loc[visits.admit <= previous, 'SUBJECT_ID'])
    counts = visits.groupby('SUBJECT_ID').size()
    selected = set(counts[counts >= 2].index) - overlapping
    chosen = admissions[admissions.SUBJECT_ID.isin(selected)]
    events = diagnoses.merge(chosen, on=['SUBJECT_ID', 'HADM_ID'], validate='many_to_one')
    events = events.rename(columns={'SUBJECT_ID': 'patient_id', 'HADM_ID': 'visit_id', 'ADMITTIME': 'timestamp'})
    return events[['patient_id', 'visit_id', 'timestamp', 'icd_code']]


def prepare(args):
    root = Path(args.run).resolve()
    if root.exists():
        raise ValueError('Choose a new run directory; existing data will not be overwritten.')
    bundle, reference = experiment_bundle(args)
    admissions = read_mimic(Path(args.mimic_root), 'ADMISSIONS',
                           ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME'])
    diagnoses = read_mimic(Path(args.mimic_root), 'DIAGNOSES_ICD',
                          ['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE'])
    mapping = pd.read_csv(bundle/'icd2ccs.csv', dtype=str, keep_default_na=False)
    events = cohort_events(admissions, diagnoses, set(mapping.icd))
    root.mkdir(parents=True)
    events_path = root/'events.csv'
    events.to_csv(events_path, index=False)
    ontology = CCSOntology.from_files(str(bundle/'icd2ccs.csv'), str(bundle/'ccs_hierarchy.csv'))
    data = preprocess(str(events_path), ontology, str(root/'processed'),
                      visit_id_col='visit_id', split_seed=42)
    splits = json.loads((root/'processed/splits.json').read_text())
    digest = check_cohort(data, splits, reference)
    write_json(root/'cohort.json', {'cohort_sha256': digest, 'patients': len(data['trajectories']),
               'splits': {key: {'patients': len(ids), 'transitions': sum(len(data['trajectories'][pid])-1 for pid in ids)}
                          for key, ids in splits.items()}})
    print(json.dumps(json.loads((root/'cohort.json').read_text())), flush=True)


def targets(data, ids):
    offset, labels = data['meta']['icd_vocab_size'], data['meta']['ccs_label_vocab_size']
    rows = []
    for pid in ids:
        for visit in data['trajectories'][pid][:-1]:
            y = np.zeros(labels, dtype=np.int32)
            y[np.asarray(visit['label_ccs'], dtype=int) - offset] = 1
            rows.append(y)
    return np.stack(rows)


def experiment_predictions(data, ids, graph, encoder):
    """Run all model branches on exactly the same ordered, causal histories."""
    head = graph.predictor.history_head
    # Keep the graph session's existing policy and frozen uncertainty branch.
    del graph.predictor.history_head
    scores = {key: [] for key in ROWS if key != 'training_prevalence'}
    try:
        for index, pid in enumerate(ids):
            graph.reset()
            encoder.reset()
            history = set()
            for visit in data['trajectories'][pid][:-1]:
                raw, _ = graph.predict_visit(visit)
                stage1, _ = encoder.predict_visit(visit)
                linear, history = head(visit['node_ids'], history)
                linear = linear.cpu()
                for key, logits in [('linear_history', linear), ('stage1', stage1),
                                    ('stage2', raw), ('fusion', head.combine(raw, linear))]:
                    scores[key].append(torch.sigmoid(logits).numpy())
            if (index + 1) % 250 == 0:
                print(f'Evaluated {index+1}/{len(ids)} patients', flush=True)
    finally:
        graph.predictor.history_head = head
        graph.reset()
        encoder.reset()
    return {key: np.concatenate(value) for key, value in scores.items()}


def encoder_session(graph):
    session = copy.copy(graph)
    session.encoder = graph.baseline_encoder
    session.predictor = graph.baseline_predictor
    if session.encoder is None or session.predictor is None:
        raise ValueError('The checkpoint must include the frozen Stage 1 model.')
    session.baseline_encoder = session.baseline_predictor = None
    session.policy_net = session.state_enc = None
    session.past_visit_embs = []
    session.baseline_past_visit_embs = []
    session.observed_history = set()
    return session


def evaluate(args):
    root = Path(args.run).resolve()
    output = Path(args.out).resolve() if args.out else root/('retrained-evaluation' if args.checkpoint else 'released-evaluation')
    if output.exists():
        raise ValueError('Choose a new --out directory; evaluation reports will not be overwritten.')
    bundle, reference = experiment_bundle(args)
    data = load_processed_data(str(root/'processed/processed.pt'))
    split_path = root/'processed/splits.json'
    splits = json.loads(split_path.read_text())
    digest = check_cohort(data, splits, reference)
    # Explicit semantic cohort verification above allows version-independent
    # local preprocessing. Vocabulary/pool/architecture checks remain strict.
    if args.checkpoint:
        graph = load_prediction_session(args.checkpoint, str(root/'processed/processed.pt'),
            str(bundle/'templates.jsonl'), str(bundle/'entities.json'), str(split_path), device=args.device)
    else:
        graph = load_prediction_session(str(bundle/'checkpoint.pt'), None,
            str(bundle/'templates.jsonl'), str(bundle/'entities.json'), device=args.device,
            vocabulary_path=str(bundle/'vocabulary.json'))
    if graph.ck['contract'].get('training_split') != 'train':
        raise ValueError('Held-out evaluation requires a checkpoint fitted on the train split only.')
    if not hasattr(graph.predictor, 'history_head'):
        raise ValueError('Use the complete demo checkpoint including its history head.')
    encoder = encoder_session(graph)
    train_y = targets(data, splits['train'])
    frequency = train_y.mean(axis=0).astype(np.float32)
    val_y = targets(data, splits['val'])
    val_scores = experiment_predictions(data, splits['val'], graph, encoder)
    val_scores['training_prevalence'] = np.broadcast_to(frequency, val_y.shape)
    thresholds = {key: select_f1_threshold(val_y, val_scores[key]) for key in ROWS}
    del val_scores
    test_y = targets(data, splits['test'])
    test_scores = experiment_predictions(data, splits['test'], graph, encoder)
    test_scores['training_prevalence'] = np.broadcast_to(frequency, test_y.shape)
    results = {key: compute_all(test_y, probs=test_scores[key], threshold=thresholds[key]) for key in ROWS}
    for value in results.values():
        value['num_patients'] = len(splits['test'])
    output.mkdir(parents=True, exist_ok=False)
    report = {'cohort_sha256': digest, 'metrics': results, 'validation_thresholds': thresholds,
              'mode': 'retrained' if args.checkpoint else 'released', 'metric_tolerance': 5e-5}
    if not args.checkpoint:
        expected = json.loads((bundle/'results.json').read_text())['metrics']
        report['absolute_errors'] = {key: {m: abs(results[key][m]-expected[key][m]) for m in METRICS} for key in ROWS}
        report['matches_release'] = all(v <= 5e-5 for errors in report['absolute_errors'].values() for v in errors.values())
    write_json(output/'results.json', report)
    with (output/'metrics.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', *METRICS])
        writer.writerows([key, *(results[key][m] for m in METRICS)] for key in ROWS)
    print(json.dumps(report, indent=2), flush=True)
    if report.get('matches_release') is False:
        raise ValueError('Recomputed metrics differ from the release; inspect the comparison report.')


def train(args):
    root = Path(args.run).resolve()
    bundle, reference = experiment_bundle(args)
    data = load_processed_data(str(root/'processed/processed.pt'))
    check_cohort(data, json.loads((root/'processed/splits.json').read_text()), reference)
    features = Path(args.features).resolve()
    if file_fingerprint(features) != reference['concept_features_sha256']:
        raise ValueError('Use the original frozen features from the experiment resources archive.')
    destination = root/'training'
    destination.mkdir(exist_ok=False)
    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg['data'].update(processed_path=str(root/'processed/processed.pt'), split_json=str(root/'processed/splits.json'))
    cfg['knowledge'].update(templates_jsonl=str(bundle/'templates.jsonl'), entity_to_token_json=str(bundle/'entities.json'))
    cfg['model']['concept_features_path'] = str(features)
    cfg['outputs'] = {'checkpoints_dir': str(destination/'checkpoints'), 'logs_dir': str(destination/'logs')}
    cfg['train']['device'] = args.device
    path = destination/'config.yaml'
    # Preserve the original stage boundaries and validation schedule.
    previous = None
    candidates = []
    for epoch in (10, 20, 30, 45, 60):
        cfg['train']['warmup_epochs'] = epoch
        path.write_text(yaml.safe_dump(cfg))
        cmd = [sys.executable, '-m', 'reta.learning.warmup', '--config', str(path)]
        if previous:
            cmd += ['--resume', str(previous)]
        subprocess.run(cmd, check=True)
        previous = destination/'checkpoints/warmup.pt'
        snapshot = destination/f'checkpoints/warmup_epoch{epoch}.pt'
        import shutil
        shutil.copyfile(previous, snapshot)
        candidates.append(snapshot)
    config = load_config(str(path))
    selected = destination/'stage1.pt'
    select_checkpoint(candidates, config, selected, destination/'stage1-selection.json')
    subprocess.run([sys.executable, '-m', 'reta.learning.rl_train', '--config', str(path),
                    '--warmup_ckpt', str(selected)], check=True)
    select_checkpoint([destination/f'checkpoints/rl_iter{i}.pt' for i in (5, 10, 15, 20, 25, 30)],
                      config, destination/'stage2.pt', destination/'stage2-selection.json')
    subprocess.run([sys.executable, '-m', 'reta.learning.fit_history_head', '--config', str(path), '--checkpoint', str(destination/'stage2.pt'),
                    '--out', str(destination/'demo.pt'), '--report', str(destination/'history-selection.json'),
                    '--device', args.device, '--C', '0.1'], check=True)
    print(f'Training complete: {destination / "demo.pt"}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name, function in [('prepare', prepare), ('evaluate', evaluate), ('train', train)]:
        p = sub.add_parser(name)
        p.set_defaults(function=function)
        p.add_argument('--run', required=True, help='Private local experiment directory')
        p.add_argument('--bundle', required=True, help='Extracted complete demo model')
        p.add_argument('--reference', default='experiments/mimiciii.json')
        if name == 'prepare':
            p.add_argument('--mimic-root', required=True)
        else:
            p.add_argument('--device', default='auto')
        if name == 'evaluate':
            p.add_argument('--checkpoint', help='Optional newly trained demo.pt; never compared as published weights')
            p.add_argument('--out', help='New directory for metrics.csv and results.json')
        if name == 'train':
            p.add_argument('--features', required=True)
            p.add_argument('--config', default='configs/demo_mimiciii.yaml')
    args = parser.parse_args(argv)
    os.umask(0o077)
    torch.set_num_threads(2)
    args.function(args)


if __name__ == '__main__':
    main()
