"""Fit a train-only linear history model; select its ReTA blend on validation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from reta.learning.checkpointing import atomic_save
from reta.learning.history import HISTORY_CONTRACT_KEYS, history_features
from reta.learning.inference import compute_all, select_f1_threshold
from reta.learning.prediction import load_prediction_session
from reta.learning.runtime import file_fingerprint, load_checkpoint, load_config, load_processed_data, select_patient_ids


def labeled_visits(data, split_path, split):
    for pid in select_patient_ids(list(data['trajectories']), split_path, split):
        for visit in data['trajectories'][pid][:-1]:
            if visit.get('label_ccs') is not None:
                yield pid, visit


def design_matrix(visits, offset, labels):
    rows, columns, targets = [], [], []
    previous, history = None, set()
    vocab = offset + labels
    for pid, visit in visits:
        if pid != previous:
            history = set()
        features, history = history_features(visit['node_ids'], history, vocab)
        rows.extend([len(targets)] * len(features))
        columns.extend(features)
        target = np.zeros(labels, dtype=np.int32)
        target[np.asarray(visit['label_ccs'], dtype=int) - offset] = 1
        targets.append(target)
        previous = pid
    if not targets:
        raise ValueError('The requested split has no labeled transitions.')
    matrix = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, columns)),
                        shape=(len(targets), 2 * vocab))
    return matrix, np.asarray(targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True, help='Frozen graph-only ReTA checkpoint')
    parser.add_argument('--out', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--C', type=float, default=0.1)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if not np.isfinite(args.C) or args.C <= 0:
        raise ValueError('C must be finite and positive.')
    output, report = Path(args.out), Path(args.report)
    if output.exists() or report.exists() or output.resolve() == report.resolve():
        raise ValueError('Choose distinct, new output paths.')
    torch.set_num_threads(2)
    cfg = load_config(args.config)
    if not cfg.data.split_json or cfg.data.train_split != 'train':
        raise ValueError('A train/validation/test split manifest is required.')
    source_sha256 = file_fingerprint(args.checkpoint)
    checkpoint = load_checkpoint(args.checkpoint)
    if 'history_head' in checkpoint or checkpoint['contract']['training_split'] != 'train':
        raise ValueError('Use a graph-only checkpoint fitted on the training split.')
    data = load_processed_data(cfg.data.processed_path)
    session = load_prediction_session(args.checkpoint, cfg.data.processed_path,
        cfg.knowledge.templates_jsonl, cfg.knowledge.entity_to_token_json,
        cfg.data.split_json, device=args.device, data=data)
    offset, labels = session.icd_vocab_size, session.ccs_label_vocab_size
    train_x, train_y = design_matrix(labeled_visits(data, cfg.data.split_json, 'train'), offset, labels)
    val_visits = list(labeled_visits(data, cfg.data.split_json, 'val'))
    val_x, val_y = design_matrix(val_visits, offset, labels)
    counts = train_y.sum(axis=0)
    active = (counts > 0) & (counts < len(train_y))
    weight = np.zeros((labels, train_x.shape[1]), dtype=np.float32)
    prevalence = (counts + 0.5) / (len(train_y) + 1)
    bias = np.log(prevalence / (1 - prevalence)).astype(np.float32)
    if active.any():
        model = OneVsRestClassifier(LogisticRegression(C=args.C, solver='liblinear',
            max_iter=500, random_state=cfg.train.seed), n_jobs=args.jobs).fit(train_x, train_y[:, active])
        for label, estimator in zip(np.flatnonzero(active), model.estimators_):
            weight[label] = estimator.coef_[0]
            bias[label] = estimator.intercept_[0]
    print('Linear history model fitted on training patients.', flush=True)
    graph_logits, previous = [], None
    for pid, visit in val_visits:
        if pid != previous:
            session.reset()
        logits, _ = session.predict_visit(visit)
        graph_logits.append(logits)
        previous = pid
    graph_logits = torch.cat(graph_logits)
    linear_logits = torch.from_numpy(val_x @ weight.T + bias)
    records = []
    for alpha in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        probs = torch.sigmoid(alpha * graph_logits + (1 - alpha) * linear_logits).numpy()
        threshold = select_f1_threshold(val_y, probs)
        records.append({'graph_weight': alpha, 'prediction_threshold': threshold,
                        'metrics': compute_all(val_y, probs=probs, threshold=threshold)})
    selected = max(records, key=lambda record: record['metrics']['AUPRC_macro'])
    if file_fingerprint(args.checkpoint) != source_sha256:
        raise ValueError('Source checkpoint changed during fitting.')
    checkpoint['graph_selection'] = checkpoint.get('selection')
    checkpoint['history_head'] = {'kind': 'logit_blend_v1', 'graph_weight': selected['graph_weight'],
        'linear_C': args.C, 'graph_source_sha256': source_sha256,
        'data_contract': {key: checkpoint['contract'][key] for key in HISTORY_CONTRACT_KEYS}}
    checkpoint['predictor']['history_head.weight'] = torch.from_numpy(weight)
    checkpoint['predictor']['history_head.bias'] = torch.from_numpy(bias)
    checkpoint['selection'] = {'split': 'val', 'metric': 'AUPRC_macro',
        'score': selected['metrics']['AUPRC_macro'],
        'prediction_threshold': selected['prediction_threshold']}
    atomic_save(checkpoint, output)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({'selection': selected, 'candidates': records,
        'method': checkpoint['history_head'], 'checkpoint_sha256': file_fingerprint(output)}, indent=2) + '\n')
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == '__main__':
    main()
