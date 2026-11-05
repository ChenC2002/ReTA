"""Portable demo assets: full checkpoint, public vocabulary, and example traces."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import yaml

from reta.learning.runtime import file_fingerprint, load_config, load_processed_data

FILES = {'checkpoint.pt', 'vocabulary.json', 'templates.jsonl', 'entities.json',
         'descriptions.json', 'icd2ccs.csv', 'ccs_hierarchy.csv', 'config.yaml',
         'results.json', 'reference.json'}


def export_bundle(config, checkpoint, icd2ccs, ccs_hierarchy, descriptions, results, output):
    """Export only explicit model resources; never copy trajectories or splits."""
    from .app import DemoModel, EXAMPLES
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    cfg = load_config(config)
    data = load_processed_data(cfg.data.processed_path)
    public_meta = {key: data['meta'][key] for key in
                   ('icd_vocab_size', 'ccs_vocab_size', 'ccs_label_vocab_size', 'h_anc')}
    vocabulary = {'meta': public_meta, 'vocab': {'name_to_token': data['vocab']['name_to_token']}}
    (root/'vocabulary.json').write_text(json.dumps(vocabulary), encoding='utf-8')
    resources = {'checkpoint.pt': checkpoint, 'templates.jsonl': cfg.knowledge.templates_jsonl,
                 'entities.json': cfg.knowledge.entity_to_token_json, 'icd2ccs.csv': icd2ccs,
                 'ccs_hierarchy.csv': ccs_hierarchy, 'descriptions.json': descriptions,
                 'results.json': results}
    for name, source in resources.items():
        if name == 'entities.json' and source is None:
            (root/name).write_text('{}', encoding='utf-8')
        else:
            shutil.copyfile(source, root/name)
    # Preserve the complete checkpoint, including optimizer/RNG states.
    portable = asdict(cfg)
    portable['data']['processed_path'] = 'vocabulary.json'
    portable['data']['split_json'] = None
    portable['knowledge']['templates_jsonl'] = 'templates.jsonl'
    portable['knowledge']['entity_to_token_json'] = 'entities.json'
    portable['model']['concept_features_path'] = None  # already inside checkpoint
    portable['outputs'] = {'checkpoints_dir': 'checkpoints', 'logs_dir': 'logs'}
    (root/'config.yaml').write_text(yaml.safe_dump(portable), encoding='utf-8')
    model = DemoModel(config, checkpoint, icd2ccs, ccs_hierarchy, descriptions, results=results)
    reference = {'metadata': model.metadata(),
                 'records': [model.predict(example['visits'], top_k=8) for example in EXAMPLES]}
    (root/'reference.json').write_text(json.dumps(reference, allow_nan=False), encoding='utf-8')
    manifest = {'format_version': 1, 'version': 'V2.0',
                'files': {name: {'bytes': (root/name).stat().st_size,
                                 'sha256': file_fingerprint(root/name)} for name in sorted(FILES)}}
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def validate_bundle(directory):
    """Verify every portable resource before inference or experiment preparation."""
    root = Path(directory).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format_version') != 1 or set(manifest.get('files', {})) != FILES:
        raise ValueError('Unsupported or incomplete demo bundle manifest.')
    for name, expected in manifest['files'].items():
        path = root/name
        if path.stat().st_size != expected['bytes'] or file_fingerprint(path) != expected['sha256']:
            raise ValueError(f'Demo bundle checksum mismatch: {name}')
    return root


def load_bundle(directory, device='cpu'):
    from .app import DemoModel
    root = validate_bundle(directory)
    cfg = load_config(root/'config.yaml')
    cfg.knowledge.templates_jsonl = str(root/'templates.jsonl')
    cfg.knowledge.entity_to_token_json = str(root/'entities.json')
    return DemoModel(cfg, str(root/'checkpoint.pt'), str(root/'icd2ccs.csv'),
                     str(root/'ccs_hierarchy.csv'), str(root/'descriptions.json'),
                     device=device, results=str(root/'results.json'), vocabulary=str(root/'vocabulary.json'))


def verify_bundle(directory, device='cpu'):
    """Replay all released examples; compare scores, actions, and graph structure."""
    model = load_bundle(directory, device)
    reference = json.loads((Path(directory)/'reference.json').read_text(encoding='utf-8'))
    maximum_error = 0.0
    for example, expected in zip(reference['metadata']['examples'], reference['records'], strict=True):
        actual = json.loads(json.dumps(model.predict(example['visits'], top_k=8), allow_nan=False))
        for a, b in zip(actual['steps'], expected['steps'], strict=True):
            for key in ('action', 'selected_template', 'added_nodes', 'graph', 'codes'):
                if a[key] != b[key]:
                    raise ValueError(f"Example mismatch: {example['name']} / {key}")
            for p, q in zip(a['predictions'], b['predictions'], strict=True):
                if p['code'] != q['code']:
                    raise ValueError('Example prediction ranking changed.')
                maximum_error = max(maximum_error, abs(p['probability']-q['probability']),
                                    abs(p['before']-q['before']))
    if maximum_error > 1e-5:
        raise ValueError(f'Example prediction error exceeds tolerance: {maximum_error}')
    return {'examples': len(reference['records']), 'max_score_error': maximum_error,
            'actions_and_graphs_match': True}
