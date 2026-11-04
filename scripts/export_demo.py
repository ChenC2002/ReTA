"""Package a trained demo without copying patient records or split manifests."""
import argparse
import json
from pathlib import Path

import torch
from reta.demo.bundle import export_bundle, verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'icd2ccs', 'ccs-hierarchy', 'descriptions', 'results', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--viewer-data', help='Write example traces as a static viewer data.js')
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = export_bundle(args.config, args.checkpoint, args.icd2ccs, args.ccs_hierarchy,
                             args.descriptions, args.results, args.out)
    print(json.dumps(verify_bundle(args.out), indent=2))
    if args.viewer_data:
        destination = Path(args.viewer_data)
        destination.parent.mkdir(parents=True, exist_ok=True)
        reference = json.loads((Path(args.out)/'reference.json').read_text(encoding='utf-8'))
        destination.write_text('window.RETA_DEMO = '+json.dumps(reference, ensure_ascii=True, allow_nan=False)+';\n', encoding='utf-8')
    print('Bundle bytes:', sum(record['bytes'] for record in manifest['files'].values()))


if __name__ == '__main__':
    main()
