# Demo package

The [online viewer](https://chenc2002.github.io/ReTA/demo/) and offline ZIP replay
three example histories evaluated by the released checkpoint. They share the
frontend in `reta/demo/static/` with the local FastAPI service. New inputs require
the local model; the static viewer never labels recorded responses as live inference.

## Run the complete checkpoint

Install `python -m pip install ".[demo]"`, extract the V2.0 model archive, and run:

```bash
reta demo --bundle /path/to/reta-demo-model --verify
reta demo --bundle /path/to/reta-demo-model
```

The bundle contains:

| File | Purpose |
| --- | --- |
| `checkpoint.pt` | Complete selected checkpoint, including model, baseline, optimizer and RNG states |
| `vocabulary.json` | ICD/CCS token vocabulary and graph dimensions; no patient trajectories |
| `templates.jsonl`, `entities.json` | Knowledge templates and external token mapping |
| `icd2ccs.csv`, `ccs_hierarchy.csv` | Public ontology mappings used to construct edited visits |
| `descriptions.json` | Canonical diagnosis and ontology names |
| `config.yaml` | Portable launch configuration |
| `results.json`, `reference.json` | Aggregate evaluation and recorded example predictions |
| `manifest.json` | File sizes and SHA-256 integrity checks |

Prediction verifies the complete architecture, vocabulary and pool fingerprints.
Training/evaluation additionally verify the original clinical dataset and split
fingerprints. The checkpoint contains frozen text features, so inference does not
need the separate feature cache or a Transformer download. Resuming training still
requires authorized access to the original training resources.

## Refresh the release

On the authorized training host:

```bash
python scripts/export_demo.py --config /path/to/config.yaml \
  --checkpoint /path/to/selected.pt --icd2ccs /path/to/icd2ccs.csv \
  --ccs-hierarchy /path/to/ccs_hierarchy.csv \
  --descriptions /path/to/descriptions.json --results /path/to/aggregate-results.json \
  --out /path/to/reta-demo-model --viewer-data demo/data.js
```

The exporter uses an explicit file list, retains the full checkpoint, and checks
that standalone predictions match the training-host examples. Clinical records
and split manifests stay on the training host. Review the input descriptions,
knowledge files and checkpoint contents before publishing a newly trained bundle.

Build the static viewer with the standard library:

```bash
python scripts/build_demo.py --out build/site/demo
python -m http.server 8000 --directory build/site
```

Visit `http://127.0.0.1:8000/demo/`. GitHub Actions builds and publishes the same
files to Pages. Large model archives belong in GitHub Releases, keeping source
clones small.
