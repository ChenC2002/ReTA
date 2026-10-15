# Data preparation and training

Run these commands from the repository root.

## Required Inputs

Full runs require credentialed MIMIC-III or MIMIC-IV diagnosis data, ICD-to-CCS
mappings, a CCS hierarchy, PrimeKG or UMLS-derived resources, and externally
generated concept-level distillation responses.

### EHR Files

| Input                   | Required format                                              |
| ----------------------- | ------------------------------------------------------------ |
| Diagnosis events        | CSV with `patient_id,timestamp,icd_code`; add `icd_version` for mixed ICD-9/10 data |
| ICD-to-CCS mapping      | CSV with `icd,ccs`; add `icd_version` for a versioned mapping |
| Preprocessing hierarchy | CSV with `child,parent`                                      |

### Knowledge Files

| Input                  | Required format                                              |
| ---------------------- | ------------------------------------------------------------ |
| Concept inventory      | CSV with `concept_id,concept_name` and optional `density` (`sparse`, `moderate`, or `dense`) |
| Distillation responses | JSONL with exactly `definition` and `clinical_cascade`, one response per concept in matching row order |
| Biomedical inventory   | CSV with `entity_id,name` and optional `source`              |
| Embedding candidates   | Optional CSV/JSONL with `mention,entity_id,score`            |
| PrimeKG support        | CSV with `u,v` and optional `relation`                       |
| CCS filtering support  | CSV with `parent,child`                                      |

Both hierarchy readers use the named `parent` and `child` columns, so column
order does not matter. Preserve the parent/child direction and compatible CCS
identifiers; preprocessing additionally requires a single-parent hierarchy.

## Workflow

### Step 1: Preprocess

The preprocessor treats `timestamp` as an event time and creates fixed 24-hour
windows anchored at each patient's first valid event.

```bash
python -m reta.data.preprocess \
  --events data/raw/diagnoses.csv \
  --icd2ccs data/resources/icd_to_ccs.csv \
  --ccs_hierarchy data/resources/ccs_hierarchy.csv \
  --out_dir data/processed \
  --h_anc 2
```

For mixed ICD-9/10 input, include an `icd_version` column in both files and
append `--icd_version_col icd_version` and
`--mapping_icd_version_col icd_version` to the command.


- `data/processed/processed.pt`, containing vocabularies and patient
  trajectories; and
- `data/processed/splits.json`, containing deterministic, patient-disjoint
  train/validation/test identifiers.

### Step 2: Build the Pool

The versioned prompts, schemas, thresholds, and pinned model revision are in
[`reta/knowledge/releases/pool_v1`](../reta/knowledge/releases/pool_v1). Generate
one schema-valid response per concept outside this repository, then validate
and serialize those responses. The command below performs no LLM request and
stores no API credentials.

```bash
python -m reta.knowledge.distill \
  --concepts_csv data/resources/concepts.csv \
  --responses_jsonl data/resources/distillation_responses.jsonl \
  --model_family GPT-4o \
  --out_jsonl data/knowledge/artifacts.jsonl
```

Ground mentions and retain only externally supported relations:

```bash
python -m reta.knowledge.filtering \
  --artifacts-jsonl data/knowledge/artifacts.jsonl \
  --inventory-csv data/resources/inventory.csv \
  --embedding-candidates data/resources/clinicalbert_top1.csv \
  --primekg-edges-csv data/resources/primekg_edges.csv \
  --ccs-edges-csv data/resources/ccs_support_edges.csv \
  --out-grounded-jsonl data/knowledge/grounded.jsonl \
  --out-audit-jsonl results/filtering_audit.jsonl \
  --out-summary-json results/filtering_summary.json
```

Filtering uses precomputed embedding candidates and performs no model download.
Clustering loads the exact Bio_ClinicalBERT revision recorded in `pool_v1` and
may download it if it is not already cached:

```bash
python -m reta.knowledge.clustering \
  --grounded_jsonl data/knowledge/grounded.jsonl \
  --out_jsonl data/knowledge/templates.jsonl \
  --tau 0.16 \
  --projection_dim 256
```

Validate the pool and assign stable token IDs to external medoid-subgraph nodes
that are not already in the processed ICD/CCS vocabulary:

```bash
reta validate-template-pool \
  --templates_jsonl data/knowledge/templates.jsonl \
  --expected_dim 256

reta build-entity-map \
  --processed_path data/processed/processed.pt \
  --templates_jsonl data/knowledge/templates.jsonl \
  --out data/resources/entity_to_token.json
```

### Step 3: Train

Review [`configs/reta.yaml`](../configs/reta.yaml) before training. Its defaults
match the main paper settings where implemented.

```bash
python -m reta.learning.warmup --config configs/reta.yaml
python -m reta.learning.rl_train \
  --config configs/reta.yaml \
  --warmup_ckpt checkpoints/warmup.pt
```

Stage 1 writes `checkpoints/warmup.pt`. Stage 2 requires that checkpoint and
writes `checkpoints/rl_iterN.pt` after every iteration. Stage 1 saves after each
epoch. Both stages save optimizer and RNG state for boundary-level recovery:

```bash
python -m reta.learning.warmup --config configs/reta.yaml --resume checkpoints/warmup.pt
python -m reta.learning.rl_train --config configs/reta.yaml --resume checkpoints/rl_iter10.pt
```

Keep the original configuration; increasing the total epochs/iterations and
changing output directories is supported. Legacy weight-only checkpoints remain
valid for inference but cannot resume training. Stage-2 rewards compare raw and
edited visits under the same live encoder; the frozen Stage-1 model supplies
uncertainty only. This corrects older training behavior and requires retraining
to measure its effect.

Stage-2 supervised refinement uses real PyG graph minibatches with per-graph
history and Soft Import masks. Stage-1 transitions and policy rollouts remain
sequential. Each job uses one device. For independent seeds on Slurm, activate
your environment and submit from the repository root:

```bash
RETA_CONFIG=configs/reta.yaml sbatch -p YOUR_GPU_PARTITION scripts/train-slurm.sh
```

The array runs five seeds with at most two concurrent jobs and unique output
paths, then selects each seed's checkpoint on validation patients. Choose the partition and resource limits for your cluster; this is
independent-seed parallelism, not DDP.

### Step 4: Select on validation, then evaluate

Select among completed Stage-2 checkpoints using the validation split only:

```bash
python -m reta.learning.select_checkpoint --config configs/reta.yaml \
  --checkpoints checkpoints/rl_iter*.pt --out checkpoints/best.pt \
  --report results/selection.json
```

The selector rejects checkpoints trained on `val`, `test`, or `all`, checks
input fingerprints, and writes the selection metric and candidate scores.
The default metric is micro average precision. Equal scores use sorted filename
order. Evaluate the selected checkpoint on the test set after selection:


Inference checks the processed data, split, template pool, vocabulary, entity
mapping, and model fingerprints against the checkpoint contract.

```bash
python -m reta.learning.inference \
  --checkpoint checkpoints/best.pt \
  --processed_path data/processed/processed.pt \
  --templates_jsonl data/knowledge/templates.jsonl \
  --entity_to_token_json data/resources/entity_to_token.json \
  --split_json data/processed/splits.json \
  --split test \
  --out results/inference.json
```

Add `--max_patients 10` for a small checkpoint-backed evaluation. Add
`--include-sample-metadata` to record action metadata keyed by run-local patient
indices. A Hard Import that cannot attach any new structure is recorded as a
no-op rather than as a successful structural edit.

Local inference reports `AUPRC_micro`, `MicroF1@0.5`, and `Acc@20` on the
`[0, 1]` scale.

## Outputs

Paper-reported numbers are included as synchronized reference artifacts in
[`results/paper_results.json`](../results/paper_results.json) and
[`logs/paper_results.jsonl`](../logs/paper_results.jsonl).

| Output                                       | Path                                                         |
| -------------------------------------------- | ------------------------------------------------------------ |
| Paper reference results                      | `results/paper_results.json`, `logs/paper_results.jsonl`     |
| Distilled, grounded, and clustered knowledge | `data/knowledge/{artifacts,grounded,templates}.jsonl`        |
| Filtering diagnostics                        | `results/filtering_audit.jsonl`, `results/filtering_summary.json` |
| Training checkpoints                         | `checkpoints/warmup.pt`, `checkpoints/rl_iter*.pt`           |
| Local diagnosis evaluation                   | `results/inference.json`                                     |
| Runtime logs                                 | `logs/warmup.log`, `logs/rl_train.log`, `logs/inference.log` |
