# Import What You Need: Learning When and How to Augment EHR Graphs with External Knowledge

ReTA (Reinforcement learning-based dynamic Topology Augmentation) learns when
and how to augment longitudinal EHR graphs with external knowledge. At each
visit, it retrieves from an offline, quality-filtered pool of KG-grounded
templates and selects one budget-aware action: Soft Import for feature-only
semantic enrichment, Hard Import for compact topology grafting, or Skip when
augmentation is unnecessary. A decoupled encoder processes semantic and
structural signals separately before adaptive fusion.

The paper evaluates next-visit diagnosis prediction, in-hospital mortality,
and 30-day readmission on MIMIC-III and MIMIC-IV. This repository includes the
paper-wide numeric results and ordered result log, together with the runnable
implementation of the main next-visit diagnosis workflow.

## Repository and method map

| Path | Method and responsibility |
| --- | --- |
| `reta/data/` | Preserve ICD identifiers, combine events in fixed 24-hour windows, map ICD-9/10 to CCS, build visit graphs, and write local split manifests. |
| `reta/knowledge/` | Validate distillation responses, ground and filter cascades, cluster ClinicalBERT representations, and retrieve Top-K templates. |
| `reta/learning/model.py` | Encode semantic and structural visit views, apply Soft/Hard augmentation, and predict direct next-visit CCS labels. |
| `reta/learning/policy.py` | Define the masked Soft/Hard/Skip action space, policy state, paired reward, and REINFORCE update. |
| `reta/learning/runtime.py` | Enforce configuration, vocabulary, retrieval-space, split, and checkpoint contracts shared by training and inference. |
| `reta/learning/warmup.py`, `rl_train.py`, `inference.py` | Run encoder warm-up, policy refinement, and deterministic evaluation. |
| `configs/`, `examples/`, `tests/` | Hold the default experiment, a minimal valid template, and contract/regression tests. |
| `results/`, `logs/` | Store the paper-wide structured results and synchronized result log; also receive local evaluation and runtime outputs. |

The implementation is consolidated into three package domains: `data`,
`knowledge`, and `learning`. Generated datasets and pools live under `data/`;
model checkpoints live under `checkpoints/`.


## Run

### Quickstart
To run a small end-to-end check:
- Validate the tiny template artifact in `examples/tiny_templates.jsonl`.
- Run the dependency-light smoke test:
```bash
python tests/smoke_test.py
```
- Create tiny data/resources/concepts.csv, inventory.csv, support_edges.csv.
- Run Knowledge Pool Construction to generate knowledge/templates.jsonl.
- Run warm-up with a small processed dataset or a small subset of your data.
- Run inference with --max_patients 10.

### Installation

```bash
git clone https://github.com/ChenC2002/ReTA.git
cd ReTA
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Validate the bundled knowledge-pool specification, the minimal pool example,
and the repository test suite without clinical data:

```bash
reta validate-pool-release
reta validate-template-pool \
  --templates_jsonl examples/tiny_templates.jsonl \
  --expected_dim 4
python -m unittest discover -s tests -p 'test*.py'
```

### Results and log

The paper's reported numeric results are versioned in two synchronized
formats:

- `results/paper_results.json` is the canonical structured result set. It
  includes the paper title, protocol, dataset statistics, hyperparameters,
  compute profile, and records spanning the main tasks, transfer,
  ablations, robustness, efficiency, policy behavior, knowledge diagnostics,
  sensitivity, calibration, training stability, and reward analysis.
- `logs/paper_results.jsonl` is the line-oriented result log. It starts
  with manifest and protocol events, contains one event for every canonical
  result record, and ends with a completion event.

Record IDs are identical across both files, making the JSON convenient for
analysis and the JSONL convenient for streaming or indexing.

## Pipeline

The runnable path below implements the paper's main multi-label CCS diagnosis
task. The result and log files above also cover the paper's mortality,
readmission, transfer, and analysis results.

### 1. Prepare trajectories

Create `data/raw/diagnoses.csv` with one diagnosis code per row. `timestamp`
must identify the admission time used to order and combine visits:

- `patient_id`
- `timestamp`
- `icd_code`

For a single ICD version, provide:

- `data/resources/icd_to_ccs.csv` with `icd,ccs`
- `data/resources/ccs_hierarchy.csv` with `child,parent`

```bash
python -m reta.data.preprocess \
  --events data/raw/diagnoses.csv \
  --icd2ccs data/resources/icd_to_ccs.csv \
  --ccs_hierarchy data/resources/ccs_hierarchy.csv \
  --out_dir data/processed \
  --h_anc 2
```

Mixed ICD-9/10 data must include a version column in both the event and mapping
files. Add `--icd_version_col icd_version` and
`--mapping_icd_version_col icd_version`; version-qualified codes remain
distinct even when their text is identical.

Preprocessing writes `data/processed/processed.pt` and a deterministic,
patient-disjoint `data/processed/splits.json` (`train`, `val`, and `test`) for
local runs. The structured paper results retain the paper's fixed temporal
70/10/20 protocol ordered by discharge time.

### 2. Build the knowledge pool

Use the prompts and schemas in
[`reta/knowledge/releases/pool_v1`](reta/knowledge/releases/pool_v1) to create
one structured response per concept, in the same order as `concepts.csv`.
Build the validated artifacts:

```bash
python -m reta.knowledge.distill \
  --concepts_csv data/resources/concepts.csv \
  --responses_jsonl data/resources/distillation_responses.jsonl \
  --model_family GPT-4o \
  --out_jsonl data/knowledge/artifacts.jsonl
```

Ground cascade entities and retain only directly supported PrimeKG links or
CCS ancestor/descendant paths within two levels:

```bash
python -m reta.knowledge.filtering \
  --artifacts-jsonl data/knowledge/artifacts.jsonl \
  --inventory-csv data/resources/inventory.csv \
  --embedding-candidates data/resources/clinicalbert_top1.csv \
  --primekg-edges-csv data/resources/primekg_edges.csv \
  --ccs-edges-csv data/resources/ccs_hierarchy.csv \
  --out-grounded-jsonl data/knowledge/grounded.jsonl \
  --out-audit-jsonl results/filtering_audit.jsonl \
  --out-summary-json results/filtering_summary.json
```

The embedding-candidate CSV is precomputed; filtering performs no model
download. Cluster the grounded templates with the pinned ClinicalBERT revision
and validate the resulting retrieval pool:

```bash
python -m reta.knowledge.clustering \
  --grounded_jsonl data/knowledge/grounded.jsonl \
  --out_jsonl data/knowledge/templates.jsonl \
  --tau 0.16 \
  --projection_dim 256

reta validate-template-pool \
  --templates_jsonl data/knowledge/templates.jsonl \
  --expected_dim 256

reta build-entity-map \
  --processed_path data/processed/processed.pt \
  --templates_jsonl data/knowledge/templates.jsonl \
  --out data/resources/entity_to_token.json
```

The generated entity map gives every grounded PrimeKG or UMLS node a stable,
non-overlapping token ID so Hard Import retains the external subgraph.

### 3. Train

The default config uses only the patient-level `train` split. Stage 1 warms up
the encoder; Stage 2 trains the masked augmentation policy and refines the
encoder on the selected trajectories.

```bash
python -m reta.learning.warmup --config configs/reta.yaml
python -m reta.learning.rl_train \
  --config configs/reta.yaml \
  --warmup_ckpt checkpoints/warmup.pt
```

### 4. Evaluate

Inference derives model settings from the checkpoint, verifies the dataset,
split, vocabulary, entity mapping, and template-pool fingerprints, and uses
deterministic policy actions by default.

```bash
python -m reta.learning.inference \
  --checkpoint checkpoints/rl_iter50.pt \
  --processed_path data/processed/processed.pt \
  --templates_jsonl data/knowledge/templates.jsonl \
  --split_json data/processed/splits.json \
  --split test \
  --out results/inference.json
```

The default inference result contains aggregate metrics only. Optional
per-sample action metadata uses run-local patient indices rather than source
patient identifiers.

### Outputs

| Genre | Paths |
| --- | --- |
| Paper results | `results/paper_results.json` |
| Paper result log | `logs/paper_results.jsonl` |
| Knowledge build | `data/knowledge/{artifacts,grounded,templates}.jsonl` |
| Training | `checkpoints/warmup.pt`, `checkpoints/rl_iter*.pt` |
| Local evaluation | `results/inference.json`, `results/filtering_{summary,audit}.*` |
| Local runtime logs | `logs/warmup.log`, `logs/rl_train.log`, `logs/inference.log` |

The paper result and result-log files are versioned. Other generated
clinical-data derivatives, checkpoints, results, and runtime logs are ignored
by Git. PyTorch serialization is used for `processed.pt`, so load only a
locally generated or otherwise trusted processed dataset. Checkpoints use
restricted tensor-only loading and are bound to their exact runtime contracts.

