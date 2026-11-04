# Import What You Need: Learning When and How to Augment EHR Graphs with External Knowledge

![Overview of ReTA: offline knowledge pool, visit-level augmentation policy, and decoupled encoder](assets/reta_overview.png)

ReTA (**Re**inforcement learning-based dynamic **T**opology **A**ugmentation)
learns whether, what, and how to import external knowledge at each visit in a
patient trajectory. Instead of expanding every EHR graph with the same
knowledge, ReTA selects a retrieved template and applies **Soft Import**,
**Hard Import**, or **Skip** according to the evolving patient state.

## Method Map

The implementation follows the same sequence as the paper method:

| Paper stage                         | Implementation                                               | Responsibility                                               |
| ----------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Visit Graph Construction (§2.1)** | [`reta/data/`](reta/data)                                    | Normalize ICD codes, map ICD to CCS, group admission-level visits, and construct graphs |
| **Knowledge Pool (§2.2)**           | [`distill.py`](reta/knowledge/distill.py), [`filtering.py`](reta/knowledge/filtering.py), [`clustering.py`](reta/knowledge/clustering.py), [`features.py`](reta/knowledge/features.py), [`pool.py`](reta/knowledge/pool.py) | Ground and cluster templates; cache frozen ClinicalBERT features; retrieve with a shared trainable code projection |
| **Augmentation Policy (§2.3)**      | [`policy.py`](reta/learning/policy.py), [`runtime.py`](reta/learning/runtime.py), [`rl_train.py`](reta/learning/rl_train.py) | Build policy states, select Soft/Hard/Skip actions, compute paired rewards, and optimize with REINFORCE |
| **Decoupled Encoding (§2.4)**       | [`model.py`](reta/learning/model.py)                         | Encode semantic and structural views, apply augmentation, fuse both channels, and predict CCS categories |
| **Two-Stage Training (§2.5)**       | [`warmup.py`](reta/learning/warmup.py), [`rl_train.py`](reta/learning/rl_train.py), [`reta.yaml`](configs/reta.yaml) | Warm up the encoder, then refine the policy and live encoder |
| **Evaluation (§3)**                 | [`select_checkpoint.py`](reta/learning/select_checkpoint.py), [`inference.py`](reta/learning/inference.py) | Select by validation macro-AUPRC, tune Micro-F1 threshold on validation, and evaluate the held-out set |

The core task is next-visit diagnosis prediction. Implementation choices and
optional training commands are in [workflow notes](docs/workflow.md).

## Quick Start

### Install

Use Python 3.10–3.13 in a virtual environment:

```bash
git clone https://github.com/ChenC2002/ReTA.git
cd ReTA
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

[Setup notes](docs/setup.md) cover pinned dependency profiles and package checks.

### Run without clinical data

```bash
python examples/run_toy.py
```

This runs warm-up → policy training → validation selection → test inference
on example trajectories and cached features. Outputs go to `runs/toy`;
no model download is needed. Its scores are **not clinical results**. Use
`--out runs/toy-2` for a new run.

```bash
python -m unittest discover -s tests -p 'test*.py'
```

## Required Inputs

Full runs require credentialed MIMIC-III or MIMIC-IV diagnosis data, ICD-to-CCS
mappings, a CCS hierarchy, PrimeKG or UMLS-derived resources, and externally
generated concept-level distillation responses.

### EHR Files

| Input                   | Required format                                              |
| ----------------------- | ------------------------------------------------------------ |
| Diagnosis events        | CSV with `patient_id,visit_id,timestamp,icd_code`; add `icd_version` for mixed ICD-9/10 data |
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
| Canonical descriptions | JSON object mapping every ICD/CCS/KG ID in the token vocabulary to its canonical description |

Both hierarchy readers use the named `parent` and `child` columns, so column
order does not matter. Preserve the parent/child direction and compatible CCS
identifiers; preprocessing additionally requires a single-parent hierarchy.

## Workflow

### Step 1: Preprocess

For MIMIC exports, use the admission ID as `visit_id` and admission time as
`timestamp`. All diagnoses from the same admission form one visit, including
stays spanning multiple days.

```bash
python -m reta.data.preprocess \
  --events data/raw/diagnoses.csv \
  --visit_id_col visit_id \
  --icd2ccs data/resources/icd_to_ccs.csv \
  --ccs_hierarchy data/resources/ccs_hierarchy.csv \
  --out_dir data/processed \
  --h_anc 2
```

For mixed ICD-9/10 input, include an `icd_version` column in both files and
append `--icd_version_col icd_version` and
`--mapping_icd_version_col icd_version` to the command.

The command writes `processed.pt` and a patient-disjoint `splits.json`.
Without `--visit_id_col`, legacy event data uses 24-hour windows.

### Step 2: Build the Pool

Generate one concept-level response per row using the [prompt and schema](reta/knowledge/releases/pool_v1).
The following command validates supplied responses; it does not call an LLM.

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

Clustering uses pinned Bio_ClinicalBERT features and saves both the pool and its
projection (`templates.jsonl.projection.pt`):

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

Cache canonical-description features using the **same** encoder and projection:

```bash
python -m reta.knowledge.features \
  --processed_path data/processed/processed.pt \
  --entity_to_token_json data/resources/entity_to_token.json \
  --descriptions_json data/resources/descriptions.json \
  --templates_jsonl data/knowledge/templates.jsonl \
  --projection data/knowledge/templates.jsonl.projection.pt \
  --out data/knowledge/concept_features.pt
```

For example, `descriptions.json` contains entries such as
`{"ICD:25000": "Diabetes mellitus without complication", "CCS:49": "Diabetes mellitus without complication"}`.
Provide descriptions for all vocabulary tokens, including external KG nodes.
Training freezes these cached features and learns the shared `W_code` projection.

### Step 3: Train

Set input paths in [`configs/reta.yaml`](configs/reta.yaml), then run:

```bash
python -m reta.learning.warmup --config configs/reta.yaml
python -m reta.learning.rl_train \
  --config configs/reta.yaml \
  --warmup_ckpt checkpoints/warmup.pt
```

Stage 1 learns the encoder with stochastic Soft/Hard exposure. Stage 2 trains
the Soft/Hard/Skip policy with REINFORCE and refines the encoder on selected
augmentations. Paired rewards compare raw and edited inputs under the same live
encoder with dropout disabled; the frozen Stage-1 encoder supplies uncertainty.
Both stages save optimizer and RNG state for [resuming](docs/workflow.md).

### Step 4: Select on validation, then evaluate

Select among completed Stage-2 checkpoints using the validation split only:

```bash
python -m reta.learning.select_checkpoint --config configs/reta.yaml \
  --checkpoints checkpoints/rl_iter*.pt --out checkpoints/best.pt \
  --report results/selection.json
```

Selection uses **macro average precision** and saves a global Micro-F1 threshold
chosen on validation predictions. Test inference reuses both the selected model
and that threshold:

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

Inference reports `AUPRC_macro`, `MicroF1`, and `Acc@20` on `[0, 1]`, with
micro average precision retained as an auxiliary metric. Macro-AUPRC uses
scikit-learn average precision, including zero for labels without positives.
Unselected checkpoints use threshold 0.5. Add `--max_patients 10` for a small run.

## Outputs

The existing paper-result files are archived references from the previous
manuscript, not outputs from this implementation or results for the latest PDF.

| Output                                       | Path                                                         |
| -------------------------------------------- | ------------------------------------------------------------ |
| Archived paper reference results                      | `results/paper_results.json`, `logs/paper_results.jsonl`     |
| Distilled, grounded, and clustered knowledge | `data/knowledge/{artifacts,grounded,templates}.jsonl`        |
| Filtering diagnostics                        | `results/filtering_audit.jsonl`, `results/filtering_summary.json` |
| Frozen text features and pool projection | `data/knowledge/concept_features.pt`, `data/knowledge/templates.jsonl.projection.pt` |
| Training checkpoints                         | `checkpoints/warmup.pt`, `checkpoints/rl_iter*.pt`, `checkpoints/best.pt`           |
| Validation checkpoint selection             | `results/selection.json`                                   |
| Local diagnosis evaluation                   | `results/inference.json`                                     |
| Runtime logs                                 | `logs/warmup.log`, `logs/rl_train.log`, `logs/inference.log` |

