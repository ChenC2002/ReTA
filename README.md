# Import What You Need: Learning When and How to Augment EHR Graphs with External Knowledge

![ReTA: knowledge retrieval, visit-level import policy, and decoupled graph encoding](assets/reta_overview.png)

ReTA (**Re**inforcement learning-based dynamic **T**opology **A**ugmentation)
learns whether, what, and how to import external knowledge at each visit.
It retrieves relevant templates, chooses **Soft Import**, **Hard Import**, or
**Skip**, and predicts the next visit’s diagnosis categories from the evolving
patient history.

[Interactive Demo](https://chenc2002.github.io/ReTA/demo/) · [Downloads](#downloads) · [Method Map](#method-map) · [Quick Start](#quick-start) · [Data](#data) · [Workflow](#workflow)

## Demo

[**Open the interactive demo**](https://chenc2002.github.io/ReTA/demo/)

[![ReTA diagnosis explorer](assets/reta_demo.png)](https://chenc2002.github.io/ReTA/demo/)

Explore diabetes and kidney disease, heart failure, and respiratory examples.
Switch visits, compare scores before and after knowledge import, and inspect the
selected template and graph. The online viewer replays actual model outputs for
three example histories. The local demo accepts edited inputs and runs the same
prediction code used in evaluation.

The released model was trained on a **7,494-patient MIMIC-III cohort** using
frozen ClinicalBERT concept features and 196 CCS ontology templates. Four
configurations were compared on validation; the selected model uses square-root
positive-class weights. Held-out evaluation covers **1,499 patients and 2,518
next-visit predictions**, across 283 diagnosis labels.

| Model | Macro-AUPRC | Micro-F1 | Acc@20 |
| --- | ---: | ---: | ---: |
| Diagnosis frequency | 4.32% | 33.53% | 44.56% |
| Linear history model | **21.70%** | **48.88%** | **63.46%** |
| ReTA encoder | 18.64% | 43.04% | 58.16% |
| ReTA adaptive knowledge | 18.82% | 44.06% | 59.05% |

These are results of the released demo experiment: a patient-disjoint split,
seed 42, and validation-selected checkpoint and F1 threshold. The editable
examples illustrate inference; they are not test cases with known future labels.

### Downloads

| Package | Size | Contents |
| --- | ---: | --- |
| [Offline viewer](https://github.com/ChenC2002/ReTA/releases/download/V2.0/reta-demo-offline.zip) | 15.3 KB | Open `index.html`; includes all three recorded examples and evaluation |
| [Complete model package](https://github.com/ChenC2002/ReTA/releases/download/V2.0/reta-demo-model.zip) | 129.5 MB | Full checkpoint, vocabulary, knowledge pool, mappings, descriptions, configuration, and reference predictions |
| [SHA-256 checksums](https://github.com/ChenC2002/ReTA/releases/download/V2.0/SHA256SUMS) | <1 KB | Verify the downloaded archives |

The model archive expands to approximately **143 MB**. Its **139.9 MB checkpoint**
contains the encoder, predictor, policy, state encoder, frozen baseline models,
optimizer states, and random-number states. Cached concept features are already
inside the checkpoint. No MIMIC patient records or patient split lists are included.

## Method Map

| Paper stage | Code | What it does |
| --- | --- | --- |
| **Visit graphs** | [`reta/data/`](reta/data) | Group admissions, map ICD to CCS, and construct visit graphs |
| **Knowledge pool** | [`reta/knowledge/`](reta/knowledge) | Validate, ground, cluster, and retrieve concept-level templates |
| **Import policy** | [`policy.py`](reta/learning/policy.py), [`rl_train.py`](reta/learning/rl_train.py) | Select Soft/Hard/Skip actions and learn from paired rewards |
| **Decoupled encoding** | [`model.py`](reta/learning/model.py) | Fuse semantic and structural representations with causal visit memory |
| **Two-stage training** | [`warmup.py`](reta/learning/warmup.py), [`rl_train.py`](reta/learning/rl_train.py) | Warm up the encoder, then refine the policy and predictor |
| **Evaluation and demo** | [`inference.py`](reta/learning/inference.py), [`prediction.py`](reta/learning/prediction.py) | Share prediction logic between held-out evaluation and interactive inference |

## Quick Start

Use Python 3.10–3.13:

```bash
git clone https://github.com/ChenC2002/ReTA.git
cd ReTA
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ".[demo]"
reta demo
```

`reta demo` opens the online viewer. Use `reta demo --no-open` to print its URL.
To edit histories and run the trained model locally, download and extract the
[model package](https://github.com/ChenC2002/ReTA/releases/download/V2.0/reta-demo-model.zip), then run:

```bash
reta demo --bundle /path/to/reta-demo-model --verify
reta demo --bundle /path/to/reta-demo-model
```

Open `http://127.0.0.1:8765`. CPU inference is sufficient; no dataset access or
ClinicalBERT download is needed. Bundle checksums are checked before loading.
`--verify` reproduces all three examples, including scores, actions, and graph structure.

## Data

Training uses credentialed clinical data and prepared knowledge resources:

| Input | Format |
| --- | --- |
| Diagnosis events | CSV: `patient_id,visit_id,timestamp,icd_code` |
| ICD-to-CCS mapping | CSV: `icd,ccs`; add `icd_version` for mixed ICD-9/10 data |
| CCS hierarchy | CSV: `child,parent` |
| Knowledge templates | Validated `templates.jsonl` with concept definitions and grounded subgraphs |
| External entity mapping | JSON: canonical concept ID → token ID |
| Concept descriptions/features | Canonical description JSON and cached text features |

The [preparation guide](docs/pipeline.md) contains the complete input tables,
knowledge-pool construction commands, and output paths. Keep clinical records
on a host with authorized data access.

## Workflow

Set the paths and hyperparameters in [`configs/reta.yaml`](configs/reta.yaml).

### 1. Train

```bash
python -m reta.learning.warmup --config configs/reta.yaml
python -m reta.learning.rl_train --config configs/reta.yaml \
  --warmup_ckpt checkpoints/warmup.pt
```

Warm-up exposes the encoder to stochastic knowledge imports. Stage 2 learns
Soft/Hard/Skip decisions through REINFORCE while refining the encoder. PyTorch
and PyTorch Geometric provide optimization, attention, recurrent state, and graph batching.

### 2. Select and evaluate

```bash
python -m reta.learning.select_checkpoint --config configs/reta.yaml \
  --checkpoints checkpoints/rl_iter*.pt --out checkpoints/best.pt \
  --report results/selection.json

python -m reta.learning.inference --checkpoint checkpoints/best.pt \
  --processed_path data/processed/processed.pt \
  --templates_jsonl data/knowledge/templates.jsonl \
  --entity_to_token_json data/resources/entity_to_token.json \
  --split_json data/processed/splits.json --split test \
  --out results/inference.json
```

Checkpoint selection uses validation macro average precision; F1 threshold
selection also uses validation only. Test inference reuses both choices.
[Implementation notes](docs/workflow.md) cover retrieval, class weighting, resume,
and independent seeds. [Setup and tests](docs/setup.md) cover dependency profiles
and package verification. [Demo packaging](docs/demo.md) explains the release files.
