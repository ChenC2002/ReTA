# ReTA: Dynamic Topology Augmentation with Budget-Aware Knowledge Graph Injection for Healthcare Prediction
## Overview
ReTA performs next-visit diagnosis prediction from longitudinal EHR diagnosis-code trajectories. Each visit graph contains observed ICD codes and CCS ancestors up to `h` hierarchy levels, with untyped undirected message passing. Targets are next-visit CCS categories.

1. Offline Knowledge Pool
   - Each ICD/CCS concept is distilled into a one-sentence Definition for Soft Import and an adaptive Clinical Cascade for Hard Import.
   - Cascade length follows KG neighborhood density: sparse concepts receive longer cascades, dense concepts shorter cascades.
   - Mentions are grounded to ICD, CCS, and PrimeKG identifiers by exact match or embedding similarity, then filtered by external KG or CCS-hierarchy support.
   - Grounded artifacts are embedded with ClinicalBERT, clustered under cosine distance, projected to the model dimension, and stored as reusable templates.

2. Visit-Level Augmentation Policy
   - At each visit, ReTA retrieves Top-K templates with history-aware scoring:
     `Score(V_t, s_t, p_k) = (1-alpha) max_i cos(e_ci, p_k) + alpha cos(s_t, p_k)`.
   - The policy state contains GRU-compressed patient history, the current base visit embedding, base-prediction uncertainty, and running template utilities.
   - The action space is `P_sub(t) x {Soft, Hard} union {Skip}`, giving `2K+1` actions under a one-action-per-visit budget.
   - Soft Import adds a template-derived feature offset to observed visit codes.
   - Hard Import grafts the selected compact KG subgraph onto the visit graph.
   - Skip leaves the visit unaugmented when augmentation is unnecessary.

3. Decoupled Encoding
   - The semantic channel applies self-attention to observed visit codes and their post-Soft features.
   - The structure channel applies GAT message passing over the full graph, including Hard Import nodes, and attends over past visit embeddings.
   - Adaptive gates fuse within-visit/across-visit structure signals and semantic/structural views. The final visit embedding sums over observed visit codes only.
## Structure
```text
data/             EHR preprocessing, CCS ontology utilities, transition dataset
knowledge/        Offline distillation, grounding, clustering, template pool validation
model/            Visit graph construction, decoupled encoder, prediction head
policy/           Soft/Hard/Skip actions, policy state, paired reward, REINFORCE
train/            Stage 1 warm-up and Stage 2 policy refinement
inference/        Sequential dynamic augmentation and evaluation
utils/            Metrics, graph utilities, logging
configs/          Default ReTA hyperparameters
examples/         Tiny template-pool artifact
tests/            Lightweight smoke checks
reta/             Command-line helpers and package entrypoint
```

## Method And Code Map

| Method component | Code path | Role |
| --- | --- | --- |
| Visit construction | `data.preprocess`, `data.ontology` | Aggregates diagnoses into 24-hour visits, maps ICD to CCS, and builds visit graphs with CCS ancestors. |
| Template artifact | `knowledge.templates` | Defines distilled, grounded, and clustered template contracts plus template-pool validation. |
| Grounding/filtering | `knowledge.grounding` | Maps cascade mentions to biomedical identifiers and keeps externally supported links. |
| Template clustering | `knowledge.clustering` | Embeds definition+cascade text, clusters redundant artifacts, and projects template vectors into `R^d`. |
| History-aware retrieval | `knowledge.pool` | Retrieves Top-K templates from current visit codes and trajectory state. |
| Soft/Hard/Skip actions | `policy.action` | Encodes the `2K+1` action space. |
| Policy state | `policy.state` | Combines GRU history, current base visit embedding, uncertainty, and template utilities. |
| Paired reward | `policy.reward` | Computes `L_raw - L_edit` with Hard Import cost and zero reward for Skip. |
| Policy optimization | `policy.reinforce` | Computes REINFORCE updates with discounted returns and a running baseline. |
| Decoupled encoder | `model.encoder`, `model.semantic_encoder`, `model.structure_encoder`, `model.fusion` | Separates feature-only semantic encoding from full-graph structural message passing. |
| Stage 1 training | `train.warmup` | Walks patient trajectories in order and trains encoder/predictor under stochastic Soft/Hard exposure. |
| Stage 2 training | `train.rl_train` | Learns the visit-level policy and refines the encoder. |
| Inference | `inference.infer` | Runs sequential retrieval, policy selection, augmentation, prediction, and metric reporting. |

## Quickstart
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


## Running
### Installation
```bash
pip install -r requirements.txt
```
### Step 1: Data Preparation
This code expects a patient-level sequence of visits derived from an event-level diagnosis table (one row per diagnosis event). Visits are aggregated with a 24-hour window and duplicates removed.
1. Prepare an event-level diagnosis file:
Create data/raw/diagnoses.csv with at least:
   - patient_id
   - timestamp
   - icd_code (ICD-9/10 string)
2. Prepare ICD→CCS mapping and CCS hierarchy:
Create two CSV resources:
   - data/resources/icd_to_ccs.csv with columns: icd, ccs
   - data/resources/ccs_hierarchy.csv with columns: child, parent
3. Run preprocessing
This produces data/processed/processed.pt containing trajectories and per-visit graphs.
```bash
python -m reta.data.preprocess \
  --events data/raw/diagnoses.csv \
  --icd2ccs data/resources/icd_to_ccs.csv \
  --ccs_hierarchy data/resources/ccs_hierarchy.csv \
  --out_dir data/processed \
  --h_anc 2
```
### Step 2: Knowledge Pool Construction
Knowledge pool construction is performed offline and does not use any patient data.
1. Distill artifacts (Definition + Cascade)
```bash
python -m reta.knowledge.distill \
  --concepts_csv data/resources/concepts.csv \
  --out_jsonl knowledge/artifacts.jsonl
```
`concepts.csv` may include an optional `density` column with `sparse`, `moderate`, or `dense`.
2. Ground and filter
```bash
python -m reta.knowledge.grounding \
  --artifacts_jsonl knowledge/artifacts.jsonl \
  --inventory_csv data/resources/inventory.csv \
  --support_edges_csv data/resources/support_edges.csv \
  --out_jsonl knowledge/grounded.jsonl
```
3. Cluster into templates (final pool)
```bash
python -m reta.knowledge.clustering \
  --grounded_jsonl knowledge/grounded.jsonl \
  --out_jsonl knowledge/templates.jsonl \
  --tau 0.15 \
  --projection_dim 256
```
The final online retrieval pool is:
```bash
knowledge/templates.jsonl
```

Validate a generated template pool before training:
```bash
python -m reta.cli validate-template-pool \
  --templates_jsonl knowledge/templates.jsonl \
  --expected_dim 256
```

Template JSONL records follow this shape:
```json
{
  "template_id": 0,
  "vector": [0.12, 0.04],
  "medoid_idx": 0,
  "medoid": {
    "root_concept_id": "CCS:49",
    "root_name": "Diabetes mellitus",
    "definition": "One-sentence pathology summary.",
    "cascade_entities": [
      {"entity_id": "UMLS:C0035309", "name": "Retinopathy", "source": "PrimeKG", "score": 0.94}
    ],
    "subgraph_nodes": ["CCS:49", "UMLS:C0035309"],
    "subgraph_edges": [["CCS:49", "UMLS:C0035309"]],
    "meta": {"density": "moderate"}
  },
  "member_indices": [0]
}
```

For Hard Import, `ICD:` and `CCS:` nodes are mapped from `processed.pt`. External KG nodes such as `UMLS:` or `PrimeKG:` identifiers require an optional JSON mapping passed through `knowledge.entity_to_token_json` or `--entity_to_token_json`; nodes without a token mapping are skipped.

### Step 3: Model Training
Training follows a two-stage curriculum. Stage 1 walks patient trajectories in temporal order and warms up the encoder with Bernoulli exposure to stochastic Soft/Hard imports. Stage 2 learns the augmentation policy with REINFORCE and a running-mean baseline, using paired rewards while continuing supervised encoder refinement.

Inspect the configured training stages:
```bash
python -m reta.cli show-training-stages
```

#### Stage 1: Encoder Warm-up
```bash
python -m reta.train.warmup --config configs/reta.yaml
```
Outputs:
```bash
checkpoints/warmup.pt
```
#### Stage 2: Policy learning (REINFORCE) + encoder refinement
```bash
python -m reta.train.rl_train \
  --config configs/reta.yaml \
  --warmup_ckpt checkpoints/warmup.pt
```
Outputs:
```bash
checkpoints/rl_iter*.pt
```
### Step 4: Inference and Evaluation
Run inference with dynamic augmentation using a trained checkpoint:
```bash
python -m reta.inference.infer \
  --checkpoint checkpoints/rl_iter10.pt \
  --processed_path data/processed/processed.pt \
  --templates_jsonl knowledge/templates.jsonl \
  --split all \
  --device cuda \
  --deterministic \
  --out results/infer.json
```

