# ReTA: Dynamic Topology Augmentation with Budget-Aware Knowledge Graph Injection for Healthcare Prediction
## Overview
1. Refined Knowledge Pool (offline)
+ LLM-distilled medical knowledge (Definition + Clinical Cascade)
+ Grounded to biomedical KGs and clustered into reusable templates
2. Dynamic Augmentation Policy (online)
+ RL policy selects one action per visit
+ Action = (knowledge template, import mode)
   + Soft Import: semantic feature injection (low cost)
   + Hard Import: graph topology augmentation (high cost)
3. Decoupled Multi-GAT Encoder
+ Semantic channel (feature-only)
+ Structure channel (graph message passing)
+ Adaptive gating to fuse both views
## Structure
```text
reta/
├── data/            # EHR preprocessing and datasets
├── knowledge/       # Knowledge pool construction (offline)
├── model/           # Encoders and prediction heads
├── policy/          # RL policy, reward, PPO
├── train/           # Training pipelines (warm-up + RL)
├── inference/       # Inference-time augmentation & prediction
└── utils/           # Metrics, graph utilities, logging
```
## Running
### Installation
```bash
pip install -r requirements.txt
```
### Step 1: Data Preparation
This code expects a patient-level sequence of visits derived from an event-level diagnosis table (one row per diagnosis event). Visits are aggregated with a 24-hour window and duplicates removed.
1. Prepare an event-level diagnosis file:
Create data/raw/diagnoses.csv with at least:
+ patient_id
+ timestamp
+ icd_code (ICD-9/10 string)
2. Prepare ICD→CCS mapping and CCS hierarchy:
Create two CSV resources:
+ data/resources/icd_to_ccs.csv with columns: icd, ccs
+ data/resources/ccs_hierarchy.csv with columns: child, parent
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
  --tau 0.15
```
The final online retrieval pool is:
```bash
knowledge/templates.jsonl
```
### Step 3: Model Training
Training follows a two-stage curriculum: warm-up the encoder under stochastic augmentation exposure, then learn a PPO policy while refining the encoder.
#### Stage 1: Encoder Warm-up
```bash
python -m reta.train.warmup --config configs/reta.yaml
```
Outputs:
```bash
checkpoints/warmup.pt
```
#### Stage 2: Policy learning (PPO) + encoder refinement
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
## Quickstart
To verify the pipeline runs end-to-end without large datasets:
+ Create tiny data/resources/concepts.csv, inventory.csv, support_edges.csv (few rows).
+ Run Knowledge Pool Construction to generate knowledge/templates.jsonl.
+ Run warm-up with a small processed dataset or a small subset of your data.
+ Run inference with --max_patients 10.
This confirms your installation, imports, and module wiring.
## Disclaimer
This code is for research purposes only and is not intended for clinical use.
