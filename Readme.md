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
├── knowledge/       # Knowledge pool construction
├── model/           # Encoders and prediction heads
├── policy/          # RL policy, reward, PPO
├── train/           # Training pipelines (warm-up + RL)
├── inference/       # Inference-time augmentation & prediction
└── utils/           # Metrics, graph utilities, logging
```
## Running
### Step 1: Data Preparation
This project uses MIMIC-III and MIMIC-IV for evaluation.
1. Place the raw data under:
```bash
data/raw/MIMICIII/data
data/raw/MIMICIV
```
2. Preprocess EHR data and construct visit graphs:
```bash
python data/preprocess.py \
  --dataset mimic-iv \
  --output_dir data/processed/
```
### Step 2: Knowledge Pool Construction
Knowledge pool construction is performed offline and does not use any patient data.
```bash
python knowledge/distill.py
python knowledge/grounding.py
python knowledge/clustering.py
```
The resulting knowledge templates will be saved under:
```bash
knowledge/pool/
```
### Step 3: Model Training
Training follows a two-stage curriculum.
#### Stage 1: Encoder Warm-up
```bash
python train/warmup.py \
  --config configs/reta.yaml
```
#### Stage 2: Policy Learning (PPO)
```bash
python train/rl_train.py \
  --config configs/reta.yaml
```
### Step 4: Inference and Evaluation
```bash
python inference/infer.py \
  --checkpoint checkpoints/reta_best.pt \
  --split test
```
## Disclaimer
This code is for research purposes only and is not intended for clinical use.
