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
## Repository Structure
'
reta/
├── data/            # EHR preprocessing and datasets
├── knowledge/       # Knowledge pool construction (offline)
├── model/           # Encoders and prediction heads
├── policy/          # RL policy, reward, PPO
├── train/           # Training pipelines (warm-up + RL)
├── inference/       # Inference-time augmentation & prediction
└── utils/           # Metrics, graph utilities, logging
'
