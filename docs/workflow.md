# Core implementation notes

The [README](../README.md) contains the input tables and end-to-end commands.
The implementation covers the next-visit diagnosis framework; it does not add
appendix ablations, clinical audits, binary-outcome pipelines, or cross-dataset
experiment runners.

## Model and retrieval

- Frozen ClinicalBERT descriptions are cached once. A shared trainable linear
  `W_code` supplies both node features and code-level retrieval vectors.
  Its initialization uses the projection fitted to external template centroids.
- Retrieval evaluates Eq. 1 exactly over the small template matrix. Stage 2 uses
  a GRU history/current state with fixed `W_ret = [I/2, I/2]`; Stage 1 uses
  code-only retrieval before a policy state encoder is trained. These are explicit
  simplifications of the basic framework, not a learned retrieval objective.
- Soft Import adds the projected template offset to observed code features;
  Hard Import grafts connected template structure; Skip leaves the input alone.
  PyG handles GAT layers and graph batching; PyTorch handles attention and GRUs.
- REINFORCE uses discounted returns minus the previous running-mean baseline.
  Stage 2 alternates policy updates with supervised encoder updates. Encoder
  outputs and past-visit memories are detached during policy rollouts; gradients
  do not pass through discrete retrieval or across the full patient trajectory.

`model.concept_features_path: null` retains lookup embeddings for synthetic
fixtures and older checkpoints. The supplied training config enables text
features. Cached features are embedded in new checkpoints, so inference does
not need the original feature-cache file. Old lookup checkpoints remain readable
but must be retrained to use the new text representation and policy update.

## Visits and evaluation

Use `--visit_id_col` for admission-level visits. The generic preprocessor emits
patient-disjoint splits; it does **not** reproduce the latest paper's
admission-level discharge-time split protocol. Full cohort reproduction requires
the original data and split artifacts. The selected Micro-F1 threshold is fitted
only on validation patients and stored with the selected checkpoint.

## Resume and independent seeds

```bash
python -m reta.learning.warmup --config configs/reta.yaml --resume checkpoints/warmup.pt
python -m reta.learning.rl_train --config configs/reta.yaml --resume checkpoints/rl_iter10.pt
```

Keep the input configuration unchanged; increasing the total epochs/iterations
and changing output directories is supported. Recovery is at epoch/iteration
boundaries and requires checkpoints with optimizer and RNG state.

For independent seeds on an existing Slurm installation:

```bash
RETA_CONFIG=configs/reta.yaml sbatch -p YOUR_PARTITION scripts/train-slurm.sh
```

Each job runs on one device, saves its own outputs, and selects on validation.
