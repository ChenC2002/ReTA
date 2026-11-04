# Core implementation notes

The [preparation guide](pipeline.md) contains the complete input tables and commands.
The implementation covers visit-level knowledge augmentation and next-visit
diagnosis prediction.

## Model and retrieval

- Frozen ClinicalBERT descriptions are cached once. A shared trainable linear
  `W_code` supplies both node features and code-level retrieval vectors.
  Its initialization uses the projection fitted to external template centroids.
- Retrieval evaluates Eq. 1 exactly over the small template matrix. Stage 2 uses
  a GRU history/current state with fixed `W_ret = [I/2, I/2]`; Stage 1 uses
  code-only retrieval before a policy state encoder is trained.
- For ontology pools, `knowledge.grounded_candidates_only: true` restricts roots
  to observed visit concepts; no match permits Skip only. Set `retrieval_alpha: 0`
  for code-only ranking within this set.
- Soft Import adds the projected template offset to observed code features;
  Hard Import grafts connected template structure; Skip leaves the input alone.
  PyG handles GAT layers and graph batching; PyTorch handles attention and GRUs.
- REINFORCE uses discounted returns minus the previous running-mean baseline.
  Stage 2 alternates policy updates with supervised encoder updates. Encoder
  outputs and past-visit memories are detached during policy rollouts. Each
  update follows the active policy or supervised loss.

`model.concept_features_path: null` retains lookup embeddings for example
fixtures and older checkpoints. The supplied training config enables text
features. Inference restores cached features directly from the checkpoint.
To use text representations, train with `model.concept_features_path` configured.

## Visits and evaluation

The supplied configuration batches independent patient graphs while retaining
separate causal histories (`train.graph_batching: true`). Optional
`train.positive_weight_power: 0.5` uses square-root class weights fitted on
training targets, capped at 20; the default `0` uses unweighted BCE. Choose
between these settings on validation before evaluating the test split.

Use `--visit_id_col` to group diagnoses by admission. Preprocessing writes
patient-disjoint training, validation, and test splits. Model selection and
Micro-F1 threshold tuning use the validation split; test inference reuses the
selected checkpoint and its saved threshold.

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
