# Reproduce the demo experiment

The following workflow recomputes all five rows in the demo's Experiment view:
diagnosis frequency, diagnosis-history baseline, ReTA encoder, adaptive knowledge,
and history fusion. It requires authorized local access to MIMIC-III; patient
records and split membership stay on your machine.

## 1. Install and download

Run these commands from the cloned repository. The original environment used
Python 3.13.5 and PyTorch 2.6.0; the pinned experiment dependencies are in
[`requirements-experiment.txt`](../requirements-experiment.txt). Install the
appropriate PyTorch wheel for your machine, then:

```bash
python -m pip install -r requirements-experiment.txt -e .
curl -fL -O https://github.com/ChenC2002/ReTA/releases/download/V2.0/reta-demo-model.zip
unzip reta-demo-model.zip
```

Archive checksums are provided in the release's
[SHA256SUMS](https://github.com/ChenC2002/ReTA/releases/download/V2.0/SHA256SUMS).
Use the **complete demo model**, which contains the selected graph model, its
frozen Stage 1 model, and the history head. No additional model download is
needed for evaluation.

## 2. Rebuild the cohort

Supply the directory containing `ADMISSIONS.csv` and `DIAGNOSES_ICD.csv`;
`.csv.gz` files are also supported. Choose a new private output directory:

```bash
python -m reta.experiments.mimiciii prepare \
  --mimic-root /path/to/mimiciii \
  --bundle reta-demo-model --run runs/mimiciii
```

Preparation uses the released ICD/CCS mappings, retains patients with at least
two mapped admissions, excludes overlapping stays, groups diagnoses by admission,
and creates patient-disjoint splits with seed 42. It must reconstruct 7,494
patients: 5,246 training, 749 validation, and 1,499 test patients. The test split
contains 2,518 next-visit predictions over 283 diagnosis labels.

The complete cohort, visit order, graph inputs, targets, vocabulary and split
membership are checked against a content fingerprint in
[`experiments/mimiciii.json`](../experiments/mimiciii.json). This check is independent
of Torch serialization and fails on a different cohort. Only aggregate hashes
and counts are published; patient IDs remain local.

## 3. Recompute the complete table

```bash
python -m reta.experiments.mimiciii evaluate \
  --bundle reta-demo-model --run runs/mimiciii --device auto
```

The command runs the checkpoint over validation and test histories. Diagnosis
frequency is fitted on training targets, and each F1 threshold is selected on
validation only. Every patient starts with an empty causal history. Saved demo
metrics are read only after computation, for the final comparison.

Outputs are `runs/mimiciii/released-evaluation/metrics.csv` and `results.json`.
The JSON includes all five rows, validation thresholds, absolute differences and
`matches_release`. Evaluation exits with an error if any of the 20 metrics
differs by more than `0.00005` on the 0–1 scale. The tolerance accommodates
floating-point variation; it is not a criterion for selecting a model. To run
again, supply a new `--out` directory. CPU and CUDA inference are supported.

This command recomputes the entire experiment table. The separate
`reta demo --bundle reta-demo-model --verify` command checks the three illustrative
histories only.

## Train the released configuration from scratch

The original frozen concept features and **initial** projection are available
separately; trained projection weights are not used to initialize a fresh run.

```bash
curl -fL -O https://github.com/ChenC2002/ReTA/releases/download/V2.0/reta-experiment-resources.zip
unzip reta-experiment-resources.zip

python -m reta.experiments.mimiciii train \
  --bundle reta-demo-model --run runs/mimiciii \
  --features reta-experiment-resources/concept_features.pt --device cuda

python -m reta.experiments.mimiciii evaluate \
  --bundle reta-demo-model --run runs/mimiciii \
  --checkpoint runs/mimiciii/training/demo.pt --device cuda
```

The [experiment config](../configs/demo_mimiciii.yaml) uses square-root class
weights, observed-concept retrieval, 60 warm-up epochs and 30 policy iterations.
Warm-up checkpoints at epochs 10/20/30/45/60 and policy checkpoints every five
iterations are selected on validation macro average precision. The history
model uses `C=0.1`; its combination weight and F1 threshold are selected on
validation. The recipe runs the released configuration, rather than repeating
the earlier hyperparameter search.

Fresh training writes complete checkpoints and selection reports under
`runs/mimiciii/training/`; its independently computed table goes to
`retrained-evaluation/`. GPU reductions and library versions can change training
outcomes. Exact published-table verification uses the released-checkpoint workflow
above; newly trained results are recorded separately and never replace the demo
automatically.
