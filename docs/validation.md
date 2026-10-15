# Validation and remaining scope

Validated on 2026-10-15 in a clean Python 3.13 environment using
`requirements-cuda.txt` and PyTorch 2.13.0+cu130:

- `pip check`: no broken requirements.
- Full CPU suite: 85 tests passed.
- CUDA training suite on RTX PRO 6000: 9 tests passed.
- Public pinned Bio_ClinicalBERT download/loading and embedding: passed on two
  generic phrases (finite 2 × 768 output; no patient data).
- Source distribution and wheel built; installed wheel passed pool-resource
  validation from outside the source checkout.

Coverage includes two-stage synthetic training, checkpoint recovery, mixed-action
batch outputs and gradients, validation-only checkpoint selection, held-out
inference, input guards, and loading/pooling a real tiny local Transformer.
CPU recovery checks exact tensor equality; CUDA checks allow numerical tolerance.
GitHub Actions runs the CPU suite and packaging checks for both dependency profiles.

The tests establish software behavior, not clinical quality or paper reproduction.
The following remain outside the validated pipeline:

- Complete frozen-pool construction and real-data preprocessing/training.
- Paper metrics, clinical audits, ablations and multi-seed performance measurements.
- Mortality/readmission tasks; the implemented task is next-visit diagnosis.
- DDP, AMP and batched causal rollouts. These are optional performance extensions,
  not prerequisites for the supported single-device pipeline.
