# Validation record

Review date: 2026-10-15 (user-facing date). Tests used synthetic fixtures, not MIMIC records. The timestamps inside the logs are the remote host's timestamps. Account-specific home and scratch prefixes are redacted as `$HOME` and `$SCRATCH`; test output is otherwise preserved.

## Results

| Check | Result | Evidence |
|---|---|---|
| Full CPU suite | 79 tests passed, no skips; 36.537 s reported by unittest | [CPU log](validation/cpu-tests.log) |
| CUDA training suite | 4 tests passed; 18.430 s reported by unittest | [CUDA log](validation/cuda-tests.log) |
| Python syntax | compileall passed | Local source compilation |
| Slurm launcher syntax | bash -n passed | Launcher not used for a full clinical training array |

CPU integration compares uninterrupted versus resumed Stage 1 and Stage 2 model tensors exactly. CUDA comparisons allow 1e-5 tolerance because scatter reductions need not be deterministic. Both paths include checkpoint-backed held-out synthetic inference. Mixed Soft/Hard/Skip batch tests compare outputs and gradients with separate graph executions at dropout zero. Recovery tests check weights-only deserialization, RNG restoration and preservation of the previous checkpoint after a failed save. The integration test checks that Skip raw/edited BCE agree during learning, which detects using the frozen model as the reward baseline after the live encoder changes.

## Environment and reproduction

GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB reported total memory. Allocated through front1 using `srun -p zyao --gres=gpu:1`, not on the login node. Ampere was inspected but not used because its three A6000 GPUs were mostly occupied.

Runtime: Python 3.13, PyTorch 2.13.0+cu130, PyG 2.7.0. Additional direct package versions are in `requirements-cuda.txt`. Validation reused an existing Python/Torch runtime read-only with a separate `PYTHONPATH` dependency directory at `~/scratch/reta-review-20261015/deps`; existing environments were not modified. This is a tested source execution environment, not a claim that a clean installation or all dependency combinations inside package bounds have been validated. The complete ClinicalBERT model download/embedding path was not executed.

Remote source: `~/scratch/reta-review-20261015/repo`. Test interpreter: `~/scratch/vapa-cuda-5QAMcQ/venv/bin/python`. From the remote repository directory:

```bash
PYTHONPATH=../deps OMP_NUM_THREADS=2 \
  ~/scratch/vapa-cuda-5QAMcQ/venv/bin/python -m unittest discover -s tests -p 'test*.py'
srun -p zyao --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=00:10:00 \
  env PYTHONPATH=../deps RETA_TEST_DEVICE=cuda OMP_NUM_THREADS=2 \
  ~/scratch/vapa-cuda-5QAMcQ/venv/bin/python -m unittest discover -s tests -p 'test_training*.py'
```

An initial CPU attempt ran before dependency installation completed and failed with missing packages; it was superseded by the complete successful run above. Framework deprecation warnings remain in the logs. No test result is described as a full model-quality or performance benchmark.

## Not established

- Reproduction of paper AUPRC, calibration, latency or five-seed confidence intervals.
- Full frozen pool availability and integrity, real-data preprocessing or complete ClinicalBERT generation.
- DDP, AMP, cross-host scaling or speedup from minibatching on clinical workloads.
- Legacy pinned environment regression on this host; it remains a separate reproduction profile.
- A full clinical seed-array run or validation-based model selection.

These require the authorized experiment input paths and a separately recorded run. No synthetic inference score is included as a clinical result.
