#!/usr/bin/env bash
#SBATCH --job-name=reta
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-4%2
# Submit from the repository root with the desired Python environment activated:
# RETA_CONFIG=configs/reta.yaml sbatch -p zyao scripts/train-slurm.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit using sbatch from the repository root}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export RETA_RUN_ROOT="${RETA_RUN_ROOT:-runs}"
: "${RETA_CONFIG:?Set RETA_CONFIG to a complete training configuration}"
run_config=$(python - <<'PY'
from dataclasses import asdict
import os
from pathlib import Path
import yaml
from reta.learning.runtime import load_config
cfg = load_config(os.environ['RETA_CONFIG'])
index = int(os.environ['SLURM_ARRAY_TASK_ID'])
cfg.train.seed += index
cfg.train.device = 'cuda'
run = Path(os.environ['RETA_RUN_ROOT']) / os.environ['SLURM_ARRAY_JOB_ID'] / f'seed-{cfg.train.seed}'
run.mkdir(parents=True, exist_ok=False)
cfg.outputs.checkpoints_dir = str(run / 'checkpoints')
cfg.outputs.logs_dir = str(run / 'logs')
path = run / 'config.yaml'
path.write_text(yaml.safe_dump(asdict(cfg)))
print(path)
PY
)
python -m reta.learning.warmup --config "$run_config"
python -m reta.learning.rl_train --config "$run_config"
