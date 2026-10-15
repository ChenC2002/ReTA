# Environment setup

Use a separate virtual environment for each profile. `requirements.txt` preserves
the original Python 3.10/3.11 dependencies. `requirements-cuda.txt` pins the direct
dependencies tested with Python 3.13 on RTX PRO 6000; it is not required for CPU
use and is not a complete transitive lockfile. Package compatibility bounds live
in `requirements-base.txt`.

## CUDA / Blackwell

Choose a PyTorch CUDA wheel compatible with the GPU and NVIDIA driver using the
[official installer](https://pytorch.org/get-started/locally/). The tested CUDA
13.0 installation is:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-cuda.txt -e .
python -m pip check
RETA_TEST_DEVICE=cuda python -m unittest discover -s tests -p 'test_training*.py'
```

The original Torch 2.1.2 pin predates Blackwell support. Do not install it over
a working Blackwell environment. Each training process uses one GPU; the Slurm
launcher schedules independent seeds, not distributed training of one model.

## Tests and packaging

```bash
python -m unittest discover -s tests -p 'test*.py'
python -m pip install build
python -m build
```

CI checks both dependency profiles on CPU and verifies that the wheel includes
the pool release resources. The embedding test creates and loads a tiny local
Transformer, so it does not download a model. Production pool construction uses
the pinned Bio_ClinicalBERT revision and may need a download or local cache.
