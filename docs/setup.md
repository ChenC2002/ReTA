# Environment setup

Python 3.10–3.13 is supported. Install in an isolated virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip check
```

Package compatibility bounds are in `requirements-base.txt`. CI also tests the
original Python 3.11 pins in `requirements.txt` and the newer Python 3.13 pins in
`requirements-cuda.txt`. The latter is a dependency profile, not a requirement
to use a GPU. For GPU execution, install the appropriate PyTorch wheel from the
[official installer](https://pytorch.org/get-started/locally/) before installing ReTA.

## Tests and packaging

```bash
python -m unittest discover -s tests -p 'test*.py'
python -m pip install build
python -m build
```

Tests use example inputs and a tiny local Transformer; no model download is
required. Real pool/feature construction loads the pinned Bio_ClinicalBERT model
and needs either a download or a populated local model cache.
