"""Epoch/iteration-boundary recovery using native PyTorch state dictionaries."""
from dataclasses import asdict
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def capture_rng():
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [name, keys.tolist(), pos, has_gauss, cached],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    if state["cuda"] and (
        not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count()
    ):
        raise ValueError("Resume requires the same visible CUDA device count.")
    random.setstate(state["python"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(payload, path):
    """Replace a checkpoint only after its complete serialization succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def training_state(stage, completed, optimizers, **extra):
    return dict(stage=stage, completed=completed, rng=capture_rng(),
                optimizers={name: opt.state_dict() for name, opt in optimizers.items()}, **extra)


def validate_resume(checkpoint, cfg, stage):
    state = checkpoint.get("training_state")
    if not isinstance(state, dict) or state.get("stage") != stage:
        raise ValueError(f"Checkpoint has no resumable {stage} training state.")
    completed = state.get("completed")
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 1:
        raise ValueError("Resume checkpoint must record a positive completed boundary.")
    budget = cfg.train.warmup_epochs if stage == "warmup" else cfg.train.rl_iters
    if budget <= completed:
        raise ValueError("Resume budget must exceed the completed boundary; increase epochs/iterations.")
    expected_optimizers = {"encoder"} if stage == "warmup" else {"encoder", "policy"}
    if not isinstance(state.get("optimizers"), dict) or not expected_optimizers <= state["optimizers"].keys():
        raise ValueError("Resume checkpoint is missing optimizer state.")
    if not isinstance(state.get("rng"), dict):
        raise ValueError("Resume checkpoint is missing RNG state.")
    # Permit extending a run and relocating outputs, not silently changing its method.
    old = checkpoint["cfg"]
    new = asdict(cfg)
    for section in ("data", "knowledge", "model", "train"):
        a, b = dict(old[section]), dict(new[section])
        if section == "train":
            for key in ("warmup_epochs", "rl_iters"):
                a.pop(key, None)
                b.pop(key, None)
        if a != b:
            raise ValueError(f"Resume config differs in {section}; use the original configuration.")
    return state
