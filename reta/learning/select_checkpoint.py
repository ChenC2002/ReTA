"""Choose a checkpoint using validation patients only; test evaluation is separate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .checkpointing import atomic_save
from .inference import run_inference
from .runtime import file_fingerprint, load_checkpoint, load_config


def select_checkpoint(checkpoints, cfg, output, report, metric="AUPRC_micro"):
    if metric not in {"AUPRC_micro", "MicroF1@0.5", "Acc@20"}:
        raise ValueError(f"Unsupported selection metric: {metric}")
    if not cfg.data.split_json:
        raise ValueError("Checkpoint selection requires a patient-disjoint split manifest.")
    paths = sorted({Path(path).resolve() for path in checkpoints})
    if not paths:
        raise ValueError("No candidate checkpoints supplied.")
    output, report = Path(output).resolve(), Path(report).resolve()
    protected = set(paths) | {
        Path(cfg.data.processed_path).resolve(), Path(cfg.data.split_json).resolve(),
        Path(cfg.knowledge.templates_jsonl).resolve(),
    }
    if cfg.knowledge.entity_to_token_json:
        protected.add(Path(cfg.knowledge.entity_to_token_json).resolve())
    if output in protected or report in protected or output == report:
        raise ValueError("Selection outputs must not overwrite inputs or each other.")

    # Validate every candidate before evaluating any; refuse training leakage.
    for path in paths:
        checkpoint = load_checkpoint(str(path))
        if checkpoint.get("contract", {}).get("training_split") != "train":
            raise ValueError("Selection requires checkpoints trained on the train split only.")
        del checkpoint

    records = []
    best_path, best_score = None, -math.inf
    for path in paths:
        digest = file_fingerprint(str(path))
        result = run_inference(
            checkpoint_path=str(path), processed_path=cfg.data.processed_path,
            templates_jsonl=cfg.knowledge.templates_jsonl,
            entity_to_token_json=cfg.knowledge.entity_to_token_json,
            split_json=cfg.data.split_json, split="val", device=cfg.train.device,
            deterministic=True,
        )
        score = float(result["metrics"][metric])
        if not math.isfinite(score):
            raise ValueError(f"Non-finite validation metric for {path.name}.")
        if digest != file_fingerprint(str(path)):
            raise ValueError("Candidate checkpoint changed during evaluation.")
        records.append({"checkpoint": str(path), "sha256": digest, "metrics": result["metrics"]})
        # Sorted-path order resolves equal scores reproducibly.
        if score > best_score:
            best_path, best_score = path, score

    selected = next(record for record in records if record["checkpoint"] == str(best_path))
    if file_fingerprint(str(best_path)) != selected["sha256"]:
        raise ValueError("Selected checkpoint changed during evaluation.")
    checkpoint = load_checkpoint(str(best_path))
    selection = {"split": "val", "metric": metric, "score": best_score,
                 "source_sha256": selected["sha256"], "source": str(best_path)}
    checkpoint["selection"] = selection
    atomic_save(checkpoint, output)
    summary = {"selection": selection, "output": str(output), "candidates": records}
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--out", default="checkpoints/best.pt")
    parser.add_argument("--report", default="results/selection.json")
    parser.add_argument("--metric", choices=["AUPRC_micro", "MicroF1@0.5", "Acc@20"], default="AUPRC_micro")
    args = parser.parse_args()
    result = select_checkpoint(args.checkpoints, load_config(args.config), args.out, args.report, args.metric)
    print(json.dumps(result["selection"], sort_keys=True))


if __name__ == "__main__":
    main()
