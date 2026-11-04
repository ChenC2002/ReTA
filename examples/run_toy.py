"""Run the core pipeline without clinical data or model downloads.

The trajectories, labels, and cached text features are example inputs. The resulting
metrics check software wiring only; they have no clinical interpretation.
"""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from reta.learning.inference import run_inference
from reta.learning.runtime import Config, file_fingerprint, mapping_fingerprint
from reta.learning.select_checkpoint import select_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/toy", help="New directory for example inputs and outputs")
    args = parser.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = Config()
    cfg.model.dim, cfg.model.attn_heads, cfg.model.gnn_layers = 4, 2, 1
    cfg.data.batch_size = 2
    cfg.train.device = "cpu"
    cfg.train.warmup_epochs, cfg.train.rl_iters, cfg.train.rollout_patients = 1, 2, 2
    cfg.knowledge.retrieval_K = 1
    cfg.data.processed_path = str(root / "processed.pt")
    cfg.data.split_json = str(root / "splits.json")
    cfg.knowledge.templates_jsonl = str(Path(__file__).resolve().with_name("tiny_templates.jsonl"))
    cfg.knowledge.entity_to_token_json = str(root / "entities.json")
    cfg.model.concept_features_path = str(root / "features.pt")
    cfg.outputs.checkpoints_dir = str(root / "checkpoints")
    cfg.outputs.logs_dir = str(root / "logs")
    base = {"ICD:25000": 0, "CCS:49": 1, "CCS:50": 2}
    external = {"UMLS:C0035309": 3, "UMLS:C0442874": 4}
    (root / "entities.json").write_text(json.dumps(external))
    (root / "splits.json").write_text(json.dumps({"train": ["a", "b"], "val": ["c"], "test": ["d"]}))
    trajectories = {
        pid: [dict(node_ids=[0, 1], edge_index=[[0, 1], [1, 0]], icd_tokens=[0],
                   ccs_ancestors=[1], label_ccs=[1 + t % 2] if t < 2 else None)
              for t in range(3)] for pid in "abcd"
    }
    torch.save({"trajectories": trajectories,
                "meta": {"icd_vocab_size": 1, "ccs_vocab_size": 2, "ccs_label_vocab_size": 2},
                "vocab": {"name_to_token": base}}, cfg.data.processed_path)
    torch.save({"features": torch.randn(5, 6), "projection": torch.eye(4, 6),
                "mapping_sha256": mapping_fingerprint({**base, **external}),
                "templates_sha256": file_fingerprint(cfg.knowledge.templates_jsonl)},
               cfg.model.concept_features_path)
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(asdict(cfg)))
    for module in ("warmup", "rl_train"):
        subprocess.run([sys.executable, "-m", f"reta.learning.{module}", "--config", str(config_path)], check=True)
    checkpoints = Path(cfg.outputs.checkpoints_dir)
    select_checkpoint(list(checkpoints.glob("rl_iter*.pt")), cfg, checkpoints / "best.pt", root / "selection.json")
    result = run_inference(str(checkpoints / "best.pt"), cfg.data.processed_path,
                           cfg.knowledge.templates_jsonl, cfg.knowledge.entity_to_token_json,
                           cfg.data.split_json, "test", device="cpu", out_path=str(root / "inference.json"))
    print("Example workflow completed (not clinical results):")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Outputs: {root}")


if __name__ == "__main__":
    main()
