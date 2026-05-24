"""
UMLS / KG grounding and support filtering.

Inputs:
1) artifacts.jsonl from distill.py
2) inventory.csv with columns [entity_id, name, (source)]
3) support_edges.csv with columns [u, v] (canonical ids)

Output:
grounded.jsonl (GroundedTemplate)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .templates import DistilledArtifact, GroundedEntity, GroundedTemplate
except ImportError:  # allow running as a script
    from templates import DistilledArtifact, GroundedEntity, GroundedTemplate  # type: ignore


def normalize_mention(x: str) -> str:
    x = (x or "").strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


class Inventory:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["name_norm"] = self.df["name"].map(normalize_mention)
        self.name_to_row = {}
        for _, row in self.df.iterrows():
            n = row["name_norm"]
            if n and n not in self.name_to_row:
                self.name_to_row[n] = row
        self._emb = None
        self._emb_model = None
        self._emb_kind = None
        self._emb_tokenizer = None

    @staticmethod
    def from_csv(path: str) -> "Inventory":
        df = pd.read_csv(path)
        if "entity_id" not in df.columns or "name" not in df.columns:
            raise ValueError("inventory.csv must contain columns: entity_id, name")
        if "source" not in df.columns:
            df["source"] = "unknown"
        return Inventory(df)

    def exact(self, mention: str) -> Optional[GroundedEntity]:
        m = normalize_mention(mention)
        row = self.name_to_row.get(m)
        if row is None:
            return None
        return GroundedEntity(
            entity_id=str(row["entity_id"]),
            name=str(row["name"]),
            source=str(row.get("source", "unknown")),
            score=1.0,
        )

    def _encode_transformer(self, texts: List[str]) -> np.ndarray:
        import torch

        outs = []
        with torch.no_grad():
            for start in range(0, len(texts), 32):
                batch = texts[start : start + 32]
                enc = self._emb_tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
                device = next(self._emb_model.parameters()).device
                enc = {k: v.to(device) for k, v in enc.items()}
                hidden = self._emb_model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                outs.append(pooled.cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)

    def _ensure_embeddings(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        if self._emb is not None:
            return
        names = self.df["name"].astype(str).tolist()
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
            emb = model.encode(names, show_progress_bar=False, normalize_embeddings=True)
            self._emb = np.array(emb, dtype=np.float32)
            self._emb_model = model
            self._emb_kind = "sentence_transformer"
            return
        except Exception:
            pass

        try:
            from transformers import AutoModel, AutoTokenizer

            self._emb_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._emb_model = AutoModel.from_pretrained(model_name)
            self._emb_model.eval()
            self._emb_kind = "transformer"
            self._emb = self._encode_transformer(names)
            return
        except Exception:
            self._emb = None
            self._emb_model = None
            self._emb_tokenizer = None
            self._emb_kind = None

    def fallback_embed(
        self,
        mention: str,
        tau_map: float = 0.90,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    ) -> Optional[GroundedEntity]:
        self._ensure_embeddings(model_name=model_name)
        if self._emb is None:
            return None

        if self._emb_kind == "sentence_transformer":
            m_emb = self._emb_model.encode([mention], show_progress_bar=False, normalize_embeddings=True)
            m_emb = np.array(m_emb, dtype=np.float32)
        elif self._emb_kind == "transformer":
            m_emb = self._encode_transformer([mention])
        else:
            return None
        sims = (self._emb @ m_emb.T).reshape(-1)

        j = int(np.argmax(sims))
        if float(sims[j]) < tau_map:
            return None

        row = self.df.iloc[j]
        return GroundedEntity(
            entity_id=str(row["entity_id"]),
            name=str(row["name"]),
            source=str(row.get("source", "unknown")),
            score=float(sims[j]),
        )


class SupportGraph:
    def __init__(self, edges: List[Tuple[str, str]]):
        self.adj: Dict[str, List[str]] = {}
        for u, v in edges:
            self.adj.setdefault(u, []).append(v)
            self.adj.setdefault(v, []).append(u)

    @staticmethod
    def from_csv(path: str, u_col: str = "u", v_col: str = "v") -> "SupportGraph":
        df = pd.read_csv(path)
        if u_col not in df.columns or v_col not in df.columns:
            raise ValueError(f"support_edges.csv must contain columns: {u_col}, {v_col}")
        edges = []
        for u, v in zip(df[u_col], df[v_col]):
            u_s, v_s = str(u).strip(), str(v).strip()
            if not u_s or not v_s or u_s.lower() == "nan" or v_s.lower() == "nan":
                continue
            edges.append((u_s, v_s))
        return SupportGraph(edges)

    def has_edge(self, u: str, v: str) -> bool:
        return v in self.adj.get(u, [])

    def within_hops(self, u: str, v: str, h: int = 2) -> bool:
        if u == v:
            return True
        if self.has_edge(u, v):
            return True
        frontier = {u}
        visited = {u}
        for _ in range(h):
            nxt = set()
            for x in frontier:
                for y in self.adj.get(x, []):
                    if y == v:
                        return True
                    if y not in visited:
                        visited.add(y)
                        nxt.add(y)
            frontier = nxt
            if not frontier:
                break
        return False


def load_artifacts_jsonl(path: str) -> List[DistilledArtifact]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(DistilledArtifact.from_dict(json.loads(line)))
    return out


def save_grounded_jsonl(items: List[GroundedTemplate], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")


def ground_one(
    art: DistilledArtifact,
    inv: Inventory,
    support: Optional[SupportGraph] = None,
    tau_map: float = 0.90,
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
) -> GroundedTemplate:
    entities: List[GroundedEntity] = []
    for m in art.cascade:
        e = inv.exact(m)
        if e is None:
            e = inv.fallback_embed(m, tau_map=tau_map, model_name=model_name)
        if e is not None:
            entities.append(e)

    unique_entities: List[GroundedEntity] = []
    seen_entity_ids = set()
    for e in entities:
        if e.entity_id in seen_entity_ids:
            continue
        seen_entity_ids.add(e.entity_id)
        unique_entities.append(e)
    entities = unique_entities

    # Compact subgraph: root + cascade entities, retaining only externally
    # supported links as in Appendix B.3.
    root = art.concept_id
    nodes = [root] + [e.entity_id for e in entities]

    edges: List[Tuple[str, str]] = []
    if support is not None:
        seen_edges = set()
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                if support.within_hops(u, v, h=2) and (u, v) not in seen_edges:
                    edges.append((u, v))
                    seen_edges.add((u, v))
        supported_ids = {root}
        for u, v in edges:
            supported_ids.add(u)
            supported_ids.add(v)
        entities = [e for e in entities if e.entity_id in supported_ids]
        nodes = [root] + [e.entity_id for e in entities]

    return GroundedTemplate(
        root_concept_id=root,
        root_name=art.concept_name,
        definition=art.definition,
        cascade_entities=entities,
        subgraph_nodes=list(dict.fromkeys(nodes)),
        subgraph_edges=edges,
        meta=dict(art.meta),
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ground and filter distilled artifacts into KG-aligned templates.")
    p.add_argument("--artifacts_jsonl", type=str, required=True)
    p.add_argument("--inventory_csv", type=str, required=True)
    p.add_argument("--support_edges_csv", type=str, default=None)
    p.add_argument("--out_jsonl", type=str, required=True)
    p.add_argument("--tau_map", type=float, default=0.90)
    p.add_argument("--embed_model", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    return p


def main():
    args = build_argparser().parse_args()
    arts = load_artifacts_jsonl(args.artifacts_jsonl)
    inv = Inventory.from_csv(args.inventory_csv)
    support = SupportGraph.from_csv(args.support_edges_csv) if args.support_edges_csv else None

    out: List[GroundedTemplate] = []
    for a in arts:
        out.append(ground_one(a, inv, support=support, tau_map=args.tau_map, model_name=args.embed_model))

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    save_grounded_jsonl(out, args.out_jsonl)
    print(f"[grounding] wrote {len(out)} grounded templates to {args.out_jsonl}")


if __name__ == "__main__":
    main()
