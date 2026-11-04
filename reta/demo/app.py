"""Small FastAPI application; model loading happens once at startup."""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
from pathlib import Path
import threading
import time

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from reta.data.ontology import CCSOntology, normalize_icd_code
from reta.learning.prediction import load_prediction_session
from reta.learning.runtime import load_config, load_json_object_strict, load_processed_data


EXAMPLES = [
    {"name": "Diabetes & kidney disease", "visits": [["25000", "4019", "2724"], ["25040", "5853", "4019", "2724"]]},
    {"name": "Heart failure", "visits": [["4019", "41401", "2724"], ["4280", "42731", "4019", "41401"]]},
    {"name": "Respiratory conditions", "visits": [["496", "4019"], ["49121", "486", "4019"]]},
]


class PredictionRequest(BaseModel):
    visits: list[list[str]] = Field(min_length=1, max_length=6)
    top_k: int = Field(default=10, ge=1, le=20)


class DemoModel:
    def __init__(self, config, checkpoint, icd2ccs, ccs_hierarchy, descriptions, device="cpu", results=None):
        cfg = load_config(config)
        data = load_processed_data(cfg.data.processed_path)
        self.h_anc = int(data['meta'].get('h_anc', 2))
        self.session = load_prediction_session(
            checkpoint, cfg.data.processed_path, cfg.knowledge.templates_jsonl,
            cfg.knowledge.entity_to_token_json, cfg.data.split_json, device=device, data=data,
        )
        self.ontology = CCSOntology.from_files(icd2ccs, ccs_hierarchy)
        if self.ontology.name_to_token != self.session.mapper.base_mapping:
            raise ValueError("Demo ontology does not match the checkpoint vocabulary.")
        self.descriptions = load_json_object_strict(descriptions, "concept descriptions")
        if any(not isinstance(value, str) for value in self.descriptions.values()):
            raise ValueError("Concept descriptions must be strings.")
        self.names = {token: name for name, token in {
            **self.session.mapper.base_mapping, **self.session.mapper.external_mapping,
        }.items()}
        self.lock = threading.Lock()
        self.results = json.loads(Path(results).read_text()) if results else None
        self.threshold = float(self.session.ck.get("selection", {}).get("prediction_threshold", 0.5))

    def name(self, canonical):
        return self.descriptions.get(canonical, canonical)

    def build_visit(self, codes):
        if not 1 <= len(codes) <= 20:
            raise ValueError("Each visit needs between 1 and 20 ICD-9-CM codes.")
        if any(not isinstance(code, str) or len(code) > 20 for code in codes):
            raise ValueError("Invalid ICD-9-CM code.")
        normalized = list(dict.fromkeys(normalize_icd_code(code) for code in codes))
        unknown = [code for code in normalized if self.ontology.icd_to_token(code) is None]
        if unknown:
            raise ValueError("Unknown ICD-9-CM codes: " + ", ".join(unknown))
        tokens = sorted(self.ontology.icd_to_token(code) for code in normalized)
        labels = self.ontology.icd_tokens_to_ccs_tokens(tokens)
        ancestors = self.ontology.ancestors_within_h(labels, h=self.h_anc)
        nodes = sorted(set(tokens + ancestors))
        return {"icd_tokens": tokens, "ccs_ancestors": ancestors, "node_ids": nodes,
                "edge_index": self.ontology.edges_among_nodes(nodes)}

    def predict(self, visits, top_k=10):
        graphs = [self.build_visit(codes) for codes in visits]
        started = time.perf_counter()
        with self.lock:
            self.session.reset()
            try:
                steps = []
                for index, visit in enumerate(graphs):
                    logits, trace = self.session.predict_visit(visit, trace=True)
                    probabilities = torch.sigmoid(logits)[0]
                    ranking = torch.argsort(probabilities, descending=True)[:top_k].tolist()
                    predictions = []
                    for label in ranking:
                        canonical = self.names[self.session.icd_vocab_size + label]
                        predictions.append({"code": canonical.removeprefix("CCS:"), "name": self.name(canonical),
                                            "probability": float(probabilities[label]),
                                            "before": trace["raw_probabilities"][label]})
                    candidates = []
                    for candidate in trace["candidates"]:
                        template = self.session.pool.get_template(candidate["template_id"])
                        candidates.append({**candidate, "name": template.medoid.root_name,
                                           "definition": template.medoid.definition,
                                           "nodes": len(template.medoid.subgraph_nodes),
                                           "selected": candidate["template_id"] == trace["template_id"]})
                    original = set(trace["original_node_ids"])
                    graph_nodes = [{"id": token, "code": self.names[token], "name": self.name(self.names[token]),
                                    "kind": "imported" if token not in original else
                                            "observed" if token in visit["icd_tokens"] else "ontology"}
                                   for token in trace["node_ids"]]
                    edge_index = trace["edge_index"]
                    node_ids = trace["node_ids"]
                    edges = sorted({tuple(sorted((node_ids[a], node_ids[b])))
                                    for a, b in zip(*edge_index) if a != b})
                    steps.append({"visit": index + 1, "action": trace["action"],
                                  "selected_template": trace["template_id"], "added_nodes": trace["added_nodes"],
                                  "candidates": candidates, "predictions": predictions,
                                  "graph": {"nodes": graph_nodes, "edges": edges},
                                  "codes": [{"code": self.names[t].removeprefix("ICD:"), "name": self.name(self.names[t])}
                                            for t in visit["icd_tokens"]]})
            finally:
                self.session.reset()
        return {"steps": steps, "elapsed_ms": round((time.perf_counter() - started) * 1000),
                "threshold": self.threshold}

    def metadata(self):
        return {"version": "V1.0", "examples": EXAMPLES,
                "templates": len(self.session.pool.templates), "labels": self.session.ccs_label_vocab_size,
                "has_policy": self.session.policy_net is not None, "results": self.results}


def create_app(model_factory):
    @asynccontextmanager
    async def lifespan(app):
        app.state.model = model_factory()
        yield
        del app.state.model

    app = FastAPI(title="ReTA Demo", version="1.0", lifespan=lifespan)

    @app.get("/api/meta")
    def metadata():
        return app.state.model.metadata()

    @app.get("/api/concepts")
    def concepts(q: str = ""):
        if len(q) > 80:
            raise HTTPException(422, "Search text is too long.")
        model = app.state.model
        query = q.strip().lower()
        return [{"code": name[4:], "name": model.name(name)}
                for name in model.ontology.name_to_token
                if name.startswith("ICD:") and (query in name.lower() or query in model.name(name).lower())][:30]

    @app.post("/api/predict")
    def predict(request: PredictionRequest):
        try:
            return app.state.model.predict(request.visits, request.top_k)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    app.mount("/", StaticFiles(directory=Path(__file__).with_name("static"), html=True), name="demo")
    return app
