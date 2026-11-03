"""Cache frozen canonical-description features for Eq. 1; no patient text is used."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .clustering import TextEmbedder
from .pool import KnowledgePool
from reta.learning.checkpointing import atomic_save
from reta.learning.runtime import (
    EntityTokenMapper, file_fingerprint, load_json_object_strict,
    load_processed_data, mapping_fingerprint,
)


def load_concept_features(path, mapper, dim, templates_path):
    """Validate that cached rows and the projection belong to this vocabulary/pool."""
    if path is None:
        return {}
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    mapping = {**mapper.base_mapping, **mapper.external_mapping}
    if artifact.get("mapping_sha256") != mapping_fingerprint(mapping):
        raise ValueError("concept features do not match the token mapping.")
    if artifact.get("templates_sha256") != file_fingerprint(templates_path):
        raise ValueError("concept features do not match the template pool.")
    features, projection = artifact["features"], artifact["projection"]
    if (features.ndim != 2 or features.shape[0] != mapper.vocab_size
            or projection.shape != (dim, features.shape[1])):
        raise ValueError("concept feature/projection dimensions do not match the model.")
    if not torch.isfinite(features).all() or not torch.isfinite(projection).all():
        raise ValueError("concept features and projection must be finite.")
    return {"text_features": features, "text_projection": projection}


def cache_concept_features(processed_path, entity_map, descriptions_json,
                           templates_path, projection_path, out_path):
    data = load_processed_data(processed_path)
    base_size = int(data["meta"]["icd_vocab_size"]) + int(data["meta"]["ccs_vocab_size"])
    mapper = EntityTokenMapper.from_sources(data["vocab"]["name_to_token"], entity_map, base_size)
    mapping = {**mapper.base_mapping, **mapper.external_mapping}
    descriptions = load_json_object_strict(descriptions_json, "canonical descriptions")
    ordered = sorted(mapping, key=mapping.get)
    missing = [key for key in ordered if not isinstance(descriptions.get(key), str)
               or not descriptions[key].strip()]
    if missing:
        raise ValueError(f"Missing canonical descriptions for {len(missing)} tokens, e.g. {missing[:5]}")
    destination = Path(out_path).resolve()
    if destination in {Path(p).resolve() for p in
                       (processed_path, descriptions_json, templates_path, projection_path, entity_map)
                       if p is not None}:
        raise ValueError("Feature output must not overwrite an input.")
    projection = torch.load(projection_path, map_location="cpu", weights_only=True)
    pool = KnowledgePool.load_jsonl(templates_path)
    vectors = projection.get("template_vectors")
    if vectors is None or vectors.shape != pool.P.shape or not np.allclose(vectors.numpy(), pool.P, atol=1e-6):
        raise ValueError("Projection does not belong to this template pool; rerun clustering.")
    embedder = TextEmbedder(model_name=projection["model_name"],
                            revision=projection["model_revision"], backend=projection["backend"])
    features = torch.from_numpy(embedder.encode([descriptions[key] for key in ordered]))
    weight = projection["weight"]
    if weight.ndim != 2 or features.shape[1] != weight.shape[1]:
        raise ValueError("Text encoder and saved projection dimensions do not match.")
    artifact = {"features": features, "projection": weight,
                "mapping_sha256": mapping_fingerprint(mapping),
                "templates_sha256": file_fingerprint(templates_path),
                "model_name": projection["model_name"], "model_revision": projection["model_revision"]}
    atomic_save(artifact, destination)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed_path", required=True)
    parser.add_argument("--entity_to_token_json", required=True)
    parser.add_argument("--descriptions_json", required=True, help="Canonical entity ID -> description JSON object")
    parser.add_argument("--templates_jsonl", required=True)
    parser.add_argument("--projection", required=True, help="Projection artifact saved by clustering")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    artifact = cache_concept_features(args.processed_path, args.entity_to_token_json,
                                      args.descriptions_json, args.templates_jsonl, args.projection, args.out)
    print(f"Cached {artifact['features'].shape[0]} canonical descriptions to {args.out}")


if __name__ == "__main__":
    main()
