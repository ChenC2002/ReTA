# ReTA knowledge release `pool_v1`

Version `1.0.0`

This directory contains the versioned configuration assets for ReTA's offline
knowledge-pool pipeline. Reusable Python implementations live directly in
`reta/knowledge/`.

Clinical text embeddings use deterministic attention-mask mean pooling with
the exact model name and immutable model revision recorded in `config.yaml`.
Grounding consumes precomputed Top-1 embedding candidates; the filtering step
does not download or execute an embedding model.

## Release files

| Path | Purpose |
| --- | --- |
| `config.yaml` | Distillation, grounding, filtering, clustering, and pool-output settings. |
| `system_prompt.txt` | Clinical knowledge-distillation system instruction. |
| `user_prompt.txt` | Diagnosis input template and structured-output requirements. |
| `schemas/llm_response.schema.json` | Distillation-response contract. |
| `schemas/distilled_artifact.schema.json` | Distilled-artifact JSONL contract. |
| `manifest.json` | Release identity, dependency metadata, file inventory, and hashes. |

## Validate

```bash
python -m reta.cli validate-pool-release
```

The validator checks every file declared in `manifest.json`, verifies release
configuration and prompt contracts, and returns a nonzero status on failure.

Clinical datasets and third-party biomedical resources remain governed by their
providers' access and redistribution terms.
