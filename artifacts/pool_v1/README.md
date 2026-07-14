# ReTA frozen knowledge-pool release `pool_v1`

Release version: `1.0.0-incomplete`

This directory is an integrity-checked provenance package, but it is **not a release of the paper's experimental knowledge pools**. The MIMIC-III/MIMIC-IV, PrimeKG, and UMLS pool files needed to reproduce the reported experiments are absent from the repository and were not available during this release audit. They are listed as missing in `manifest.json`; no records or measurements have been fabricated to replace them.

The only included pool is `demo/templates.jsonl`, a byte-for-byte copy of `examples/tiny_templates.jsonl`. It contains one hand-authored, four-dimensional fixture used by the smoke test. It is suitable for format validation only and must not be used as an experimental result.

## Included material

| Path | Purpose |
| --- | --- |
| `config.yaml` | Paper-reported generation settings, current code defaults, and explicit unknowns. |
| `prompts/figure8_system.txt` | System prompt transcribed from paper Figure 8. |
| `prompts/figure8_user.txt` | Input template and output constraints transcribed from Figure 8. |
| `schemas/llm_response.schema.json` | Release-side structured-response contract; not claimed as a paper artifact. |
| `schemas/distilled_artifact.schema.json` | JSONL contract used by the repository's `DistilledArtifact`. |
| `filtering.py` | Deterministic, fail-closed implementation of the paper's grounding and relation-support rules. |
| `verify_release.py` | Dependency-light hash, metadata, and template-invariant verifier. |
| `provenance/sources.json` | Repository, paper, model, and unavailable-source provenance. |
| `provenance/reported_diagnostics.json` | Table 6 values, clearly marked as transcribed and not recomputed. |
| `demo/templates.jsonl` | One-record development fixture copied from `examples/`. |
| `pools/README.md` | Reserved experimental filenames and missing-payload notice. |
| `manifest.json` | File hashes, included/missing pool inventory, and release status. |

## Paper-reported generation configuration

Appendix B.2 reports one independent GPT-4o request per concept, with no patient context, using temperature `0.2`, top-p `0.9`, and at most `256` tokens. Figure 8 asks for a one-sentence pathology definition and a clinical cascade containing one to five downstream complications. Sparse PrimeKG neighborhoods may receive up to five entries; dense neighborhoods may receive as few as one. The paper does not define numeric density-bin boundaries.

Appendix B.3 reports exact ontology matching followed by a ClinicalBERT Top-1 fallback accepted only when similarity is greater than `0.90`. Relation support is a direct PrimeKG edge or an ancestor/descendant relationship within two CCS hierarchy levels. The paper names ClinicalBERT but does not identify a checkpoint or immutable revision.

Agglomerative clustering with cosine distance and projection to `d=256` is reported. The prose and tables do not state the clustering cut `tau` or linkage. `pool_v1` resolves `tau=0.16` by inference from the zero-delta default in Figure 9, records that inference separately from paper-reported facts, and aligns the current code default. Other values under `current_code_defaults` in `config.yaml` describe this checkout; they are not retroactively attributed to the paper.

## GPT-4o snapshot and access date

The paper and repository do not report a GPT-4o snapshot or access date. This task thread later declared `gpt-4o-2024-11-20` and `2026-07-13`. They are preserved under `thread_declared_post_hoc` with status `unverified`. No API logs or original outputs support them, they do not establish which model generated the historical experimental pools, and they do not apply to the demo fixture.

## Paper-reported diagnostics

The following values are transcribed from Table 6 and cannot be recomputed from this incomplete release:

| Diagnostic | MIMIC-III | MIMIC-IV |
| --- | ---: | ---: |
| Format pass | 97.6% | 97.1% |
| Mean tokens/request | 148 | 156 |
| Ontology mapping success | 95.3% | 93.8% |
| Externally supported links | 83.7% | 81.9% |
| First failure: format | 2.4% | 2.9% |
| First failure: mapping/missing entity | 4.7% | 6.2% |
| First failure: low external support | 11.6% | 12.8% |
| PrimeKG templates | 920 | 1,180 |
| Average nodes/template | 7.6 | 8.1 |
| Average edges/template | 16.8 | 17.9 |
| Median intracluster cosine distance | 0.14 | 0.15 |
| UMLS templates | 874 | 1,092 |

The UMLS variant is additionally reported at 7.2 nodes and 15.3 edges per template on average, without a dataset-specific density breakdown.

## Validation and intended use

Validate the included demo template contract with:

```bash
python -m reta.cli validate-template-pool \
  --templates_jsonl artifacts/pool_v1/demo/templates.jsonl \
  --expected_dim 4
```

Check `manifest.json` hashes and validate the intentionally incomplete release with:

```bash
python -m reta.cli validate-pool-release \
  --release_dir artifacts/pool_v1 \
  --allow_incomplete
```

Strict validation omits `--allow_incomplete` and must fail while the experimental pools remain absent. A verifier must not treat them as optional simply because the demo record validates.

To complete this release, supply the four missing frozen pools plus their original concept inventories, raw model responses, source versions and checksums, filtering audit, embedding/projection provenance, and immutable model revisions. Update the status only after their record counts and hashes are independently verified.
