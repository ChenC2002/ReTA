# Data preparation and training

## Required Inputs

Full runs require credentialed MIMIC-III or MIMIC-IV diagnosis data, ICD-to-CCS
mappings, a CCS hierarchy, PrimeKG or UMLS-derived resources, and externally
generated concept-level distillation responses.

### EHR Files

| Input                   | Required format                                              |
| ----------------------- | ------------------------------------------------------------ |
| Diagnosis events        | CSV with `patient_id,visit_id,timestamp,icd_code`; add `icd_version` for mixed ICD-9/10 data |
| ICD-to-CCS mapping      | CSV with `icd,ccs`; add `icd_version` for a versioned mapping |
| Preprocessing hierarchy | CSV with `child,parent`                                      |

### Knowledge Files

| Input                  | Required format                                              |
| ---------------------- | ------------------------------------------------------------ |
| Concept inventory      | CSV with `concept_id,concept_name` and optional `density` (`sparse`, `moderate`, or `dense`) |
| Distillation responses | JSONL with exactly `definition` and `clinical_cascade`, one response per concept in matching row order |
| Biomedical inventory   | CSV with `entity_id,name` and optional `source`              |
| Embedding candidates   | Optional CSV/JSONL with `mention,entity_id,score`            |
| PrimeKG support        | CSV with `u,v` and optional `relation`                       |
| CCS filtering support  | CSV with `parent,child`                                      |
| Canonical descriptions | JSON object mapping every ICD/CCS/KG ID in the token vocabulary to its canonical description |

Both hierarchy readers use the named `parent` and `child` columns, so column
order does not matter. Preserve the parent/child direction and compatible CCS
identifiers; preprocessing additionally requires a single-parent hierarchy.

## Workflow

### Step 1: Preprocess

For MIMIC exports, use the admission ID as `visit_id` and admission time as
`timestamp`. All diagnoses from the same admission form one visit, including
stays spanning multiple days.

```bash
python -m reta.data.preprocess \
  --events data/raw/diagnoses.csv \
  --visit_id_col visit_id \
  --icd2ccs data/resources/icd_to_ccs.csv \
  --ccs_hierarchy data/resources/ccs_hierarchy.csv \
  --out_dir data/processed \
  --h_anc 2
```

For mixed ICD-9/10 input, include an `icd_version` column in both files and
append `--icd_version_col icd_version` and
`--mapping_icd_version_col icd_version` to the command.

The command writes `processed.pt` and a patient-disjoint `splits.json`.
Without `--visit_id_col`, legacy event data uses 24-hour windows.

### Step 2: Build the Pool

Generate one concept-level response per row using the [prompt and schema](../reta/knowledge/releases/pool_v1).
The following command validates the supplied responses saved in JSONL format.

```bash
python -m reta.knowledge.distill \
  --concepts_csv data/resources/concepts.csv \
  --responses_jsonl data/resources/distillation_responses.jsonl \
  --model_family GPT-4o \
  --out_jsonl data/knowledge/artifacts.jsonl
```

Ground mentions and retain only externally supported relations:

```bash
python -m reta.knowledge.filtering \
  --artifacts-jsonl data/knowledge/artifacts.jsonl \
  --inventory-csv data/resources/inventory.csv \
  --embedding-candidates data/resources/clinicalbert_top1.csv \
  --primekg-edges-csv data/resources/primekg_edges.csv \
  --ccs-edges-csv data/resources/ccs_support_edges.csv \
  --out-grounded-jsonl data/knowledge/grounded.jsonl \
  --out-audit-jsonl results/filtering_audit.jsonl \
  --out-summary-json results/filtering_summary.json
```

Clustering uses pinned Bio_ClinicalBERT features and saves both the pool and its
projection (`templates.jsonl.projection.pt`):

```bash
python -m reta.knowledge.clustering \
  --grounded_jsonl data/knowledge/grounded.jsonl \
  --out_jsonl data/knowledge/templates.jsonl \
  --tau 0.16 \
  --projection_dim 256
```

Validate the pool and assign stable token IDs to external medoid-subgraph nodes
that are not already in the processed ICD/CCS vocabulary:

```bash
reta validate-template-pool \
  --templates_jsonl data/knowledge/templates.jsonl \
  --expected_dim 256

reta build-entity-map \
  --processed_path data/processed/processed.pt \
  --templates_jsonl data/knowledge/templates.jsonl \
  --out data/resources/entity_to_token.json
```

Cache canonical-description features using the **same** encoder and projection:

```bash
python -m reta.knowledge.features \
  --processed_path data/processed/processed.pt \
  --entity_to_token_json data/resources/entity_to_token.json \
  --descriptions_json data/resources/descriptions.json \
  --templates_jsonl data/knowledge/templates.jsonl \
  --projection data/knowledge/templates.jsonl.projection.pt \
  --out data/knowledge/concept_features.pt
```

For example, `descriptions.json` contains entries such as
`{"ICD:25000": "Diabetes mellitus without complication", "CCS:49": "Diabetes mellitus without complication"}`.
Provide descriptions for all vocabulary tokens, including external KG nodes.
Training freezes these cached features and learns the shared `W_code` projection.

## Train and evaluate

Follow the [training, validation selection, and evaluation commands](../README.md#workflow).
[Implementation notes](workflow.md) cover recovery and independent seeds.

## Outputs

The following files are generated when you run the pipeline.

| Output                                       | Path                                                         |
| -------------------------------------------- | ------------------------------------------------------------ |
| Distilled, grounded, and clustered knowledge | `data/knowledge/{artifacts,grounded,templates}.jsonl`        |
| Filtering diagnostics                        | `results/filtering_audit.jsonl`, `results/filtering_summary.json` |
| Frozen text features and pool projection | `data/knowledge/concept_features.pt`, `data/knowledge/templates.jsonl.projection.pt` |
| Training checkpoints                         | `checkpoints/warmup.pt`, `checkpoints/rl_iter*.pt`, `checkpoints/best.pt`           |
| Validation checkpoint selection             | `results/selection.json`                                   |
| Local diagnosis evaluation                   | `results/inference.json`                                     |
| Runtime logs                                 | `logs/warmup.log`, `logs/rl_train.log`, `logs/inference.log` |

