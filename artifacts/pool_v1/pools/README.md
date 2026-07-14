# Experimental pool payloads

The four paper-reported frozen pool files are not available in this checkout:

- `mimic_iii_primekg.templates.jsonl` — 920 templates reported
- `mimic_iv_primekg.templates.jsonl` — 1,180 templates reported
- `mimic_iii_umls.templates.jsonl` — 874 templates reported
- `mimic_iv_umls.templates.jsonl` — 1,092 templates reported

Their absence is intentional and recorded as `missing_required` in the release
manifest. Do not create placeholder or synthetic records with these filenames.
Strict release validation must continue to fail until authentic payloads and
their provenance are supplied and verified.
