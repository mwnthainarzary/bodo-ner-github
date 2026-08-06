# Legal NER Project

Trains and evaluates NER models on the corrected legal BIO-tagged dataset
(`data.csv` = your `NER-DATA-ENGLISH_corrected.csv`, 101 documents, ~14K tokens).

## Files
- `data_prep.py` — loads the CSV, groups rows into per-document sequences,
  splits by **document id** (not row) to avoid train/test leakage.
- `train_crf.py` — CRF baseline. Runs anywhere, no internet needed. **Already
  run in this environment** — see results below.
- `train_transformer.py` — fine-tunes a transformer (default: `law-ai/InLegalBERT`,
  a BERT model pretrained on Indian legal text). **Needs internet access to
  huggingface.co** to download the pretrained weights — run this in Google
  Colab or on a machine with normal internet access, not in a locked-down
  sandbox.
- `crf_model.joblib` — the trained CRF model (already produced).

## How to run

```bash
pip install -r requirements.txt

# CRF baseline (works everywhere)
python3 train_crf.py

# Transformer fine-tuning (needs internet access to the HF Hub)
python3 train_transformer.py
```

To try a different base model for the transformer script, change `MODEL_NAME`
at the top of `train_transformer.py`, e.g.:
- `bert-base-uncased` — general-domain baseline, good for a comparison row
- `nlpaueb/legal-bert-base-uncased` — general legal-domain BERT
- `law-ai/InLegalBERT` — Indian-legal-domain BERT (default; closest match to
  this data)

## CRF baseline results (already run, on this dataset)

Split: 70 train / 15 dev / 16 test documents (document-level split, seed=42).

| Split | Micro F1 |
|---|---|
| Dev  | 0.588 |
| Test | 0.573 |

Well-supported entity types do reasonably well (entity-level F1):
- `DATE`: 0.83–0.90
- `SECTION`: 0.31–0.83 (varies a lot by split — small test set)
- `DESIGNATION`: 0.69–0.80
- `COURT`: 0.67–0.75
- `POLICE_STATION`: 1.00 (dev split; small support)

Sparse entity types (`ANIMAL`, `BANK`, `EDUCATION`, `EXAM`, `TENDER`, etc.,
each with only 1–5 examples total) score 0 — there simply isn't enough data
for the model to learn them. **This is worth stating explicitly in the
chapter** as a limitation of the corpus size rather than a modeling failure.

## Suggested chapter narrative

1. Describe the dataset and the annotation-consistency issues found and
   corrected (typo'd tags, systematic role-word mislabeling, BIO-span
   errors) — this is a genuinely useful, honest section most papers skip.
2. Report the CRF baseline numbers above.
3. Fine-tune `law-ai/InLegalBERT` (and optionally `bert-base-uncased` for
   comparison) using `train_transformer.py`, report entity-level F1 the same
   way (seqeval), and compare against the CRF baseline in a table.
4. Do error analysis on a few sentences: show where the model
   over/under-predicts, and tie it back to the sparse classes and the
   annotation-consistency issues from part 1.
5. Limitations: small corpus (101 documents), long-tail label imbalance,
   single-annotator inconsistency (before correction).
