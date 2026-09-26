# Business entity resolution

Local train, hold-out evaluation, and submission checks for the three-source matching task.

Training and test files stay on disk. The pipeline does not call external identity, registry, or geocoding services.

The **current** method lives in `approach_b/`. This folder’s `src/` is an earlier features + LightGBM stack and is not used for scoring.

---

## Layout

Place the challenge files at the **project root** (tab-separated):

```
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

`approach_b/output/` is where the current method writes `matching_results.tsv` and `candidate_pairs.tsv`.

---

## Environment

```bash
cd approach_b
python3 -m venv .venv
. .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
PYTHONPATH=. python scripts/verify.py
```

Need a GPU with bfloat16. Inference should set `HF_HUB_OFFLINE=1`.

---

## Commands

```bash
cd approach_b
PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/05_predict_test.py

python3 ../utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir ../dataset/test
```

Full train recipe (prepare → retrieve → MiniLM → stacker → decide) is in `approach_b/README.md`.

Evaluation is macro F0.5 over every Source 1 entity, including singletons. Country values are read as strings and are not restricted to the training labels.

---

## Current method

**Problem.** Link noisy business records: Source 1 (unique reference) to Source 2 and Source 3. Gold is one-to-one on the Source 2 / Source 3 side. Train countries are India and US; test also has France (~15% of Source 1, **no train gold**). F0.5 weights precision twice recall. No external lookup.

**Prepare.** NFKC clean, anyascii romanization, then mined `name_norm` (abbreviation expansion, Indic dictionary from gold, domain Viterbi). Parsed house / street / city / state are features, never AND-filters. Missing address = no vote (MAR).

**Block (OR lists, not all-pairs).** Word TF-IDF top 50 + multilingual-e5-small kNN top 50, fused with Reciprocal Rank Fusion (k=60), keep 25 per source. Extra lists: blank-address names, segmented domains, nonce house+street keys (drop keys with DF > 200). No conjunctive PIN or city filter.

**Match.** Ditto-style MiniLM cross-encoder on `[COL] name/address/state/city/house/country [VAL] …`. Omit blank fields. Train 2 epochs on 200k stratified Source 1 entities × ~50 candidates, with address-drop, light OCR noise, country dropout. Temperature scaling on the calibration half (T ≈ 1.074).

**Stack.** LightGBM on matcher probability, per-source rank, gap, blank/nonce/domain flags, name-claimant count, house+street claimant count, prefix, house conflict. Train on calibration, score evaluation.

**Decide.** Keep if p ≥ 0.74, p ≥ 0.5 · best, rank-in-source ≤ min(6, 5 on Source 2 / 6 on Source 3), unless p ≥ 0.99. Unique Mapping Clustering assigns each Source 2/3 id to at most one Source 1. Grid on **true** calibration F0.5 (blocking misses count as misses).

**Ship numbers.** Eval F0.5 **0.983** (P 0.997, R 0.957). Test file **0.9646** (France gap).

**France.** Not in train. Source 1 last slot ≈ three régions; Source 2/3 mix région, département, and city; comma order varies; SAS/SARL vs Pvt/Ltd; `r.`/`bd` vs `rue`/`boulevard`. Unsupervised alias mining from test strings is allowed if disclosed; a geocoder is not. Do not require city match when Source 2 omitted the city. Popular cities are not standalone blocks.

**Not used.** Live lookup, AGPL romanizers, PIN AND-filters, ensembling with this folder’s `src/`.

The line-by-line recreation spec (every script, constant, and report file) is **`approach_b/README.md`**.
