# Business entity resolution

Self-contained record linkage for Amazon ML Challenge 2026. Three TSV sources: Source 1 is the unique reference; Source 2 and Source 3 are noisy copies. Training and test files stay on disk. There are no live identity, registry, or geocoding calls. Inference sets `HF_HUB_OFFLINE=1`.

This folder is the method that is scored. Models are MIT / Apache-2.0 / ISC and well under 8B.

**Contest constraints encoded here.** Country is an open string (train: India, US; test also has France). Metric is **macro F0.5** (β = 0.5) over every Source 1 entity, **including singletons**. Gold is one-to-one on the Source 2 / Source 3 side. Source 1 may match several records in the other sources.

---

## Layout

Challenge files (tab-separated) at the **project** root:

```
dataset/train/train_source{1,2,3}.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source{1,2,3}.tsv
```

This folder:

```
approach_b/
  config.py           hyperparameters and paths
  data.py             TSV I/O, blake2b 90/5/5 entity split
  text.py             NFKC clean, anyascii, Ditto serialize, parse, flags
  mining.py           Indic dictionary, address aliases, domain unigrams from gold
  lexical.py          word TF-IDF top-k
  dense.py            multilingual-e5-small encode + GPU kNN
  candidates.py       Reciprocal Rank Fusion
  postings.py         exact-key inverted lists (house|street|city)
  pairs.py            labeled candidate tables
  cross_encoder.py    Ditto MiniLM train / bf16 predict
  calibrate.py        one-scalar temperature T
  claimants.py        name-superset and house+street claimant counts
  stacker.py          LightGBM on matcher p + flags + claimants
  decide.py           threshold, margin, per-source cap, Unique Mapping Clustering
  metrics.py          per-entity F0.5
  scripts/            one stage per file
  cache/              parquet, npz, score shards
  models/             matcher, stacker, mined tables, HuggingFace hub cache
  reports/            JSON from each stage
  output/             matching_results.tsv, candidate_pairs.tsv
```

`candidate_pairs.tsv` is the last retrieval cut. Unique Mapping Clustering only **deletes** matches, so every predicted id is a subset of that file.

---

## Environment

```bash
cd approach_b
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
PYTHONPATH=. python scripts/verify.py
```

Need a GPU with bfloat16. `HF_HOME` defaults to `approach_b/models/hub`.

Pinned: torch 2.11.0, transformers 5.17.0, scikit-learn 1.9.1, anyascii 0.3.3 (ISC), lightgbm 4.6.0 (MIT). Dense encoder `intfloat/multilingual-e5-small` (MIT). Matcher `microsoft/Multilingual-MiniLM-L12-H384` (Apache-2.0 / MIT family) with the XLM-R tokenizer. Not used: Aksharamukha (AGPL), IndicXlit, Llama, Qwen3-Embedding, Gemini.

---

## Pipeline (one pass)

```
records → prepare → mine tables from gold → enrich
       → block (OR lists + RRF) → MiniLM scores → temperature
       → LightGBM stacker → keep-rule → Unique Mapping Clustering → TSVs
```

Missing fields are **MAR**: a blank address or missing city does not vote; it is not a negative. Blocking lists are **OR**, never AND geo filters. A digit-must-match PIN rule dropped 7.4% of train gold and is not used.

### 1. Entity split (`data.py`)

`blake2b(id)`: 90% train pool, 10% hold-out; hold-out split in half to **calibration** and **evaluation**. From the 90%, a **stratified 200,000** Source 1 sample is the matcher train set (`TRAIN_ENTITIES`, seed 13). Grid search and the ship metric use **true** gold counts per Source 1 (blocking misses count as misses, not as singletons).

### 2. Prepare (`scripts/01_prepare.py`)

NFKC, strip `<null>`, collapse empty comma slots, romanize with **anyascii** (fixed character table; no lookup). Write `cache/{train,test}/source{1,2,3}.parquet` with `id`, `country`, `name`, `address`, `name_roman`, `address_roman`.

### 3. Mine and enrich (`scripts/08_mine.py`, `scripts/09_enrich.py`)

From **training gold only** (hold-out Source 1 excluded for measurement tables):

- Indic word dictionary (abbreviation expansion, positional / fuzzy alignment, per-script support).
- Address alias tables (native states, abbreviations, city aliases).
- Character unigrams for domain-style name Viterbi (`amazonwebservices` → tokens).

`09_enrich.py` adds `name_norm`, `address_norm`, `blank_address`, `nonce_like`, `domain_like`, parsed `house` / `street_core` / `city` / `state`. Retrieval prefers `*_norm` when present. Parsed city/state are **features**, never AND-filters.

### 4. Block (`scripts/02_retrieve.py`)

Never all-pairs. For each Source 1 entity, build several lists against Source 2 and Source 3 separately, then fuse.

| List | How | Depth |
|---|---|---|
| Lexical | sklearn word TF-IDF, `min_df=2`, `max_df=0.01`, roman `name_norm` + `address_norm` | top 50 |
| Dense | `multilingual-e5-small`, prefix `query: ` on both sides, max length 64, exact GPU kNN | top 50 |
| Blank | TF-IDF on **name only**, documents whose address is empty (Latin names) | top 5 |
| Domain | TF-IDF on domain-like names only | top 8 |
| Nonce | inverted `house\|street\|city` keys; drop keys with DF > **200** | top 8 |

Fuse with **Reciprocal Rank Fusion** (Cormack et al. 2009, k=60). Keep **25 per source** (smallest K that keeps 99% of fused@100 recall, capped at 30). About 50 candidates per Source 1. Dense encoder is used off-the-shelf (no DPR fine-tune on the shipped run).

### 5. Match (`scripts/04_train_matcher.py`, `cross_encoder.py`)

`pairs.py` labels the fused candidates. Backbone MiniLM. Each record is Ditto-serialized:

```
[COL] name [VAL] … [COL] address [VAL] … [COL] state [VAL] … [COL] city [VAL] … [COL] house [VAL] … [COL] country [VAL] …
```

Blank extras are omitted. `[NUM]` / `[ZIP]` tags when `DK_TAGS`. Train 2 epochs, AdamW 3e-5, per-GPU batch 64, bf16, on the 200k entities × ~50 candidates. Augmentations on positives: drop address (p=0.15), light OCR 0/o/1/l noise (p=0.08), omit country (p=0.2) so the token `france` is not required. Best checkpoint by calibration NLL. Then one-scalar **temperature T** on calibration logits (Guo et al. 2017).

### 6. Stack (`scripts/07_decision.py`, `stacker.py`)

LightGBM (binary logloss, 250 trees, 63 leaves) trained on **calibration**, scored on **evaluation**. Features:

- matcher probability (after T)
- per-source rank, gap to best on that Source 1
- blank / nonce / domain flags
- name prefix, house-number conflict
- **name claimants**: how many Source 1 records contain every content token of the candidate name
- **address claimants**: how many Source 1 records share house + street
- source (2 vs 3), log1p of both claimant counts

Claimant indexes use **all** Source 1 rows in the split, so hold-out and test behave the same. If the stacker loses to raw matcher probability on eval, it is disabled; on this run it is used (`score_source=stacker`).

### 7. Decide (`decide.py`)

Ranks are **per source**. Keep a candidate if:

- best score on the entity ≥ gate (shipped gate = 0)
- score ≥ threshold **t**
- score ≥ α × best
- per-source rank ≤ min(cap, 5) on Source 2 and ≤ min(cap, 6) on Source 3

Scores ≥ **0.99** skip the cap. Then **Unique Mapping Clustering** (Papadakis et al., EDBT 2022): sort surviving triples by score; each Source 2/3 id is given to at most one Source 1.

Grid on calibration **true** macro F0.5: t ∈ {0.50…0.90}, cap 1–12, α ∈ {0.5…1.0}, gate ∈ {0…0.70}. For a calibrated classifier the Fβ-optimal threshold is about t* = F*/(1+β²) (Lipton, Elkan, Narayanaswamy 2014); with β=0.5 that is ~0.8·F*. The **grid**, not the formula, picks the point.

**Shipped rule** (`reports/07_decision.json`): T ≈ **1.074**, t=**0.74**, cap=**6**, α=**0.5**, gate=**0**.

### 8. Test (`scripts/05_predict_test.py`)

Same retrieval on test, score fused pairs with the frozen MiniLM in shards of 20k Source 1 ids (resumable parquets), apply T, stacker, rule, UMC, write TSVs, run `utils/validate_submission.py`.

---

## France

Train gold has **zero** France. Test Source 1 is about 15% France. Last slot on Source 1 is almost only three régions; Source 2/3 mix région, département, and city; comma order is not stable; legal forms differ (`SAS`/`SARL` vs `Pvt`/`Ltd`); street tokens differ (`r.`/`bd` vs `rue`/`boulevard`). Unsupervised alias mining from test **strings** is allowed if disclosed; a geocoder is not. Do not require city match when Source 2 omitted the city. Popular cities are not standalone blocks (`POSTING_MAX_DF=200`).

---

## Commands

From `approach_b/` with `PYTHONPATH=.` and the venv. GPU stages relaunch with `torchrun` when more than one device is visible. Offline inference: `HF_HUB_OFFLINE=1`.

```bash
python scripts/00_benchmark.py
python scripts/01_prepare.py
python scripts/02_retrieve.py --split train --stage entities
python scripts/08_mine.py
python scripts/09_enrich.py
python scripts/02_retrieve.py --split train --stage lexical
python scripts/02_retrieve.py --split train --stage dense --dense-input roman
python scripts/02_retrieve.py --split train --stage extra
python scripts/02_retrieve.py --split train --stage fuse --dense-tag roman_multilingual-e5-small
python scripts/02_retrieve.py --split test --stage lexical
python scripts/02_retrieve.py --split test --stage dense --dense-input roman
python scripts/02_retrieve.py --split test --stage extra
python scripts/02_retrieve.py --split test --stage fuse --dense-tag roman_multilingual-e5-small
python scripts/04_train_matcher.py --skip-bakeoff --backbone minilm
python scripts/07_decision.py
python scripts/05_predict_test.py

python3 ../utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir ../dataset/test
```

Optional: `python scripts/06_error_ledger.py` (loss by blocking miss / below threshold / cap / UMC, by country and script).

---

## Numbers (this checkout)

| Set | Macro F0.5 | Precision | Recall |
|---|---|---|---|
| Evaluation half | **0.983** | **0.997** | **0.957** |
| Calibration (rule fit on this half) | 0.983 | 0.998 | 0.956 |
| Test portal | **0.9646** | — | — |

Eval: 110,225 Source 1 entities, singleton accuracy 0.987. India slice ~0.977; US ~0.987. The test drop vs eval is consistent with France (~15% of test Source 1) not appearing in train.

---

## Design choices we measured

**Kept.** OR blocking; MAR missing fields; transliterate (anyascii + gold dictionary), do not translate; Viterbi on glued domains; RRF; UMC; LightGBM as part of the **model** (candidates stay the fused list); per-source cap with p≥0.99 exempt; country dropout in matcher train.

**Dropped.** Aksharamukha; MuRIL as a sentence encoder; IndicXlit; Metaphone/Soundex as primary; AND PIN/city filters; meaning NMT; ensembling with the older GBM stack under `code/business_entity_resolution/src/`.

---

## Tests

```bash
PYTHONPATH=. python -m pytest tests/test_normalize.py -q
```
