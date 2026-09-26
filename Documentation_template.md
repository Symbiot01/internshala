# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Approach B  
**Team Members:** [fill]  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary

Hybrid retrieval (word TF-IDF + multilingual-e5-small + routed blank/domain/nonce indexes, fused with Reciprocal Rank Fusion) followed by a Ditto-style MiniLM cross-encoder, a LightGBM stacker on matcher probability plus claimant features, and Unique Mapping Clustering. Tables for Indic names and address aliases are mined only from provided training gold (hold-out Source 1 excluded). France administrative aliases are mined from unlabeled test inputs by pivoting on shared city strings. No live lookups.

---

## 2. Methodology

### 2.1 Problem Analysis

Train countries are India and US; test also contains France (~15% of Source 1). Gold is one-to-one on the Source 2/3 side. Blank addresses are rare (~3.4%) but almost always true matches; nonce aliases and Indic-script names dominate recall loss. PIN codes are almost absent in India rows, so state/city/house are parsed as features, not filters. Transliteration dictionaries mined from gold beat generic romanizers for Indic names.

### 2.2 Solution Strategy

**Approach Type:** Hybrid blocking + Ditto cross-encoder + LightGBM stacker + Unique Mapping Clustering  
**Core Innovation:** Decision-layer stacker with global name/address claimant counts; mined Indic token dictionary; routed sub-population indexes; MAR omission of blank parsed fields.

Fair-play: `HF_HUB_OFFLINE=1` at inference. Mined tables from train gold only. France aliases from test inputs only. Synthetic France pairs were built for diagnostics; **fine-tuning on them is off** (would require disclosure). Licenses: MIT / Apache-2.0 / ISC / BSD only (see `approach_b/models/LICENSE_MANIFEST.json`).

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** word TF-IDF over `name_norm + address_norm`; e5 dense kNN; blank-address name index; domain-segmented name index; nonce house+street+city postings. Fused with RRF.
- **Candidate pairs generated:** ~25 per source per Source 1 entity (~50 total; cap ~60).
- **How you ensured true matches were not lost:** disjunctive OR lists (never AND geo filters); missing fields vote neither way; extra routed lists for the measured miss types.

---

## 4. Matching Model

**Features used:**
- Name features: Ditto serialized name (+ roman + mined `name_norm`); claimant count (Source 1 token-superset); token-prefix flag
- Address features: house / street_core / city / state (omitted if blank); house-conflict; address claimant count
- Other: matcher logit, per-source rank, gap-to-best, blank/nonce/domain flags

**Model type:** MiniLM cross-encoder + LightGBM stacker  
**Threshold selection method:** Grid on calibration-half true macro F0.5. For a calibrated classifier the Fβ-optimal threshold is about t* = F*/(1+β²) (Lipton et al. 2014); with β=0.5 that is 0.8·F*, so the search is centred near 0.76–0.78. Unique Mapping Clustering (Papadakis et al., EDBT 2022, pp. 462–474) enforces 1-1 on candidate ids.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** evaluation-half after matcher retrain + stacker: **0.9830** (P 0.9969, R 0.9566, singleton accuracy 0.9866). Phase-0 fallback test F0.5 eval was 0.9744. Reports: `approach_b/reports/07_decision.json`, `04_decision.json`.
- **Common false positives:** nonce aliases sharing a house+street with several Source 1 records.
- **Common false negatives:** remaining Indic OOV tokens (Tamil/Malayalam weakest); blocking misses on blank-address Indic names.

---

## 6. Conclusion

Precision was already saturated; recall came from the decision layer first (per-source caps, cap-exempt high probabilities, claimant-aware stacker) and then from mined normalization plus routed indexes. France is handled with unsupervised admin aliases and street abbreviations, without fine-tuning on unlabeled test pairs.

---

## Appendix

### A. Code Artefacts

Runnable stack: `approach_b/` (`scripts/01_prepare.py` … `05_predict_test.py`, plus `08_mine.py`, `09_enrich.py`, `10_france.py`). Entry point for submission TSVs: `scripts/05_predict_test.py`. Fallback Phase-0 files: `approach_b/output/fallback_phase0/`.

### B. Additional Results

Error ledger: `approach_b/reports/06_error_ledger.json`. Table provenance: `approach_b/models/tables/provenance.json`. License manifest: `approach_b/models/LICENSE_MANIFEST.json`.
