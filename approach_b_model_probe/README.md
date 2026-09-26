# Isolated model probe

Does **not** import live `approach_b/` after snapshot and does **not** write B `cache/`, `models/`, or `reports/`. Use this folder to test whether a bigger matcher helps, in parallel with B recall upgrades.

## GPU split

```bash
export CUDA_VISIBLE_DEVICES=2,3   # B keeps 0,1
cd /home/dev/amazon-ml-entity-resolution/approach_b_model_probe
```

`distributed.maybe_relaunch` uses every **visible** GPU.

Python: `approach_b/.venv/bin/python` for Stage A (XLM-R). Stage B needs `peft` in a **separate** venv if you install it; do not `pip install` into B’s venv.

## Proof order

1. Snapshot + probe set + **XLM-R 1-epoch** + splice (~1 hour).
2. Promote only if spliced eval **ΔF0.5 ≥ 0.001** and **P ≥ 0.993**.
3. Then LoRA **Qwen2.5-3B** (Phi-3-mini fallback). **No 7B.**

## Commands

```bash
export CUDA_VISIBLE_DEVICES=2,3
VENV=/home/dev/amazon-ml-entity-resolution/approach_b/.venv/bin/python
$VENV scripts/00_snapshot.py
$VENV scripts/00b_score_minilm_eval.py
$VENV scripts/01_build_probe_set.py
$VENV scripts/02_train_xlmr.py
$VENV scripts/04_score_probe.py --model xlmr
$VENV scripts/05_splice_eval.py --model xlmr
# only if reports/delta.json "promote": true
$VENV scripts/03_train_3b_lora.py
$VENV scripts/04_score_probe.py --model qwen3b
$VENV scripts/05_splice_eval.py --model qwen3b
```

Or: `CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_stage_a.sh`

Decision replay uses the **original 04_decision.json** rule (`t=0.76`, `cap=6`, `α=0.5`, `gate=0`) with per-entity ranks (not B’s later source-cap / stacker).

## Organizer question (paste; do not block Stage A)

Constraint 5: “Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters.”

- Per-net cap vs sum of all nets?
- Is “final model” only the matcher on `candidate_pairs.tsv`?
- Is e5-small + 3B matcher allowed?
- Must every weight file be MIT/Apache?

Until they answer: every HF model we might ship is MIT/Apache and each ≤ 8B.

## Will not do

Retrieve, test scoring, MiniLM 2-epoch train, 7B, train on eval FNs, ensemble with A.
