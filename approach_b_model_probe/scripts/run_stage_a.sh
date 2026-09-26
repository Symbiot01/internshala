#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VENV="${PROBE_PYTHON:-/home/dev/amazon-ml-entity-resolution/approach_b/.venv/bin/python}"
"$VENV" scripts/00_snapshot.py
"$VENV" scripts/00b_score_minilm_eval.py
"$VENV" scripts/01_build_probe_set.py
"$VENV" scripts/02_train_xlmr.py
"$VENV" scripts/04_score_probe.py --model xlmr
"$VENV" scripts/05_splice_eval.py --model xlmr
if "$VENV" -c "import json; from pathlib import Path; p=Path('reports/delta.json');
raise SystemExit(0 if p.exists() and json.loads(p.read_text()).get('promote') else 1)"; then
  echo "XLM-R promoted; Stage B requires peft in a probe venv"
  "$VENV" scripts/03_train_3b_lora.py
  "$VENV" scripts/04_score_probe.py --model qwen3b
  "$VENV" scripts/05_splice_eval.py --model qwen3b
else
  echo "GATE FAIL or delta.json missing: skip 3B"
fi
