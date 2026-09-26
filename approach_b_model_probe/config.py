"""Probe-only paths and hyperparameters. Does not write into approach_b/."""

from __future__ import annotations

import os
from pathlib import Path

PROBE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PROBE_ROOT.parent
B_ROOT = PROJECT_ROOT / "approach_b"
DATASET_DIR = PROJECT_ROOT / "dataset"

CACHE_DIR = PROBE_ROOT / "cache"
MODELS_DIR = PROBE_ROOT / "models"
REPORTS_DIR = PROBE_ROOT / "reports"
LIB_DIR = PROBE_ROOT / "lib"

os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hub"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

SEED = 13
WORKERS = max(1, min(20, (os.cpu_count() or 4) - 2))

HOLDOUT_MODULUS = 10
TRAIN_ENTITIES = 200_000

TOKENIZER_NAME = "FacebookAI/xlm-roberta-base"
DENSE_MODEL = "intfloat/multilingual-e5-small"
DENSE_PREFIX = "query: "
DENSE_MAX_LEN = 64
BACKBONES = {
    "minilm": "microsoft/Multilingual-MiniLM-L12-H384",
    "xlmr": "FacebookAI/xlm-roberta-base",
    "qwen3b": "Qwen/Qwen2.5-3B",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
}
SPECIAL_TOKENS = ["[COL]", "[VAL]", "[NUM]", "[/NUM]", "[ZIP]", "[/ZIP]"]
MAX_LEN = 160
LEARNING_RATE = 3e-5
BATCH_SIZE = 64
EPOCHS = 1
WARMUP_FRACTION = 0.05
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
PREDICT_BATCH = 4096
TOKENIZE_BATCH = 4096
EVAL_EVERY_STEPS = 10_000

BETA = 0.5
INCLUDE_COUNTRY = True
DK_TAGS = True
ATTR_DEL_P = 0.0
ADDRESS_DROP_P = 0.0
OCR_NOISE_P = 0.0
COUNTRY_DROPOUT_P = 0.0
CAP_EXEMPT = 0.0
SOURCE_CAP_S2 = 99
SOURCE_CAP_S3 = 99
ADDRESS_SUMMARY_WORDS = 36
BAKEOFF_ENTITIES = 30_000
CALIB_LOSS_PAIRS = 32_768
CALIB_T_PAIRS = 32_768
QWEN_TRAIN_ENTITIES = 8_000
QWEN_PAIR_CAP = 400_000
LORA_R = 16
LORA_ALPHA = 32
QWEN_MICRO_BATCH = 16
PROMOTE_D_F05 = 0.001
PROMOTE_MIN_P = 0.993

PROBE_BELOW = 4000
PROBE_HITS = 2500
PROBE_FPS = 2500
PROBE_SINGLETONS = 1000
ORACLE_GOLD = 1000
ORACLE_NEGS = 5

REQUIRED_PARQUET_COLUMNS = ("id", "country", "name", "address", "name_roman", "address_roman")
