"""Every Approach B setting in one place, each with the reason for its value."""

from __future__ import annotations

import os
from pathlib import Path

B_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = B_ROOT.parent
DATASET_DIR = PROJECT_ROOT / "dataset"          # the only input shared with Approach A
VALIDATOR = PROJECT_ROOT / "utils" / "validate_submission.py"  # organizer-provided checker
CACHE_DIR = B_ROOT / "cache"
MODELS_DIR = B_ROOT / "models"
REPORTS_DIR = B_ROOT / "reports"
OUTPUT_DIR = B_ROOT / "output"

# Model weights live inside approach_b so nothing is shared with other projects.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hub"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

SEED = 13
WORKERS = max(1, min(20, (os.cpu_count() or 4) - 2))  # leave cores for the OS / GPU feeder

# --- Entity split -----------------------------------------------------------
HOLDOUT_MODULUS = 10             # blake2b(id) % 10 == 0 -> hold-out (10%)
TRAIN_ENTITIES = 200_000         # stratified sample from the other 90%

# --- Retrieval --------------------------------------------------------------
LEXICAL_TOP_K = 50               # measured flat recall curve past 50 on this data
LEXICAL_MIN_DF = 2               # a token seen once cannot link two records
LEXICAL_MAX_DF = 0.01            # measured: 0.001 lost 7 recall points, 0.01 kept them
DENSE_MODEL = "intfloat/multilingual-e5-small"   # MIT, 384-d, ~100 languages
DENSE_PREFIX = "query: "         # model card: use "query: " on both sides for symmetric similarity
DENSE_MAX_LEN = 64               # names plus addresses fit; longer inputs only cost time
DENSE_TOP_K = 50
RRF_K = 60                       # Cormack et al. 2009: near-optimal and not critical

# --- Cross-encoder (Ditto) --------------------------------------------------
TOKENIZER_NAME = "FacebookAI/xlm-roberta-base"   # MiniLM ships XLM-R's vocabulary
BACKBONES = {
    "minilm": "microsoft/Multilingual-MiniLM-L12-H384",   # MIT, 21M transformer params
    "xlmr": "FacebookAI/xlm-roberta-base",                # MIT, 12 x 768
}
SPECIAL_TOKENS = ["[COL]", "[VAL]", "[NUM]", "[/NUM]", "[ZIP]", "[/ZIP]"]
MAX_LEN = 160                    # measured pair lengths: p99.5 = 149, p99.9 = 166 tokens
LEARNING_RATE = 3e-5             # Ditto
BATCH_SIZE = 64                  # per GPU; global batch is BATCH_SIZE * world
EPOCHS = 2
WARMUP_FRACTION = 0.05
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
PREDICT_BATCH = 4096             # 96 GB cards; scoring batches
TOKENIZE_BATCH = 4096            # one-time pair tokenization
EVAL_EVERY_STEPS = 10_000        # avoid stalling 4 GPUs on calib NLL

# --- Decision ---------------------------------------------------------------
BETA = 0.5                       # the challenge metric is F0.5
SHIP_BASELINE = 0.948            # Approach A hold-out macro F0.5
SHIP_MARGIN = 0.005              # B replaces A only if it beats baseline by this much
INCLUDE_COUNTRY = True           # dropped only if leave-one-country-out falls hard
DK_TAGS = True                   # Ditto domain-knowledge span tags; measured as a flag
ATTR_DEL_P = 0.0                 # 0.1 to try Ditto attr_del; off because the train set is large
ADDRESS_DROP_P = 0.15            # drop the address on a positive during matcher train
OCR_NOISE_P = 0.08               # digit/letter substitutions on serialized text
COUNTRY_DROPOUT_P = 0.2          # omit the country field so France is not a shock
CAP_EXEMPT = 0.99                # p at or above this ignores the per-source cap
SOURCE_CAP_S2 = 5                # gold never has more than 5 Source-2 matches
SOURCE_CAP_S3 = 6                # gold never has more than 6 Source-3 matches
POSTING_MAX_DF = 200             # drop inverted keys that hit more documents than this
BLANK_INDEX_K = 5
DOMAIN_INDEX_K = 8
NONCE_INDEX_K = 8
KEEP_PER_SOURCE_MAX = 30         # scoring-time budget (~60 candidates per S1)
TABLES_DIR = MODELS_DIR / "tables"
STACKER_DIR = MODELS_DIR / "stacker"
DECISION_REPORT = REPORTS_DIR / "07_decision.json"
BAKEOFF_ENTITIES = 30_000        # one-epoch backbone comparison
DPR_TEMPERATURE = 0.05           # Karpukhin et al. 2020
DPR_POSITIVES = 1_000_000        # training-entity positives for the optional dense fine-tune
CALIB_LOSS_PAIRS = 32_768        # running checkpoint metric during matcher training
SCORE_SHARD_ENTITIES = 20_000    # resumable test-score files
ADDRESS_SUMMARY_WORDS = 36       # Ditto TF-IDF address summary when a pair would overflow
