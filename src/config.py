import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PA3_DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("PA3_OUTPUT_DIR", ROOT / "outputs"))

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_FILE = PROCESSED_DIR / "sd_15min.npz"
SD_META_FILE = PROCESSED_DIR / "sd_meta.csv"

FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
RESULTS_DIR = OUTPUT_DIR / "results"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

KAGGLE_DATASET = "liuxu77/largest"
META_FILE = "ca_meta.csv"
ADJ_FILE = "ca_rn_adj.npy"
HIS_FILES = {2020: "ca_his_raw_2020.h5", 2021: "ca_his_raw_2021.h5"}

SD_DISTRICT = 11
EXPECTED_NODES = 716
YEARS = [2020, 2021]
RAW_FREQ = "5min"
FREQ = "15min"
RESAMPLE_METHOD = "mean"
READ_CHUNK_ROWS = 2048

TRAIN_START = "2020-01-01"
VAL_START = "2021-01-01"
TEST_START = "2021-04-01"
TEST_END = "2022-01-01"

INPUT_LEN = 8
OUTPUT_LEN = 4
STEPS_PER_DAY = 96
HORIZON_MINUTES = [15, 30, 45, 60]

PLOT_WEEK_START = "2020-02-03"
TEST_PLOT_WEEK_START = "2021-05-03"
PLOT_SENSOR_QUANTILES = [0.1, 0.5, 0.9, 0.3, 0.7, 0.97]

SEED = 42

TRAIN_DEFAULTS = {
    "epochs": 50,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "hidden": 32,
    "embed_dim": 10,
    "graph_layers": 1,
    "dropout": 0.1,
    "topk": 0,
    "alpha": "learn",
    "patience": 10,
    "clip": 5.0,
    "ckpt_every": 2,
    "eval_batch_size": 256,
}

MODEL_TYPES = ["temporal", "fixed", "adaptive", "hybrid"]
MODEL_LABELS = {
    "temporal": ("Temporal-only", "None"),
    "fixed": ("STGNN", "Fixed"),
    "adaptive": ("STGNN", "Adaptive"),
    "hybrid": ("STGNN", "Hybrid"),
}
