"""Central configuration.

Defaults are the ones the feasibility run justified, not the ones a tutorial
would use -- see `scripts/feasibility.py` and entry 1 of NOTES.md.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EVAL_DIR = ROOT / "eval"
EVAL_SET = EVAL_DIR / "eval_set.jsonl"
ADAPTER_DIR = ROOT / "artifacts" / "adapter"
REPORTS = ROOT / "reports"

# Ungated and Apache-2.0. Deliberate: a gated checkpoint makes the repo
# unreproducible for anyone who has not accepted a licence, and this project
# already depends on nothing but a HF download.
BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# bfloat16, not float32. Measured on an M4/24 GB: fp32 needs 19.5 GB and runs at
# 69 s/step, bf16 needs 14.2 GB and runs at 4.2 s/step. The 16x is mostly
# unified-memory pressure rather than arithmetic -- fp32 sits close enough to the
# ceiling that the machine thrashes.

def __getattr__(name: str):
    """Resolve `DTYPE` and `DEVICE` on first access, importing torch lazily.

    They were module-level, which meant importing this config -- for a path, a
    seed, a filename -- dragged in torch. CI installs only pytest, on the stated
    grounds that these tests "need no model, no GPU and no network", so
    collection died at import with ModuleNotFoundError and the badge went red
    while the tests themselves were fine.

    PEP 562 keeps the call sites unchanged: `config.DTYPE` still works, and only
    train/evaluate/forgetting touch it -- all of which import torch anyway.
    Installing torch in CI would also fix the red badge, at ~200 MB against a
    workflow built to stay fast, and would leave the real coupling in place.
    """
    if name == "DTYPE":
        import torch

        return torch.bfloat16
    if name == "DEVICE":
        import torch

        return "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



SEED = 42

# --- Task -----------------------------------------------------------------
CATEGORIES = ["travel", "meals", "software", "hardware", "office", "other"]
FIELDS = ["vendor", "amount", "currency", "date", "category"]

SYSTEM_PROMPT = (
    "You extract expense details from informal messages. "
    "Reply with a single JSON object and nothing else, using exactly these keys: "
    'vendor, amount, currency, date, category. '
    "Use null for anything the message does not state. "
    "amount is a number, currency is a 3-letter ISO code, date is YYYY-MM-DD, "
    f"category is one of {CATEGORIES} or null. "
    "If the message is not about an expense, every value is null."
)

# --- LoRA -----------------------------------------------------------------
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = 0.05
# Attention projections only, which is the common default and keeps trainable
# parameters at 0.28%. Including the MLP roughly triples that.
#
# Not tuned: r=16 and this module set were chosen up front and never swept, so
# nothing here is claimed to be optimal. The rank/target sweep is listed as
# untested in the README rather than implied to have been done -- one run took
# 74 minutes on this hardware and a sweep was out of budget.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# --- Training -------------------------------------------------------------
EPOCHS = int(os.getenv("EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
LR = float(os.getenv("LR", "1e-4"))
MAX_LEN = 384
N_TRAIN = int(os.getenv("N_TRAIN", "1500"))

# --- Generation -----------------------------------------------------------
# Greedy. The task has one correct answer, so sampling would add variance to a
# measurement rather than value to an output.
MAX_NEW_TOKENS = 120
