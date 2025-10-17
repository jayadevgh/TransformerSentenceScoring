from pathlib import Path
import torch, random

# ---- random + device ----
SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

# ---- i/o ----
TRAIN_PATHS = ["train_utf8.txt", "train.txt"]  # use first that exists
OUT_PATH    = "part2.txt"

# ---- artifact roots (as requested) ----
CHAR_CKPT_DIR = Path("./CharArtifacts")
BPE_CKPT_DIR  = Path("./TokenArtifacts")

# ---- build knobs ----
PRINT_EVERY         = 10_000   # progress cadence
MAX_TRIES_PER_LINE  = 3        # try up to K different corruptions per line
MAX_EDITS           = 2        # micro-typo budget
LENGTH_DRIFT        = 0.15     # allow some char-count drift in acceptance (soft)
MAX_LEN_CHARS       = 2000     # safety clamp
