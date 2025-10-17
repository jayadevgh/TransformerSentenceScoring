# ================================
# train.py  —  Tokenized LM training
# - Builds/loads BPE tokenizer (with progress)
# - Streams A-side text, 80/10/10 split
# - Trains GPT with AMP + grad accumulation
# - Periodically logs train/val/test losses
# - Saves weights/config to checkpoints/lm.pt
# ================================

# --- Allocator hint BEFORE torch import (fresh process recommended) ---
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import time, random, math, gc
from pathlib import Path
import torch

# Optional perf knobs
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('medium')  # PyTorch 2.x
except AttributeError:
    pass

# Local modules
from tokenizer import Tokenizer, NUM_SPECIALS, PAD_ID, BOS_ID, EOS_ID, UNK_ID
from model import GPTLanguageModel
from data import encode_lines_to_stream, get_batch

# --------------------
# Config (tweak freely)
# --------------------
TRAIN_PATH        = "train_utf8.txt"
SAVE_DIR          = Path("checkpoints"); SAVE_DIR.mkdir(parents=True, exist_ok=True)
META_FILE         = SAVE_DIR / "meta.pt"
CKPT_FILE         = SAVE_DIR / "lm.pt"

# BPE training subset (only used if tokenizer isn't saved yet)
BPE_TARGET_VOCAB  = 4092
BPE_TRAIN_LINES   = 50_000      # sample this many A-side lines to learn merges
BPE_MIN_PAIR_CNT  = 2

# Model / training
batch_size        = 4           # micro-batch size (grad accumulation below)
grad_accum_steps  = 16          # effective batch = batch_size * grad_accum_steps
block_size        = 256         # context window (T)
max_steps         = 10_000 # 20_000
eval_interval     = 1000
learning_rate     = 3e-4
weight_decay      = 0.10
n_embd            = 384
n_head            = 6
n_layer           = 6
dropout           = 0.2
seed              = 1337

device  = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda")
random.seed(seed); torch.manual_seed(seed)

# ---------------------------
# 1) Load A-side + 80/10/10
# ---------------------------
A_lines = []
with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if "\t" in line:
            a, _ = line.rstrip("\n").split("\t", 1)
            A_lines.append(a)

N = len(A_lines)
n_train = int(0.8 * N)
n_val   = int(0.1 * N)

train_lines = A_lines[:n_train]
val_lines   = A_lines[n_train:n_train+n_val]
test_lines  = A_lines[n_train+n_val:]
print(f"[splits] train={len(train_lines):,} val={len(val_lines):,} test={len(test_lines):,}")

# ---------------------------------------
# 2) Tokenizer: load or train & then save
# ---------------------------------------
if META_FILE.exists():
    tok = Tokenizer.load(META_FILE)
    print("[tokenizer] loaded", META_FILE)
else:
    # Train on a random subset for speed (keeps it unbiased)
    pool = list(train_lines)
    random.shuffle(pool)
    bpe_lines = pool[:min(BPE_TRAIN_LINES, len(pool))]
    name = f"BPE({len(bpe_lines):,}-rand)"
    tok = Tokenizer.train_from_lines(
        bpe_lines, target_vocab=BPE_TARGET_VOCAB, min_pair_count=BPE_MIN_PAIR_CNT, name=name
    )
    tok.save(META_FILE)
    print("[tokenizer] trained & saved", META_FILE)

# Free any cached GPU memory before model build
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()

# --------------------------
# 3) Encode streams (BOS/EOS)
# --------------------------
train_stream = encode_lines_to_stream(train_lines, tok, add_bos_eos=True)
val_stream   = encode_lines_to_stream(val_lines, tok, add_bos_eos=True)
test_stream  = encode_lines_to_stream(test_lines, tok, add_bos_eos=True)
print("[streams] lens:", len(train_stream), len(val_stream), len(test_stream))

# ---------------
# 4) Build model
# ---------------
model = GPTLanguageModel(
    vocab_size=tok.vocab_size,
    block_size=block_size,
    n_embd=n_embd, n_head=n_head, n_layer=n_layer, dropout=dropout
).to(device)
print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# ---------------------------
# 5) Eval helper (MC estimate)
# ---------------------------
@torch.no_grad()
def estimate_loss(iters=200):
    model.eval()
    out = {}
    for name, stream in [("train", train_stream), ("val", val_stream), ("test", test_stream)]:
        losses = torch.zeros(iters)
        for k in range(iters):
            X, Y = get_batch(stream, batch_size, block_size, device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                _, loss = model(X, Y, ignore_index=PAD_ID)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out

# --------------------
# 6) Training loop
# --------------------
t0 = time.time()
model.train()
for step in range(1, max_steps + 1):
    optimizer.zero_grad(set_to_none=True)

    # Gradient accumulation to emulate larger batch
    for _ in range(grad_accum_steps):
        X, Y = get_batch(train_stream, batch_size, block_size, device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            _, loss = model(X, Y, ignore_index=PAD_ID)
            loss = loss / grad_accum_steps
        scaler.scale(loss).backward()

    # (optional) gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()

    if step % eval_interval == 0 or step == 1:
        stats = estimate_loss(iters=200)
        elapsed_min = (time.time() - t0) / 60
        print(f"step {step:6d} | train {stats['train']:.4f} | val {stats['val']:.4f} "
              f"| test {stats['test']:.4f} | {elapsed_min:.1f} min")

# --------------------
# 7) Save checkpoint
# --------------------
torch.save({
    "model": model.state_dict(),
    "config": {
        "vocab_size": tok.vocab_size,
        "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer,
        "dropout": dropout, "block_size": block_size
    }
}, CKPT_FILE)
print("[ckpt] saved", CKPT_FILE)
