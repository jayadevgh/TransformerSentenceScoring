

import os, io, math, json, time, random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------
# Config (tweak freely)
# --------------------
TRAIN_PATH = "train_utf8.txt"        # provided training pairs (A \t B) per line
TEST_PATH  = "test_utf8.rand.txt"    # randomized test pairs (A/B order unknown)
OUT_PATH   = "part1.txt"        # required submission file

SAVE_DIR   = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# Model / training
batch_size   = 64
block_size   = 256          # max context for training blocks
max_steps    = 20_000       # adjust for compute
eval_interval= 1000
learning_rate= 3e-4
n_embd       = 384
n_head       = 6
n_layer      = 6
dropout      = 0.2
seed         = 1337

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(seed)
random.seed(seed)

# Mixed precision can speed up training on GPU
use_amp = (device == 'cuda')

# ---------------------------------------------------
# 0) Build English-only corpus (contiguous, no pads)
# ---------------------------------------------------
# We stream train.txt once, extract the left field (before the tab),
# and write an on-disk corpus with '\n' separators. Then we read it back.

corpus_file = Path(SAVE_DIR) / "english_corpus.txt"
meta_file   = Path(SAVE_DIR) / "meta.pt"            # stores stoi/itos and settings
ckpt_file   = Path(SAVE_DIR) / "lm.pt"              # model weights

if not corpus_file.exists():
    print("Creating english_corpus.txt from training pairs...")
    with open(TRAIN_PATH, "r", encoding="utf-8") as fin, \
         open(corpus_file, "w", encoding="utf-8") as fout:
        for line in fin:
            if '\t' not in line:
                continue
            a, b = line.rstrip('\n').split('\t', 1)
            # write the English sentence + newline separator
            fout.write(a)
            fout.write("\n")
    print("Done.")

# Read corpus (one big string)
with open(corpus_file, "r", encoding="utf-8") as f:
    corpus = f.read()

print(f"Corpus size (chars): {len(corpus):,}")

# ---------------------------
# 1) Char vocabulary (no PAD)
# ---------------------------
# We keep a tiny <unk> to handle any unseen test characters gracefully.
# (This is NOT used as padding and never inserted by us.)

chars = sorted(list(set(corpus)))
UNK_TOKEN = "<unk>"
itos = [UNK_TOKEN] + chars
stoi = {ch:i for i, ch in enumerate(itos)}
unk_id = stoi[UNK_TOKEN]
vocab_size = len(itos)

def encode_string(s: str):
    # map to ids; unseen char -> unk
    return [stoi.get(c, unk_id) for c in s]

def decode_ids(ids):
    return ''.join(itos[i] for i in ids)

# Save meta so scoring can reload encoders
torch.save(
    {"itos": itos, "stoi": stoi, "block_size": block_size},
    meta_file
)

# ---------------------------------
# 2) Prepare training tensor stream
# ---------------------------------
# Classic contiguous-stream training: sample random fixed-length blocks.

data = torch.tensor(encode_string(corpus), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

def get_batch(split):
    source = train_data if split == 'train' else val_data
    ix = torch.randint(len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, iters=200):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(iters)
        for k in range(iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# -------------------------
# 3) Tiny GPT (Karpathyish)
# -------------------------
class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # causal mask up to block_size
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)           # (B,T,hs)
        q = self.query(x)         # (B,T,hs)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)         # (B,T,hs)
        out = wei @ v             # (B,T,hs)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa  = MultiHeadAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff  = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f  = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        x = tok_emb + pos_emb                      # (B,T,C)
        x = self.blocks(x)                         # (B,T,C)
        x = self.ln_f(x)                           # (B,T,C)
        logits = self.lm_head(x)                   # (B,T,V)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits.view(B, T, -1), loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # single-seq generation -> no padding needed
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --------------
# 4) Train LM
# --------------
model = GPTLanguageModel().to(device)
print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

model.train()
t0 = time.time()
for step in range(1, max_steps+1):
    xb, yb = get_batch('train')
    with torch.cuda.amp.autocast(enabled=use_amp):
        _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if step % eval_interval == 0 or step == 1:
        losses = estimate_loss(model)
        elapsed = time.time() - t0
        print(f"step {step:6d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {elapsed/60:.1f} min")

# save checkpoint
torch.save({"model": model.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer,
                "dropout": dropout, "block_size": block_size
            }}, ckpt_file)

# ------------------------------------------------------
# 5) Avg-NLL scorer (sliding window, no pad/masks needed)
# ------------------------------------------------------
@torch.no_grad()
def avg_nll_string(model, s: str, block=block_size):
    # Encode and compute mean negative log-likelihood across the string.
    ids = torch.tensor(encode_string(s), dtype=torch.long, device=device)
    T = ids.numel()
    if T <= 1:
        return 1e9  # degenerate, shouldn't happen with real sentences

    total_logprob, total_count = 0.0, 0
    start = 0
    while start < T - 1:
        end = min(T, start + block)
        x = ids[start:end]          # [L]
        if x.numel() <= 1:
            break
        inp = x[:-1].unsqueeze(0)   # [1, L-1]
        tgt = x[1:].unsqueeze(0)    # [1, L-1]
        logits, _ = model(inp, None)
        lp = F.log_softmax(logits, dim=-1)              # [1, L-1, V]
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [1, L-1]
        total_logprob += tok_lp.sum().item()
        total_count   += (end - start - 1)
        # move window with overlap (keep the last token as context)
        start = end - 1
    return - total_logprob / max(total_count, 1)

# ------------------------
# 6) Classify test pairs
# ------------------------
model.eval()  # important for stable logits/dropout off

print("Scoring test pairs and writing part1.txt ...")
with open(TEST_PATH, "r", encoding="utf-8") as fin, \
     open(OUT_PATH, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin, start=1):
        if '\t' not in line:
            fout.write("A\n")
            continue
        a, b = line.rstrip('\n').split('\t', 1)
        nll_a = avg_nll_string(model, a)
        nll_b = avg_nll_string(model, b)
        label = 'A' if nll_a < nll_b else 'B'
        fout.write(label + "\n")

        if i % 10_000 == 0:
            print(f"  processed {i:,} pairs...", flush=True)
            # optional: make sure progress is on disk too
            # fout.flush(); os.fsync(fout.fileno())

print(f"Done. Wrote {OUT_PATH}")