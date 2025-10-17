# model.py
# ============================================================
#  Is it English? — GPT (BPE tokenizer, BOS/EOS, long context)
#
#  PART 0: Config & splits (train/val/test)
#  PART 1: Tokenizer (byte-level BPE + GPT-2 regex + NFKC)
#  PART 2: Streams & batching (no padding)
#  PART 3: Transformer architecture (decoder-only GPT)
#  PART 4: Training loop & checkpointing
#  PART 5: Avg-NLL scorer (sliding window)
#  PART 6: Generation (temperature/top-k/top-p)
#  PART 7: Classify test pairs -> part1.txt
# ============================================================

import os, time, random, unicodedata, regex as re
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------
# PART 0: Config & splits
# --------------------
TRAIN_PATH = "train_utf8.txt"        # A \t B per line
TEST_PATH  = "test_utf8.rand.txt"    # randomized pairs
SAVE_DIR   = Path("checkpoints"); SAVE_DIR.mkdir(parents=True, exist_ok=True)

# output files
meta_file  = SAVE_DIR / "meta.pt"    # tokenizer, merges, specials, block_size
ckpt_file  = SAVE_DIR / "lm.pt"      # model weights+config
OUT_PATH   = SAVE_DIR / "part1.txt"  # submission file

# Model / training (you can tweak)
batch_size    = 64
block_size    = 512      # ↑ longer context than 256
max_steps     = 20_000
eval_interval = 1000
learning_rate = 3e-4
n_embd        = 512
n_head        = 8
n_layer       = 8
dropout       = 0.2
seed          = 1337

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(seed); random.seed(seed)
use_amp = (device == "cuda")

# Special tokens (reserved ids)
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
NUM_SPECIALS = 4

# GPT-2 style regex pretokenizer
gpt2pat = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

# Build A-side lines and split 80/10/10 by line order
A_lines = []
with open(TRAIN_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if "\t" not in line: 
            continue
        a, _ = line.rstrip("\n").split("\t", 1)
        A_lines.append(a)

N = len(A_lines)
n_train = int(0.8 * N)
n_val   = int(0.1 * N)
train_lines = A_lines[:n_train]
val_lines   = A_lines[n_train:n_train+n_val]
test_lines  = A_lines[n_train+n_val:]
print(f"[splits] A-lines={N:,} | train={len(train_lines):,} | val={len(val_lines):,} | test={len(test_lines):,}")

# --------------------
# PART 1: Tokenizer (byte-level BPE + GPT-2 regex + NFKC)
# --------------------
BPE_VOCAB_TARGET = 8192  # total target vocab (bytes+merges) + we reserve NUM_SPECIALS

def normalize_text(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def pretokenize(s: str):
    return re.findall(gpt2pat, s)

def max_pairs(tokens):
    counts = {}
    for p in zip(tokens, tokens[1:]):
        counts[p] = counts.get(p, 0) + 1
    return counts

def merge_once(tokens, pair, new_id):
    out=[]; i=0; n=len(tokens); a,b = pair
    while i<n:
        if i+1<n and tokens[i]==a and tokens[i+1]==b:
            out.append(new_id); i+=2
        else:
            out.append(tokens[i]); i+=1
    return out

def learn_bpe(byte_ids, start_id=256, target_vocab=8192, min_pair_count=2):
    merges={}; cur=start_id; toks=byte_ids[:]
    while cur < target_vocab:
        counts = max_pairs(toks)
        if not counts: break
        (a,b), c = max(counts.items(), key=lambda kv: kv[1])
        if c < min_pair_count: break
        merges[(a,b)] = cur
        toks = merge_once(toks, (a,b), cur)
        cur += 1
    return merges, cur  # merges, final raw vocab size (0..cur-1 inclusive)

def lines_to_byte_stream(lines):
    ids=[]
    for s in lines:
        s = normalize_text(s)
        for piece in pretokenize(s):
            ids.extend(piece.encode("utf-8"))
        ids.append(10)  # newline byte separator
    return ids

def build_raw_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a,b), nid in sorted(merges.items(), key=lambda kv: kv[1]):
        vocab[nid] = vocab[a] + vocab[b]
    return vocab

# Load tokenizer from meta.pt if present; else learn on train split and save
if meta_file.exists():
    meta = torch.load(meta_file, map_location="cpu")
    merges      = meta["merges"]
    NUM_SPECIALS= meta.get("num_specials", NUM_SPECIALS)
    PAD_ID      = meta["special_ids"]["pad"]
    BOS_ID      = meta["special_ids"]["bos"]
    EOS_ID      = meta["special_ids"]["eos"]
    UNK_ID      = meta["special_ids"]["unk"]
    block_size  = meta.get("block_size", block_size)
    bpe_rank    = {pair: rid for pair, rid in merges.items()}
    _raw_vocab  = build_raw_vocab(merges)
    raw_vocab_size = max(_raw_vocab.keys()) + 1
    print("[tokenizer] loaded from meta.pt")
else:
    # Learn merges on train split only
    train_bytes = lines_to_byte_stream(train_lines)
    merges, raw_vocab_size = learn_bpe(train_bytes, start_id=256, target_vocab=BPE_VOCAB_TARGET - NUM_SPECIALS)
    bpe_rank   = {pair: rid for pair, rid in merges.items()}
    _raw_vocab = build_raw_vocab(merges)
    torch.save({
        "merges": merges,
        "num_specials": NUM_SPECIALS,
        "special_ids": {"pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID, "unk": UNK_ID},
        "block_size": block_size,
        "regex": gpt2pat.pattern,
    }, meta_file)
    print("[tokenizer] learned merges and saved meta.pt")

# Greedy BPE encode using learned ranks (GPT-2 style)
def bpe_encode_bytes(raw_ids):
    if len(raw_ids) <= 1: return raw_ids[:]
    ids = raw_ids[:]
    while True:
        best=None; best_rank=10**12
        for p in zip(ids, ids[1:]):
            r = bpe_rank.get(p)
            if r is not None and r < best_rank:
                best_rank = r; best = p
        if best is None: break
        ids = merge_once(ids, best, merges[best])
    return ids

def encode_bpe(text: str, add_bos_eos=True):
    text = normalize_text(text)
    raw=[]
    for piece in pretokenize(text):
        raw.extend(piece.encode("utf-8"))
    ids = [t + NUM_SPECIALS for t in bpe_encode_bytes(raw)]
    if add_bos_eos:
        ids = [BOS_ID] + ids + [EOS_ID]
    return ids

def decode_bpe(ids):
    ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
    raw = [i - NUM_SPECIALS for i in ids]
    byts = b"".join(_raw_vocab[i] for i in raw)
    return byts.decode("utf-8", errors="replace")

# keep your original helper names:
encode_string = encode_bpe
decode_ids    = decode_bpe
vocab_size    = raw_vocab_size + NUM_SPECIALS
print(f"[tokenizer] vocab_size={vocab_size}  (raw={raw_vocab_size} + specials={NUM_SPECIALS})")

# --------------------
# PART 2: Streams & batching (no padding)
# --------------------
def encode_lines_to_stream(lines):
    out=[]
    for s in lines:
        out.extend(encode_string(s, add_bos_eos=True))
    return torch.tensor(out, dtype=torch.long)

train_data = encode_lines_to_stream(train_lines)
val_data   = encode_lines_to_stream(val_lines)
test_data  = encode_lines_to_stream(test_lines)
print("[streams] lens:", len(train_data), len(val_data), len(test_data))

def get_batch(split):
    source = {"train": train_data, "val": val_data, "test": test_data}[split]
    ix = torch.randint(len(source) - block_size - 1, (batch_size,))
    x = torch.stack([source[i:i+block_size] for i in ix])
    y = torch.stack([source[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, iters=200):
    model.eval()
    out={}
    for split in ["train", "val", "test"]:
        losses = torch.zeros(iters)
        for _ in range(iters):
            X,Y = get_batch(split)
            _, loss = model(X, Y)
            losses[_] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# --------------------
# PART 3: Transformer architecture (decoder-only GPT)
# --------------------
class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
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
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)
        # weight tying
        self.lm_head.weight = self.token_embedding_table.weight

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
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                   # (B,T,V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, vocab_size), targets.view(B*T), ignore_index=PAD_ID)
        return logits, loss

model = GPTLanguageModel().to(device)
print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M | vocab={vocab_size} | block={block_size}")

# --------------------
# PART 4: Training loop & checkpointing
# --------------------
def train_model():
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    t0 = time.time()

    model.train()
    for step in range(1, max_steps+1):
        xb, yb = get_batch('train')
        with torch.cuda.amp.autocast(enabled=use_amp):
            _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()

        if step % eval_interval == 0 or step == 1:
            stats = estimate_loss(model, iters=200)
            mins = (time.time() - t0)/60
            print(f"step {step:6d} | train {stats['train']:.4f} | val {stats['val']:.4f} | test {stats['test']:.4f} | {mins:.1f} min")

    torch.save({
        "model": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer,
            "dropout": dropout, "block_size": block_size
        }
    }, ckpt_file)
    print("[ckpt] saved:", ckpt_file)

# --------------------
# PART 5: Avg-NLL scorer (sliding window)
# --------------------
@torch.no_grad()
def avg_nll_string(model, s: str, block=block_size):
    ids = torch.tensor(encode_string(s, add_bos_eos=True), dtype=torch.long, device=device)
    T = ids.numel()
    if T <= 1: return 1e9
    total_logprob, total_count = 0.0, 0
    start = 0
    while start < T - 1:
        end = min(T, start + block)
        x = ids[start:end]
        if x.numel() <= 1: break
        inp = x[:-1].unsqueeze(0)
        tgt = x[1:].unsqueeze(0)
        logits, _ = model(inp, None)
        lp = F.log_softmax(logits, dim=-1)
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        total_logprob += tok_lp.sum().item()
        total_count   += (end - start - 1)
        start = end - 1
    return - total_logprob / max(total_count, 1)

# --------------------
# PART 6: Generation (temperature/top-k/top-p)
# --------------------
@torch.no_grad()
def generate_text(prompt: str, max_new_tokens=200, temperature=1.0, top_k=None, top_p=None, seed=1337, stream=False):
    if seed is not None: torch.manual_seed(seed)
    model.eval()
    ids = torch.tensor(encode_string(prompt, add_bos_eos=True), dtype=torch.long, device=device).unsqueeze(0)
    if stream: print(prompt, end='', flush=True)
    for _ in range(max_new_tokens):
        idx_cond = ids[:, -block_size:]
        logits, _ = model(idx_cond, None)
        logits = logits[:, -1, :]
        if temperature != 1.0:
            logits = logits / max(1e-8, temperature)
        if top_k is not None and 0 < top_k < logits.size(-1):
            vals, inds = torch.topk(logits, top_k, dim=-1)
            mask = torch.full_like(logits, float('-inf')); mask.scatter_(1, inds, vals); logits = mask
        if top_p is not None and 0.0 < top_p < 1.0:
            s_log, s_idx = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(s_log, dim=-1); cum = torch.cumsum(probs, dim=-1)
            keep = cum <= top_p; keep[...,0] = True
            s_log[~keep] = float('-inf')
            logits = torch.full_like(logits, float('-inf')); logits.scatter_(1, s_idx, s_log)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
        if stream:
            print(decode_ids([next_id.item()]), end='', flush=True)
    if stream: print()
    return decode_ids(ids[0].tolist())

# --------------------
# PART 7: Classify test pairs -> part1.txt
# --------------------
def write_part1():
    model.eval()
    print("[classify] scoring test pairs and writing", OUT_PATH)
    with open(TEST_PATH, "r", encoding="utf-8") as fin, open(OUT_PATH, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            if "\t" not in line:
                fout.write("A\n"); continue
            a, b = line.rstrip("\n").split("\t", 1)
            nll_a = avg_nll_string(model, a)
            nll_b = avg_nll_string(model, b)
            fout.write(("A" if nll_a < nll_b else "B") + "\n")
            if i % 10_000 == 0:
                print(f"  processed {i:,} pairs...", flush=True)
    print("[classify] done:", OUT_PATH)

# --------------------
# Optional main routine
# --------------------
if __name__ == "__main__":
    # Train:
    train_model()

    # Example generation:
    # print(generate_text("In conclusion, ", max_new_tokens=200, temperature=0.9, top_k=50, top_p=0.95))

    # Write predictions file for the assignment:
    write_part1()
