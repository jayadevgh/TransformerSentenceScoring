# ==== Reload trained char-GPT from lm.pt + meta.pt ====
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

# --- choose where your files live ---
SAVE_DIRS = [Path("artifacts"), Path("checkpoints"), Path("/notebooks/checkpoints")]
for _d in SAVE_DIRS:
    if (_d / "lm.pt").exists() and (_d / "meta.pt").exists():
        SAVE_DIR = _d
        break
else:
    raise FileNotFoundError("Couldn't find lm.pt and meta.pt in artifacts/ or checkpoints/")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading from:", SAVE_DIR.resolve())

# --- load meta & checkpoint ---
meta = torch.load(SAVE_DIR / "meta.pt", map_location="cpu")
ckpt = torch.load(SAVE_DIR / "lm.pt",   map_location="cpu")

itos        = meta["itos"]
stoi        = meta["stoi"]
block_size  = meta["block_size"]
vocab_size  = len(itos)

cfg      = ckpt["config"]
n_embd   = cfg["n_embd"]
n_head   = cfg["n_head"]
n_layer  = cfg["n_layer"]
dropout  = cfg["dropout"]

# --- model definition (matches your training code) ---
class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2,-1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), nn.GELU(),
            nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout)
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
    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        return logits, loss

# --- rebuild & load weights ---
model = GPTLanguageModel().to(device)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()
print("Model reloaded ✅  | params:", sum(p.numel() for p in model.parameters())/1e6, "M")

# --- helpers you can use immediately ---

def decode_ids(ids): return ''.join(itos[i] for i in ids)
def encode_string(s: str): return [stoi.get(c, stoi.get("<unk>", 0)) for c in s]

@torch.no_grad()
def avg_nll_string(model, s: str, block: int = block_size) -> float:
    ids = torch.tensor(encode_string(s), dtype=torch.long, device=device)
    T = ids.numel()
    if T <= 1: return 1e9
    total_logprob = 0.0; total_count = 0
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

@torch.no_grad()
def generate_text(prompt: str, max_new_tokens=300, temperature=1.0, top_k=None, top_p=None, seed=1337, stream=False):
    # Works with (logits, loss) forward
    if seed is not None: torch.manual_seed(seed)
    UNK = stoi.get("<unk>", None)
    def enc(c): return stoi.get(c, UNK if UNK is not None else next(iter(stoi.values())))
    ctx = torch.tensor([enc(c) for c in prompt], dtype=torch.long, device=device).unsqueeze(0)
    if ctx.numel() == 0:
        ctx = torch.tensor([[enc('\n' if '\n' in stoi else ' ')]], dtype=torch.long, device=device)
    out_ids = ctx.clone()
    if stream: print(prompt, end='', flush=True)
    for _ in range(max_new_tokens):
        idx_cond = out_ids[:, -block_size:]
        logits, _ = model(idx_cond, None)
        logits = logits[:, -1, :]
        if temperature != 1.0: logits = logits / max(1e-8, temperature)
        if top_k is not None and 0 < top_k < logits.size(-1):
            vals, inds = torch.topk(logits, top_k, dim=-1)
            tmp = torch.full_like(logits, float('-inf')); tmp.scatter_(1, inds, vals); logits = tmp
        if top_p is not None and 0.0 < top_p < 1.0:
            s_log, s_idx = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(s_log, dim=-1); cum = torch.cumsum(probs, dim=-1)
            keep = cum <= top_p; keep[..., 0] = True
            s_log[~keep] = float('-inf')
            logits = torch.full_like(logits, float('-inf')); logits.scatter_(1, s_idx, s_log)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        out_ids = torch.cat([out_ids, next_id], dim=1)
        if stream: print(itos[next_id.item()], end='', flush=True)
    if stream: print()
    return decode_ids(out_ids[0].tolist())

# quick smoke test (optional):
# print(avg_nll_string(model, "This is a perfectly normal English sentence."))
# print(generate_text("In conclusion, ", max_new_tokens=200, temperature=0.9, top_k=50, top_p=0.95))
