import math, json
from pathlib import Path
import unicodedata
import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- tiny GPT blocks ----------
class Head(nn.Module):
    def __init__(self, n_embd, head_size, dropout, block_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x); q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout, block_size) for _ in range(n_head)])
        self.proj  = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(x))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa  = MultiHeadAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff  = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout, block_size):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, ignore_index=-100):
        B,T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            B,T,C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T), ignore_index=ignore_index)
        return logits, loss

# ---------- helpers for BPE ----------
def _build_raw_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a,b), nid in sorted(merges.items(), key=lambda kv: kv[1]):
        vocab[nid] = vocab[a] + vocab[b]
    return vocab

def _normalize(s):  # NFKC like your tokenizer
    return unicodedata.normalize("NFKC", s)

# ---------- loaders ----------

def _remap_old_names(sd):
    """Map old checkpoint param names -> current module names."""
    out = {}
    for k, v in sd.items():
        if k.startswith("token_embedding_table."):
            out["tok" + k[len("token_embedding_table"):]] = v
        elif k.startswith("position_embedding_table."):
            out["pos" + k[len("position_embedding_table"):]] = v
        elif k.startswith("lm_head."):
            out["head" + k[len("lm_head"):]] = v
        else:
            out[k] = v
    return out

def load_char_lm(ckpt_dir, device="cpu"):
    meta = torch.load(ckpt_dir / "meta.pt", map_location="cpu")
    itos = meta["itos"]; stoi = meta["stoi"]
    block_size = int(meta.get("block_size", 256))
    vocab_size = len(itos)

    ckpt = torch.load(ckpt_dir / "lm.pt", map_location="cpu")
    cfg  = ckpt["config"]
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=cfg["n_embd"], n_head=cfg["n_head"], n_layer=cfg["n_layer"],
        dropout=cfg["dropout"], block_size=cfg["block_size"]
    ).to(device)

    sd = _remap_old_names(ckpt["model"])
    has_bias = any(k.endswith("head.bias") for k in sd.keys())
    if not has_bias and model.head.bias is not None:
        model.head = nn.Linear(model.head.in_features, model.head.out_features, bias=False).to(device)
    model.load_state_dict(sd, strict=has_bias)
    model.eval()

    unk_id = stoi.get("<unk>", 0)
    def encode_char(s: str):
        return [stoi.get(c, unk_id) for c in s]
    return model, encode_char, block_size

def load_bpe_lm(ckpt_dir, device="cpu"):
    meta = torch.load(ckpt_dir / "meta.pt", map_location="cpu")

    merges        = meta["merges"]
    specials      = meta.get("special_ids", {"pad":0,"bos":1,"eos":2,"unk":3})
    NUM_SPECIALS  = int(meta.get("num_specials", 4))
    PAD_ID        = specials.get("pad", 0)
    BOS_ID        = specials.get("bos", 1)
    EOS_ID        = specials.get("eos", 2)
    UNK_ID        = specials.get("unk", 3)
    block_size    = int(meta.get("block_size", 256))
    regex_pattern = meta.get(
        "regex",
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    )
    gpt2pat = re.compile(regex_pattern)

    bpe_rank = {pair: rid for pair, rid in merges.items()}
    _raw_vocab_size = max([255, *merges.values()]) + 1  # not used externally here

    ckpt = torch.load(ckpt_dir / "lm.pt", map_location="cpu")
    cfg  = ckpt["config"]
    model = GPTLanguageModel(
        vocab_size=cfg["vocab_size"],
        n_embd=cfg["n_embd"], n_head=cfg["n_head"], n_layer=cfg["n_layer"],
        dropout=cfg["dropout"], block_size=cfg["block_size"]
    ).to(device)

    sd = _remap_old_names(ckpt["model"])
    has_bias = any(k.endswith("head.bias") for k in sd.keys())
    if not has_bias and model.head.bias is not None:
        model.head = nn.Linear(model.head.in_features, model.head.out_features, bias=False).to(device)
    model.load_state_dict(sd, strict=has_bias)
    model.eval()

    # --- minimal BPE encode, matching your tokenizer ---
    import unicodedata
    def _normalize(s: str) -> str:
        return unicodedata.normalize("NFKC", s)

    def _merge_once(ids, pair, new_id):
        out=[]; i=0; n=len(ids); a,b = pair
        while i<n:
            if i+1<n and ids[i]==a and ids[i+1]==b:
                out.append(new_id); i+=2
            else:
                out.append(ids[i]); i+=1
        return out

    def _bpe_encode_bytes(raw_ids):
        if len(raw_ids) <= 1: return raw_ids[:]
        ids = raw_ids[:]
        while True:
            best=None; best_rank=10**12
            for p in zip(ids, ids[1:]):
                r = bpe_rank.get(p)
                if r is not None and r < best_rank:
                    best_rank = r; best = p
            if best is None: break
            ids = _merge_once(ids, best, merges[best])
        return ids

    def encode_bpe(text: str, add_bos_eos=True):
        text = _normalize(text)
        raw=[]
        for piece in re.findall(gpt2pat, text):
            raw.extend(piece.encode("utf-8"))
        ids = [t + NUM_SPECIALS for t in _bpe_encode_bytes(raw)]
        if add_bos_eos:
            ids = [BOS_ID] + ids + [EOS_ID]
        return ids

    specials_pack = {"pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID, "unk": UNK_ID}
    return model, encode_bpe, block_size, specials_pack