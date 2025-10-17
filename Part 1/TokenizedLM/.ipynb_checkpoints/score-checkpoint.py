# score.py
import torch
from pathlib import Path
from tokenizer import Tokenizer, PAD_ID, BOS_ID, EOS_ID
from model import GPTLanguageModel

TRAIN_PATH = "train_utf8.txt"
TEST_PATH  = "test_utf8.rand.txt"
SAVE_DIR   = Path("checkpoints"); SAVE_DIR.mkdir(parents=True, exist_ok=True)
META_FILE  = SAVE_DIR / "meta.pt"
CKPT_FILE  = SAVE_DIR / "lm.pt"
OUT_PATH   = SAVE_DIR / "part1.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"

# load tokenizer + model
tok = Tokenizer.load(META_FILE)
ckpt = torch.load(CKPT_FILE, map_location=device)
cfg = ckpt["config"]
model = GPTLanguageModel(
    vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
    n_embd=cfg["n_embd"], n_head=cfg["n_head"], n_layer=cfg["n_layer"], dropout=cfg["dropout"]
).to(device)
model.load_state_dict(ckpt["model"]); model.eval()
print("[loaded] tokenizer+model")

@torch.no_grad()
def avg_nll_string(model, s: str, block=None):
    if block is None: block = cfg["block_size"]
    ids = torch.tensor(tok.encode(s, add_bos_eos=True), dtype=torch.long, device=device)
    T = ids.numel()
    if T <= 1: return 1e9
    total_logprob, total_count = 0.0, 0
    start = 0
    while start < T-1:
        end = min(T, start + block)
        x = ids[start:end]
        if x.numel() <= 1: break
        inp = x[:-1].unsqueeze(0)
        tgt = x[1:].unsqueeze(0)
        logits, _ = model(inp, None, ignore_index=PAD_ID)
        lp = torch.log_softmax(logits, dim=-1)
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        total_logprob += tok_lp.sum().item()
        total_count   += (end - start - 1)
        start = end - 1
    return - total_logprob / max(total_count, 1)

# write part1.txt
print("[classify] scoring test set and writing", OUT_PATH)
with open(TEST_PATH, "r", encoding="utf-8") as fin, open(OUT_PATH, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin, 1):
        if "\t" not in line:
            fout.write("A\n"); continue
        a, b = line.rstrip("\n").split("\t", 1)
        nlla = avg_nll_string(model, a); nllb = avg_nll_string(model, b)
        fout.write(("A" if nlla < nllb else "B") + "\n")
        if i % 10_000 == 0:
            print(f"  processed {i:,} pairs...", flush=True)
print("[done]", OUT_PATH)
