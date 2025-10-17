# generate.py
import torch
from pathlib import Path
from tokenizer import Tokenizer, PAD_ID
from model import GPTLanguageModel

SAVE_DIR = Path("checkpoints")
META_FILE= SAVE_DIR / "meta.pt"
CKPT_FILE= SAVE_DIR / "lm.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = Tokenizer.load(META_FILE)
ckpt = torch.load(CKPT_FILE, map_location=device)
cfg = ckpt["config"]
model = GPTLanguageModel(
    vocab_size=cfg["vocab_size"], block_size=cfg["block_size"],
    n_embd=cfg["n_embd"], n_head=cfg["n_head"], n_layer=cfg["n_layer"], dropout=cfg["dropout"]
).to(device)
model.load_state_dict(ckpt["model"]); model.eval()

@torch.no_grad()
def generate_text(prompt: str, max_new_tokens=200, temperature=1.0, top_k=None, top_p=None, seed=1337, stream=False):
    if seed is not None: torch.manual_seed(seed)
    ids = torch.tensor(tok.encode(prompt, add_bos_eos=True), dtype=torch.long, device=device).unsqueeze(0)
    block_size = cfg["block_size"]
    if stream: print(prompt, end='', flush=True)
    for _ in range(max_new_tokens):
        idx_cond = ids[:, -block_size:]
        logits, _ = model(idx_cond, None, ignore_index=PAD_ID)
        logits = logits[:, -1, :]
        if temperature != 1.0:
            logits = logits / max(1e-8, temperature)
        if top_k is not None and 0 < top_k < logits.size(-1):
            vals, inds = torch.topk(logits, top_k, dim=-1)
            mask = torch.full_like(logits, float('-inf')); mask.scatter_(1, inds, vals); logits = mask
        if top_p is not None and 0.0 < top_p < 1.0:
            s_log, s_idx = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(s_log, dim=-1); cum = torch.cumsum(probs, dim=-1)
            keep = cum <= top_p; keep[...,0] = True
            s_log[~keep] = float('-inf')
            logits = torch.full_like(logits, float('-inf')); logits.scatter_(1, s_idx, s_log)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
        if stream:
            print(tok.decode([next_id.item()]), end='', flush=True)
    if stream: print()
    return tok.decode(ids[0].tolist())

if __name__ == "__main__":
    print(generate_text("In conclusion, ", max_new_tokens=200, temperature=0.9, top_k=50, top_p=0.95))
