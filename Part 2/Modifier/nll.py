import torch
import torch.nn.functional as F

def avg_nll_from_ids(model, ids, block_size, device):
    T = len(ids)
    if T <= 1: return 1e9
    total_logprob = 0.0
    count = 0
    pos = 0
    while pos < T - 1:
        end = min(T, pos + block_size)
        x = torch.tensor(ids[pos:end-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(ids[pos+1:end], dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(x, None)
        lp = F.log_softmax(logits, dim=-1)
        total_logprob += lp.gather(-1, y.unsqueeze(-1)).squeeze(-1).sum().item()
        count += (end - pos - 1)
        pos = end - 1
    return - total_logprob / max(count, 1)

def make_char_nll(model, encode_char, block_size, device):
    return lambda s: avg_nll_from_ids(model, encode_char(s), block_size, device)

def make_bpe_nll(model, encode_bpe, block_size, device):
    return lambda s: avg_nll_from_ids(model, encode_bpe(s, add_bos_eos=True), block_size, device)
