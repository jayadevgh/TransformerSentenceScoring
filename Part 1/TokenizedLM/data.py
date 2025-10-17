# data.py
import torch

def encode_lines_to_stream(lines, tokenizer, add_bos_eos=True):
    out=[]
    for s in lines:
        out.extend(tokenizer.encode(s, add_bos_eos=add_bos_eos))
    return torch.tensor(out, dtype=torch.long)

def get_batch(stream, batch_size, block_size, device):
    ix = torch.randint(len(stream) - block_size - 1, (batch_size,))
    x = torch.stack([stream[i:i+block_size] for i in ix])
    y = torch.stack([stream[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)
