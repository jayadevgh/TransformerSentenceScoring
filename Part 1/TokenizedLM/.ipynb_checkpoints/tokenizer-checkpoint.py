# tokenizer.py
import time, unicodedata, regex as re, torch
from pathlib import Path

# specials
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
NUM_SPECIALS = 4

# gpt-2 style regex
GPT2_PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def normalize_text(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def pretokenize(s: str):
    return re.findall(GPT2_PAT, s)

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

def learn_bpe(byte_ids, start_id=256, target_vocab=8192, min_pair_count=2,
              progress_every=None, name="BPE"):
    merges = {}; toks = byte_ids[:]; cur = start_id; t0 = time.time()
    total_merges = max(0, target_vocab - start_id)
    if progress_every is None:
        progress_every = max(50, total_merges // 20) if total_merges > 0 else 100
    print(f"[{name}] start: stream_len={len(toks):,} target_merges={total_merges:,}")
    while cur < target_vocab:
        counts = max_pairs(toks)
        if not counts:
            print(f"[{name}] no pairs left; stopping.")
            break
        (a,b), c = max(counts.items(), key=lambda kv: kv[1])
        if c < min_pair_count:
            print(f"[{name}] max pair freq {c} < min_pair_count {min_pair_count}; stopping.")
            break
        merges[(a,b)] = cur
        toks = merge_once(toks, (a,b), cur)
        cur += 1
        done = (cur - start_id)
        if (done % progress_every == 0) or (cur >= target_vocab):
            pct = (100.0 * done / max(1, total_merges)) if total_merges else 100.0
            elapsed = time.time() - t0
            print(f"[{name}] merges={done:6d}/{total_merges:<6d} ({pct:5.1f}%) "
                  f"max_pair_freq={c:<8d} stream_len={len(toks):,} elapsed={elapsed:6.1f}s",
                  flush=True)
    vocab_size = cur
    print(f"[{name}] done: total_merges={len(merges):,} raw_vocab_size={vocab_size} "
          f"time={time.time()-t0:.1f}s")
    return merges, vocab_size

def lines_to_byte_stream(lines):
    ids=[]
    for s in lines:
        s = normalize_text(s)
        for piece in pretokenize(s):
            ids.extend(piece.encode("utf-8"))
        ids.append(10)  # newline byte
    return ids

def build_raw_vocab(merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (a,b), nid in sorted(merges.items(), key=lambda kv: kv[1]):
        vocab[nid] = vocab[a] + vocab[b]
    return vocab

class Tokenizer:
    def __init__(self, merges, specials, regex_pattern, num_specials=NUM_SPECIALS):
        self.merges = merges
        self.bpe_rank = {pair: rid for pair, rid in merges.items()}
        self._raw_vocab = build_raw_vocab(merges)
        self.regex_pattern = regex_pattern
        self.num_specials = num_specials
        self.pad = specials["pad"]; self.bos = specials["bos"]
        self.eos = specials["eos"]; self.unk = specials["unk"]
        self.vocab_size = (max(self._raw_vocab.keys()) + 1) + num_specials

    @classmethod
    def train_from_lines(cls, lines, target_vocab=8192, min_pair_count=2, name="BPE"):
        stream = lines_to_byte_stream(lines)
        merges, raw_size = learn_bpe(
            stream, start_id=256, target_vocab=target_vocab - NUM_SPECIALS,
            min_pair_count=min_pair_count, name=name
        )
        specials = {"pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID, "unk": UNK_ID}
        return cls(merges, specials, GPT2_PAT.pattern, NUM_SPECIALS)

    def save(self, path: Path):
        torch.save({
            "merges": self.merges,
            "num_specials": self.num_specials,
            "special_ids": {"pad": self.pad, "bos": self.bos, "eos": self.eos, "unk": self.unk},
            "regex": self.regex_pattern
        }, path)

    @classmethod
    def load(cls, path: Path):
        meta = torch.load(path, map_location="cpu")
        merges = meta["merges"]
        specials = meta["special_ids"]
        num_specials = meta.get("num_specials", NUM_SPECIALS)
        regex_pattern = meta.get("regex", GPT2_PAT.pattern)
        return cls(merges, specials, regex_pattern, num_specials)

    def bpe_encode_bytes(self, raw_ids):
        if len(raw_ids) <= 1: return raw_ids[:]
        ids = raw_ids[:]
        while True:
            best=None; best_rank=10**12
            for p in zip(ids, ids[1:]):
                r = self.bpe_rank.get(p)
                if r is not None and r < best_rank:
                    best_rank = r; best = p
            if best is None: break
            ids = merge_once(ids, best, self.merges[best])
        return ids

    def encode(self, text: str, add_bos_eos=True):
        text = normalize_text(text)
        raw=[]
        for piece in re.findall(self.regex_pattern, text):
            raw.extend(piece.encode("utf-8"))
        ids = [t + self.num_specials for t in self.bpe_encode_bytes(raw)]
        if add_bos_eos:
            ids = [self.bos] + ids + [self.eos]
        return ids

    def decode(self, ids):
        ids = [i for i in ids if i not in (self.pad, self.bos, self.eos)]
        raw = [i - self.num_specials for i in ids]
        byts = b"".join(self._raw_vocab[i] for i in raw)
        return byts.decode("utf-8", errors="replace")
