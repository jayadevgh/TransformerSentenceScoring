# part2.py
import os, time, random, re
from pathlib import Path

from constants import (
    TRAIN_PATHS, OUT_PATH, PRINT_EVERY,
    MAX_TRIES_PER_LINE, DEVICE, CHAR_CKPT_DIR, BPE_CKPT_DIR, MAX_LEN_CHARS
)
from loaders import load_char_lm, load_bpe_lm
from nll import make_char_nll, make_bpe_nll
from corruptions import FAMILIES


def _pick_train_path():
    for p in TRAIN_PATHS:
        if Path(p).exists():
            return p
    raise FileNotFoundError("train_utf8.txt or train.txt not found in CWD.")


def _count_output_pairs(path: Path) -> int:
    if not path.exists():
        return 0
    # Count only lines that look like valid A\tB pairs (robust to any junk)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return sum(1 for ln in f if "\t" in ln)


def main():
    train_path = _pick_train_path()

    # Ensure output dir exists and compute resume count
    out_path = Path(OUT_PATH).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already = _count_output_pairs(out_path)
    print(f"[resume] output → {out_path}")
    print(f"[resume] lines already present: {already:,}")

    # Load models + scorers
    print("[load] char LM …")
    ch_model, ch_encode, ch_block = load_char_lm(CHAR_CKPT_DIR, device=DEVICE)
    ch_model.eval()
    char_nll = make_char_nll(ch_model, ch_encode, ch_block, DEVICE)

    print("[load] BPE  LM …")
    bp_model, bp_encode, bp_block, _specials = load_bpe_lm(BPE_CKPT_DIR, device=DEVICE)
    bp_model.eval()
    bpe_nll = make_bpe_nll(bp_model, bp_encode, bp_block, DEVICE)

    # Append mode; line-buffered so each write flushes a lot, plus fsync below
    out = open(out_path, "a", encoding="utf-8", buffering=1, newline="")
    t0 = time.time()
    wrote = already  # keeps even/odd alternation stable across resumes
    pairs_seen = 0   # counts ONLY lines with a tab in the input

    print(f"[build] reading A-sentences from {train_path}")
    with open(train_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.rstrip("\n")
            a, sep, b = s.partition("\t")
            if sep != "\t":
                # not a valid pair; don't advance pairs_seen/wrote
                continue

            # resume: skip until we've matched the number of pairs already written
            if pairs_seen < already:
                pairs_seen += 1
                continue
            pairs_seen += 1

            # Only truncate AFTER split so we don't chop away the tab
            A = a[:MAX_LEN_CHARS]
            B_gold = b  # to avoid accidentally duplicating gold

            # Alternate scorer: even -> char, odd -> bpe (based on output index)
            use_char = ((wrote + 1) % 2 == 0)
            base_nll = (char_nll if use_char else bpe_nll)(A)

            best_delta = None
            best_B = None
            tried = set()
            families = FAMILIES[:]  # copy & shuffle
            random.shuffle(families)
            attempts = 0

            for fam in families:
                if attempts >= MAX_TRIES_PER_LINE:
                    break
                B = fam(A)
                if not B:
                    continue
                B = B[:MAX_LEN_CHARS]  # keep length sane
                if B == A or B == B_gold:
                    continue  # must differ from A and from the real corruption
                if not re.search(r"\w", B):
                    continue  # keep some alnum content
                if B in tried:
                    continue
                tried.add(B)
                attempts += 1

                nllB = (char_nll if use_char else bpe_nll)(B)
                delta = nllB - base_nll  # >0 means "harder than A"
                if delta <= 0:
                    continue
                if (best_delta is None) or (delta < best_delta):
                    best_delta = delta
                    best_B = B

            if best_B is None:
                # deterministic tiny tweak fallback that avoids matching gold/A
                B = A + " ."
                if B == B_gold or B == A:
                    B = A + " .."
            else:
                B = best_B

            out.write(A + "\t" + B + "\n")
            wrote += 1

            if wrote % PRINT_EVERY == 0:
                elapsed = time.time() - t0
                print(f"[{wrote:,}] lines written  |  elapsed={elapsed/60:.1f} min", flush=True)
                try:
                    out.flush()
                    os.fsync(out.fileno())
                except Exception:
                    pass

    try:
        out.flush()
        os.fsync(out.fileno())
    except Exception:
        pass
    out.close()
    print(f"[done] wrote {out_path}  (total lines: {wrote:,})")


if __name__ == "__main__":
    main()
