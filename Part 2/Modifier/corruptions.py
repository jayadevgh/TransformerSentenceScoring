import random, re
from typing import List
from constants import MAX_EDITS

_letters = "abcdefghijklmnopqrstuvwxyz"
_fun_words = ["the","a","an","in","on","at","to","of","for","and","or","that","this","these","those","with","by","from","as","is","are","was","were","be","been","being"]
_fun_swaps = {
    "in":"on","on":"in","to":"of","of":"to","a":"the","the":"a","and":"or","or":"and",
    "is":"are","are":"is","was":"were","were":"was","this":"that","that":"this"
}

def _rand_letter_like(ch):
    if ch.isalpha():
        base = _letters.upper() if ch.isupper() else _letters
        return random.choice(base)
    return random.choice(_letters)

# 1) micro typos (swap/del/dup/replace)
def micro_typo(s: str) -> str:
    if not s: return s
    chars = list(s)
    edits = random.randint(1, MAX_EDITS)
    for _ in range(edits):
        if not chars: break
        op = random.choice(["swap","del","dup","rep"])
        i  = random.randrange(len(chars))
        if op == "swap" and i+1 < len(chars):
            chars[i], chars[i+1] = chars[i+1], chars[i]
        elif op == "del":
            del chars[i]
        elif op == "dup":
            chars.insert(i, chars[i])
        elif op == "rep":
            chars[i] = _rand_letter_like(chars[i])
    return "".join(chars)

# 2) punctuation / spacing tweaks
def punct_space(s: str) -> str:
    if not s: return s
    ops = []
    if "," in s or "." in s: ops.append("drop_punct")
    ops += ["add_comma","double_space","strip_space_before_punct"]
    op = random.choice(ops)

    if op == "drop_punct":
        out = re.sub(r"([,.;:!?])", "", s, count=1)
        return out if out != s else s + " "
    if op == "add_comma":
        tokens = s.split()
        if len(tokens) >= 3:
            j = random.randrange(1, len(tokens)-1)
            tokens[j] = tokens[j] + ","
            return " ".join(tokens)
        return s + ","
    if op == "double_space":
        return re.sub(r"\s+", "  ", s, count=1)
    if op == "strip_space_before_punct":
        return re.sub(r"\s+([,.;:!?])", r"\1", s, count=1)
    return s

# 3) function word tweak (delete/replace/move)
def function_word_tweak(s: str) -> str:
    words = s.split()
    if not words: return s
    idxs = [i for i,w in enumerate(words) if w.lower() in _fun_words]
    if not idxs:  # insert odd function word
        j = random.randrange(len(words))
        w = random.choice(["the","a","of","to"])
        words.insert(j, w)
        return " ".join(words)
    i = random.choice(idxs)
    w = words[i]
    op = random.choice(["delete","swap","replace"])
    if op == "delete":
        del words[i]
    elif op == "swap" and i+1 < len(words):
        words[i], words[i+1] = words[i+1], words[i]
    else:  # replace
        repl = _fun_swaps.get(w.lower(), random.choice(["and","or","of","to","in","on"]))
        words[i] = repl if w.islower() else repl.capitalize()
    return " ".join(words)

# 4) number tweak (digits preferred; else number words)
_num_word_map = {
    "zero":"one","one":"two","two":"three","three":"four","four":"five","five":"six","six":"seven",
    "seven":"eight","eight":"nine","nine":"ten","ten":"eleven"
}
def number_tweak(s: str) -> str:
    # digits first
    m = list(re.finditer(r"\d+", s))
    if m:
        span = random.choice(m).span()
        digits = list(s[span[0]:span[1]])
        k = random.randrange(len(digits))
        # mutate one digit (±1 mod 10)
        d = int(digits[k])
        digits[k] = str((d + random.choice([-1,1])) % 10)
        return s[:span[0]] + "".join(digits) + s[span[1]:]
    # word numbers
    words = s.split()
    for i,w in enumerate(words):
        lw = w.lower().strip(",.;:!?")
        if lw in _num_word_map:
            words[i] = _num_word_map[lw]
            return " ".join(words)
    # fallback: insert a spurious digit
    j = random.randrange(len(s)+1)
    return s[:j] + " 2 " + s[j:]

# 5) local re-order
def local_reorder(s: str) -> str:
    words = s.split()
    if len(words) < 2: return s
    i = random.randrange(len(words)-1)
    words[i], words[i+1] = words[i+1], words[i]
    return " ".join(words)

# export
FAMILIES = [micro_typo, punct_space, function_word_tweak, number_tweak, local_reorder]
