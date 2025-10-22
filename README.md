(Note, I advise you to look at the README.txt for each project for a more in-depth understanding of my work!)

## What my project does (Summary)

I train two lightweight decoder-only Transformers (its structure is taken from nanoGPT by Andrej Karpathy; an implementation of *Attention Is All You Need*), one at character level and one with a byte-pair-encoding tokenizer, and I use average negative log likelihood to judge which sentence looks more like fluent English. Then I synthesize a hard but subtle corruption for each original sentence so the classifier must pay attention to small grammar and phrasing cues rather than obvious noise.

---

### How I score sentences

Given a string, I compute mean NLL with a sliding window up to a fixed context of 256. Lower is better. Examples:

```text
avg_nll_string(model, "This is an English sentence.") ->  char ~1.10   bpe ~3.67
avg_nll_string(model, "Th1s is n0t gr8 Engl1sh!!!")   ->  char ~4.82   bpe ~8.18
```

Windows overlap so the last token of one window becomes context for the next, which keeps estimates smooth and avoids padding.

---

### Models in short

**Character LM** — I map characters to IDs with a small lookup and an `<unk>` escape for weird bytes. The model has 10.95 M params, a 256-token context, and I trained on the first sentences only. In a quick pass I saw 93.59 % accuracy on my internal training-style evaluation although I never properly made a training split. Known limits include no explicit BOS/EOS, short context, and heavy penalties when non-English bytes appear.
**BPE Token LM** — Same Transformer core but inputs are BPE tokens with `<pad>`, `<bos>`, `<eos>`, `<unk>`.
I run with
$n_{\text{embd}} = 384$  
$n_{\text{head}} = 6$  
$n_{\text{layer}} = 6$  
$dropout = 0.2$  
$block_{size} = 256 \\$
for about 12.31 M params.
Training schedule is a fixed 10 k steps with mixed precision and gradient accumulation.
A held-out internal check landed at 91.64 % accuracy with final val loss 4.49.

---

### Tokenization and BPE in practice

I keep tokenization clean-room by learning merges only from the A-side. Text is normalized with NFKC, lightly pre-tokenized with a GPT-2-style regex into words, numbers, punctuation, and whitespace, then I drop everything to UTF-8 bytes. I start from the 256-byte alphabet and repeatedly fuse the most frequent adjacent byte pair on A-side data until I learn about 3.8 k merges, which yields a raw vocab of 4092.

Encoding applies merges greedily by their learned order, shifts IDs by the number of specials, and can add `<bos>` or `<eos>`.
Decoding removes specials, unshifts, concatenates bytes, and decodes as UTF-8 with `errors="replace"`.
I cache merges, specials, regex, and block size in `checkpoints/meta.pt` so later runs load instantly.

---

### Training loop sketch

I build long 1 D streams per split (80/10/10 by original order).
Batches are contiguous windows where targets are inputs shifted by one, so all tokens contribute to loss and padding is unnecessary.
I optimize with AdamW at $\texttt{lr}=3\cdot10^{-4}$, mixed precision on CUDA, small micro-batches plus accumulation to stay within memory, and I log train/val/test losses at a fixed cadence.
I save `checkpoints/lm.pt` with weights and a minimal config so I can rebuild shapes safely.

---

### Part 2 builder

Goal is to write `part2.txt` with pairs `A\tB` where $B$ is a corruption of $A$ that is barely worse.
For each input line I alternate the scoring LM (even lines char, odd lines BPE), compute NLL on $A$, sample several candidate edits, and accept the smallest positive $\Delta = \text{NLL}(B) - \text{NLL}(A)$ that clears a tiny floor.
I reject exact duplicates and matches to any gold $B$ from training, and if nothing survives after a few tries I fall back to a deterministic change so progress continues.
I append and flush each pair and can `fsync` so the job can resume safely.

I aim for a small bruise rather than a rewrite, so my $\Delta$ values sit in a narrow band near zero and almost never cross below it.
In a 30 k sample I saw only 2 pairs with $\Delta \le 0$.
The training file shows a much wider spread with heavy outliers in both directions and 2 211 pairs where $B$ beats $A$, which suggests stronger noise and some misalignments or rare → common substitutions that a next-token model likes statistically.
My style better matches the task “make a corrupted version of the previous sentence” because meaning stays intact and the decision depends on agreement, local syntax, and punctuation fit.

---

### Corruption families I use

**Micro typos** — I apply a tiny number of character-level edits like swap, delete, duplicate, or replace with case-aware letters so the token shape stays familiar and NLL rises a little.
**Punctuation and spacing** — I remove or insert an interior comma, double one whitespace span, or strip a space before punctuation. I avoid end-of-sentence fixes that can improve NLL.
**Function words** — I delete, swap across a neighbor, or replace small glue words like articles, prepositions, or simple auxiliaries using a short map of plausible but wrong choices, which nudges agreement or selection.
**Numbers** — I flip a single digit in a detected number or step a simple number word up or down. If none exist I may inject a small stray digit.
**Local reorder** — I swap one adjacent pair of words to disturb short-range syntax while keeping bag-of-words intact.
Across families I cap edits with `MAX_EDITS` and run an acceptance loop that targets a small positive $\Delta$, which keeps difficulty high but fair.
