NOTICE: BEFORE READING THIS README, PLEASE GO OVER THE CHARACTERLEVELLM FIRST. THANK YOU.

This is a BPE-tokenized language model (decoder-only Transformer). Like the char LM, we score each pair by average negative log-likelihood (lower NLL ⇒ “more English”) and pick A or B accordingly. The big change is tokenization: GPT-2–style pretokenization → UTF-8 bytes → BPE merges learned only from the A-side.

Here’s a quick NLL check (per BPE token, not per byte):
avg_nll_string(model, "This is an English sentence.")   → ~ 3.67
avg_nll_string(model, "Th1s is n0t gr8 Engl1sh!!!")     → ~ 8.18

Lower is better, as before.

---------------------------------------------------------------------------------------------------------------------

STATS:
Parameters: 12.31M
Tokenizer (raw): 4092 (bytes+merges) + specials (<pad>, <bos>, <eos>, <unk>)
Context window: 256 tokens
A-side stream lengths (example run): train ≈ 28.60M, val ≈ 3.58M, test ≈ 3.58M tokens
Training schedule: 10,000 steps (fixed; no early stopping)
Final val loss: 4.49
Held-out internal test (gold=A, 20k sampled from last 10%): 91.64% accuracy
Artifacts: checkpoints/meta.pt (tokenizer), checkpoints/lm.pt (weights+config), checkpoints/part1.txt

ISSUES & LESSONS LEARNED
- OOM pain (GPU): Batch 64 blew up; settled on small micro-batch + grad accumulation and block_size=256.
- Tokenizer time: Learning ~3.8k merges on 50k A-lines took a while; added progress prints and cache to meta.pt.
- 20k steps was too far: Loss started to creep up; 10k gave the best validation & accuracy. (I didn’t use early stopping in this run.)
- Per-byte NLL not used: I stuck to per-BPE-token NLL; byte-level NLL could help more with unicode/typo tricks later.
- Char LM vs BPE: The char LM previously edged this on my quick check; BPE should win with bigger vocab/context/steps or early-stop.
- Long-range cues: Still limited by 256 context; bigger windows cost more memory and my gpus are running out of it.

---------------------------------------------------------------------------------------------------------------------

Here is are some examples of the model making an incorrect prediction:
Top mistakes (model preferred B):

#1 margin=-1.5118 (nll_a=4.5390, nll_b=3.0271)
A: I welcome the Garosci report.
B: I welcome the Greece report.

#2 margin=-1.2611 (nll_a=4.3502, nll_b=3.0892)
A: It is not a ‘Lamfalussy’ directive.
B: It is not a integrated directive.

#3 margin=-1.1642 (nll_a=5.6996, nll_b=4.5354)
A: Road Transport
B: On Transport

#4 margin=-1.1146 (nll_a=5.5550, nll_b=4.4404)
A: everything stops yeah
B: everythings tops yeah

#5 margin=-0.8877 (nll_a=5.1521, nll_b=4.2644)
A: Click again.
B: Crick again.

NOW LETS GO OVER EACH FILE: 
---------------------------------------------------------------------------------------------------------------------

Here is a "Table of Contents":
TokenizedLM/
├─ tokenizer.py      # train/load BPE (with progress), encode/decode
├─ model.py          # transformer blocks + GPTLanguageModel
├─ data.py           # load A-side, split 80/10/10, build streams, batching
├─ train.py          # trains the LM, logs, saves lm.pt
├─ score.py          # avg_nll_string + write part1.txt
├─ generate.py       # generate_text sampler (temp/top-k/top-p)
└─ checkpoints/      # meta.pt (tokenizer), lm.pt (weights), part1.txt

---------------------------------------------------------------------------------------------------------------------

File 1: tokenizer.py - Merging vocabulary (BPE)

I treat the tokenizer as a clean room that never learns from anything but the A side. I normalize text with NFKC, split it with a GPT-2 style regex into word pieces, numbers, punctuation, and whitespace, then I convert the pieces to raw UTF-8 bytes so odd characters do not crash anything. I learn BPE merges over bytes rather than code points, which keeps behavior stable, and if something truly strange shows up it falls to <unk>. 

In this setup I start with a tiny alphabet that is just the 256 byte values, then during training I look at the A-side byte stream and count which adjacent byte pairs show up the most, and I keep fusing the most frequent pair into a new symbol, then I recount and fuse again, and I repeat this until I have learned about 3.8k merges so the raw vocab reaches 4092. The merge list is ordered, so at encode time I take the input bytes and apply those merges greedily by priority, which means common chunks like the bytes for “th” or “ing” collapse into single tokens, and rarer stuff stays as smaller pieces so I do not invent tokens that the model never saw. Because everything happens at the byte level, I do not risk breaking on odd Unicode and I only fall back to <unk> when something is truly un-decodable, and because I learned the merges only from A-side text I keep the boundary clean while still getting longer tokens for frequent patterns.

On a fresh run when checkpoints/meta.pt is missing I sample fifty thousand A lines, build a byte stream, learn about 3.8k merges, and end up with a raw vocab of 4092 while printing progress so I can see the merge count and elapsed time. I save merges, special tokens, the regex pattern, and the current block size in meta.pt so the next run loads instantly. Encoding walks forward through normalize, regex chunks, bytes, greedy merges, and an ID shift by the number of specials with optional <bos> and <eos>. Decoding strips specials, removes the offset, joins bytes, and decodes with UTF-8 using replace on errors. The idea is the same motivation as the character-level writeup, only I push everything to the byte layer to be robust and to keep the training boundary strict.

---------------------------------------------------------------------------------------------------------------------

File 2: model.py - Same GPT core, new token IDs

The model keeps the same decoder-only Transformer that I used before with masked self-attention, LayerNorm, residuals, and an MLP. I only changed the inputs to come from the BPE tokenizer with <bos>, <eos>, <pad>, and <unk>. To avoid out-of-memory I set n_embd=384, n_head=6, n_layer=6, dropout=0.2, and block_size=256, which is about 12.31 million parameters. I sum token and positional embeddings, run the stack, apply a final norm, and project to vocabulary logits, and the forward call returns logits and an optional loss that can ignore PAD with ignore_index. 

I added temperature, top-k, and top-p in generate so I can do quick qualitative checks without extra code. The attention and residual intuition from the char model still applies since I did not change the core architecture. All the intuition about attention heads, LayerNorm placement, residual paths, and MLP expansion is exactly what I already covered in the char-level README. This file simply plugs a different tokenizer in front of the same proven architecture.

---------------------------------------------------------------------------------------------------------------------

File 3: data.py - A-side streams, simple blocks

I tokenize only the A side to respect the data boundary, split by original order into 80, 10, and 10 percent for train, validation, and test, then I pack each split into a single long stream of token ids. Batches are sliding windows where inputs are the tokens in the window and targets are the same window shifted by one. Every token contributes to the loss which means no padding and good throughput. I do not add special bucketing or sentence rules because this task does not need them and the flow matches what I did for the char model.

---------------------------------------------------------------------------------------------------------------------

File 4: train.py - Train, log, save

This script wires the tokenizer, the streams, and the model, and it trains in a way that fits a tight GPU. I use AdamW with a learning rate of 3e-4, I enable mixed precision on CUDA, and I log train, validation, and test losses every evaluation interval so I can watch how things move. I do not use early stopping here because I found that ten thousand steps landed at a better generalization point than twenty thousand, so I fix the schedule at ten thousand. To avoid OOM I support small micro-batches with gradient accumulation and I keep block_size=256 which stabilizes memory use. At the end I save checkpoints/lm.pt with the weights and a small config, and I rely on checkpoints/meta.pt for merges and special ids so reloads are consistent. If you need the long-form rationale for optimizer choice, AMP, or the evaluate-log cadence, that’s already explained in your previous README. Here the main differences are the BPE IDs at the input, the fixed 10k schedule, and the OOM-friendly setup.

---------------------------------------------------------------------------------------------------------------------

File 5: score.py — Make part1.txt

This script reads the official test pairs and writes part1.txt. For each line I split into A and B on the tab, compute the average negative log likelihood for each string with a sliding window up to the 256 token context, then I pick the side with the lower value since lower means more likely under the model. The score is per BPE token, not per byte, and the sliding window keeps context fresh without padding. I print progress every ten thousand pairs and write one label per line to checkpoints/part1.txt. The method is the same logic I used for the char model, only with the BPE encoder.

---------------------------------------------------------------------------------------------------------------------

File 6: generate.py — Sanity-check sampling

This is my quick sanity check. I put the model in eval mode, encode a prompt with the BPE tokenizer, and sample with temperature plus top-k and top-p. I can set a seed for repeatability and stream prints to watch outputs as they arrive. Moderate temperature and sensible k and p usually give coherent text with some variety.

---------------------------------------------------------------------------------------------------------------------

File 7: checkpoints/ — Everything you need

This folder holds the artifacts I need to reproduce everything without retraining. meta.pt stores the tokenizer merges, the regex, the special ids, and the block size. lm.pt has the model weights and a minimal config so I can rebuild the same architecture and load the state dict. part1.txt is the submission from score.py. With these files I can reload the tokenizer and model, run the NLL scorer, and generate samples the same way I did during training.