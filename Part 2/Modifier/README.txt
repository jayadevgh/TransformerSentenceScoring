This stage builds part2.txt: for every original string (the first tab-separated column of train.txt), we synthesize a challenging corruption in the second column. We score candidates with the same language-model NLL idea as Part 1 (lower NLL ⇒ “more English”), but here we seek the smallest positive NLL increase—i.e., “hard negatives” that are wrong, yet close enough to tempt a model.

Before you read further: please read the CharacterLevelLM README first and the Tokenizer/BPE second before coming here. Details are already covered in the TokenizedLM README. I reference them below instead of repeating.

---------------------------------------------------------------------------------------------------------------------

STATS
- Input size: 1,000,000 original strings (A-side of train.txt)
- Output: part2.txt (A\tB for each line), flushed/append-safe (resume if preempted)
- Scoring LMs: both char LM and BPE LM (alternate by line: even→char, odd→BPE)
- Context window: 256 tokens/chars (sliding-window average NLL, same as Part 1)
- Corruption families used: reorder, drop/insert function words, light typos, number/date tweaks, small lexical swaps, minimal clause edits (kept compact for speed/variety)
- From a sample of the first 30k sentences: 
    - part2.txt: Δ<=0: 2  (0.01%)  Δ>0: 29,998
    - train_utf8.txt: Δ<=0: 2,211  (7.37%)  Δ>0: 27,789
- Look at setup.ipynb for graphs on Δ distribution

Guards:
- must differ from A and from gold B (in training file);
- delta floor: I try to require a tiny positive NLL margin (just > 0) so B is barely worse;
- fallback if no candidate passes (deterministic, still non-trivial).

CAVEATS & LESSONS
- No byte-level “broken UTF-8” corruptions.
- In the original corpus, some B-sides include undecodable bytes that your char LM treated as <unk>. Here we must produce a valid UTF-8 plaintext file, so we cannot replicate those exact byte-level failures. We instead simulate with near-equivalents (odd symbols, replacement of rare codepoints, etc.) that remain decodable.
- Subtle corruptions are intentional.
- Because we choose the smallest positive NLL increase, many B’s look “almost fine”. That’s by design: they’re the hardest for a Part 1-style scorer. This subtlety also occurs in the original training pairs, so Part 2 mirrors that difficulty.

Runtime & robustness.
Scoring every candidate with an LM is the heavy part. To make long runs safe on limited sessions, we:
- open part2.txt in append mode and flush+fsync after each line
- resume by skipping already-written lines on restart
- keep corruption families lightweight and avoid exploding the number of trials per line.

---------------------------------------------------------------------------------------------------------------------

(From the first 30k examples)
part2.txt:

[examples] Top 2 most negative Δ (model prefers B):

#1  line=9001  Δ=-0.3968
A: good uh let's see so uh were we right in the Middle East
B: good uh let's see so uh were we right in the Middle East .

#2  line=27489  Δ=-0.0162
A: To what extent are we maintaining more than just superficial contact with government bodies?
B: To what extent are we maintaining more than just superficial contact with government bodies? .

train_utf8.txt:

[examples] Top 5 most negative Δ (model prefers B):

#1  line=19324  Δ=-1.2730
A: http:\/\/movies.nytimes.com\/movie\/361895\/In-3-Tagen-Bist-Du-Tot\/overview
B: Charlie

#2  line=8651  Δ=-1.0078
A: Laura ignores him and marries Glyde .
B: Laura ignores him and marries Paul .

#3  line=9503  Δ=-0.9438
A: As they get off the mountain , the last pterodactyl attacks .
B: As they get off the mountain , the last friend attacks .

#4  line=14347  Δ=-0.9433
A: who's the who's the author
B: who's the who's author 

#5  line=7863  Δ=-0.8997
A: Shh, she said.
B: because she said.

---------------------------------------------------------------------------------------------------------------------

Commentary on my corruptions:

I think my data tells a clear story about what I am are trying to do and why it looks different from the training file. Since I am trying to make B feel like the previous sentence with a small bruise rather than a rewrite, the changes are light and local and the model almost always thinks A is better, and the Δ values live in a narrow band between zero and two with almost no negatives, which is exactly what I would expect from subtle corruptions that keep meaning and shape while nudging fluency down a notch.

When I look at the two negative flips in part2 I see the kind of thing that tricks a language model in the other direction. B adds a final period or a stray space before a period, and the model likes the sense of closure and the very common punctuation token, so NLL goes down even though I meant to corrupt, and that is why my negatives are rare and tiny and almost always punctuation related. This is the cost of working so close to the surface, because a tiny edit can sometimes tidy the sentence rather than harm it, and a byte-level BPE really rewards common tokens like periods.

Now compare that to train where Δ goes all the way out to four and where a non-trivial slice of Bs actually beat A by a mile. Those examples look like entity swaps, rare-to-common substitutions, dropped function words, and a few straight misalignments, like a URL on A paired with “Charlie” on B, and that is not a corruption of the previous sentence so much as a different sentence altogether. The model learns that replacing a rare token with a frequent token lowers NLL and that deleting a tricky span can also lower NLL, so sometimes B “makes more sense” than A in the narrow, statistical way that a next-token model measures sense, and that gives me big negative Δ outliers that I never see in my test.

For the task “make a corrupted version of the previous sentence” I think the distribution is closer to the spirit of the goal. I keep meaning intact and I introduce small errors that humans would spot but that still read like the same sentence, so the decision boundary sits near zero and the model wins by a modest margin. The training file is teaching a different habit. It is pushing the model to expect heavy corruption and label noise, which widens the Δ spread and sometimes makes B cleaner than A. That mismatch matters because it can push my scorer to rely on crude frequency cues rather than the small grammar and agreement cues I want it to notice in part2.

LETS GO OVER THE PROJECT FILES:
---------------------------------------------------------------------------------------------------------------------

Table of Contents:

project/
├─ constants.py        # paths, device, thresholds, print cadence
├─ loaders.py          # restore char/BPE LMs from artifacts (see Part 1 READMEs)
├─ nll.py              # average NLL (sliding window), char+BPE variants
├─ corruptions.py      # small, fast corruption families (no external data)
├─ part2.py            # build part2.txt: alternate LM, pick “closest worse” B, append+fsync
├─ quick_sanity.py     # quick end-to-end smoke test (encoders/logit shapes/NLLs)
├─ bench_part2.py      # tiny throughput check on a sample
└─ CharArtifacts/ and TokenArtifacts/ # my saved meta/weights from parts 1 (char) and tokenizer LM
                        
---------------------------------------------------------------------------------------------------------------------

constants.py

I put all the knobs in one place so part2.py stays small and so I do not drift between runs. This file defines file paths for inputs and outputs, artifact roots for the character model and the tokenized model, a simple device autodetect that picks CUDA when it is present and falls back to CPU when it is not, and a print cadence so progress lines show up at a steady pace. I also keep separate delta floors for the char model and the BPE model since they behave a little differently near the decision boundary, and I store the maximum number of attempts I will make per line when I am trying to find a subtle corruption that sits just above zero. There is a hard cap on string length so I do not waste time on outliers, and there is a toggle for fsync so I can choose between safer writes and faster throughput. I can change behavior for a whole run by touching one file and it keeps the generation script focused on the actual logic through centrralizing these values.

---------------------------------------------------------------------------------------------------------------------

loaders.py

These are thin helpers that rebuild the model exactly as it was trained, then load the weights without complaining about old names. I read the saved config and recreate the shapes for embeddings, attention, and the head, and I accept both the older variable names like token_embedding_table and lm_head and the newer short names like tok and head, which lets me load checkpoints from different stages of this project without any hacks. For BPE runs I also load meta.pt which holds merges, special tokens, and the regex pattern, and I expose the same encode routine I used earlier so scoring and generation see the text in the same way. Architecturally nothing surprising happens here and the details match the READMEs from the character and tokenized LMs, I just make the path from disk to ready model as short and reliable as possible.

---------------------------------------------------------------------------------------------------------------------

nll.py

This module computes average negative log likelihood with a sliding window up to the fixed context of 256, which is the same method I used before because it is stable on long strings and easy on the GPU. For the character model I report NLL per character and for the BPE model I report NLL per BPE token, which keeps numbers comparable inside each model family and avoids length bias. The sliding window crops each chunk to fit the context and advances by an overlap so the last token of one window becomes context for the next, and I then average across all predicted positions. This keeps the estimate smooth, it avoids padding, and it gives me a clean scalar I can compare between A and candidate B so I can decide whether a corruption is subtle and still worse.

---------------------------------------------------------------------------------------------------------------------

corruptions.py

Here I define a small set of data-free edits that usually nudge NLL up but do not wreck the sentence. I keep the changes tiny so the corrupted string stays close to the original and the classifier has to pay attention to grammar and micro fluency rather than just frequency. I do not use external lexicons or language tools since I want to respect the no external data rule and I want this to be fast enough to run inside an acceptance loop. I also respect a global MAX_EDITS which keeps me from overdoing it in one pass. A candidate can still accidentally improve the string, especially with punctuation, so the main script filters on a small positive delta and tries again until it finds a near miss that is worse, which keeps me in the subtle regime.

Let's go over each type of corruption that I opted for:

micro typos

I use tiny character-level edits like swap, delete, duplicate, and replace, and I keep the count under MAX_EDITS so the sentence still reads as the same thought. A swap turns “form” into “from” or the other way around which can hurt or help, a delete drops a single character which usually harms, a duplicate creates a stutter like “the the,” and a replace picks a letter with the same case so I do not change the look of the token too much. These edits are strong enough to push NLL up a little under both the character model and the BPE model since they disrupt common substrings, yet they rarely break the whole sentence. If a swap accidentally lands on a more common pattern, the acceptance filter catches it and I resample.

punctuation and spacing tweaks

I apply small punctuation moves like removing a comma, adding a comma in a neutral interior position, doubling one whitespace span, or stripping a space before a punctuation mark. These are quick edits that change rhythm and local tokenization without touching content. On byte-level BPE the period and comma are very common tokens so simply adding a final period can lower NLL, which is why I lean on interior commas and spacing noise and why I keep the acceptance check in place. The goal is to make the sentence feel slightly off rather than cleaner, so I avoid edits that tidy the ending and prefer ones that disturb spacing or pause placement inside the line.

function word tweaks

I change small glue words like articles, prepositions, and simple auxiliaries, and I sometimes move a function word across its neighbor. Deleting “the,” swapping “in” with “on,” or replacing “is” with “are” tends to raise NLL because agreement and selectional fit get worse while the surface still looks familiar. I keep a small list of common function words and a small map of reasonable swaps so the edit feels natural in style while being wrong in detail. This stays close to the original meaning and length, and it teaches the model to notice agreement and choice of preposition rather than only big content words.

number tweaks

I nudge digits first since they are easy to detect and a one digit change is enough to make the sentence inconsistent with itself. If there are no digits I look for simple number words and step them up along a tiny ladder like one to two or two to three, and if I find nothing I may inject a small stray digit in a neutral place. Numbers tend to tokenize cleanly and are frequent enough that the model has an expectation for common strings, so small mutations usually raise NLL while keeping everything else untouched. I keep the tweak local so I do not create a new word boundary or a different sentence shape.

local reorder

I pick one adjacent pair of words and swap them which is just enough to disturb syntax without changing the bag of words. This can create things like “on based data” instead of “based on data,” which raises NLL under both tokenizations while preserving length and vocabulary. I only swap neighbors and I do it once per candidate so the sentence still looks like itself, and if the swap happens to be benign, the delta filter rejects it and I try again. This type pairs well with the function word tweak since they both test short-range dependencies.

number of edits and acceptance loop

Across all families I keep edits tiny by obeying MAX_EDITS and I run an acceptance loop that measures delta with the right model and only accepts candidates that land in a small positive band. This lets me keep the subtle style I want while avoiding cases that tip negative. It also explains why my distribution is narrow and almost never crosses zero, since I actively steer candidates toward the closest worse outcome and away from accidental improvements.

---------------------------------------------------------------------------------------------------------------------

part2.py

This script builds part2.txt by streaming the A side and writing a matched B that is barely worse. For each line I pick which language model to use based on the index so the character model and the BPE model both get coverage, I compute the base NLL on A, and I draw a handful of candidate edits from the corruption families. Then I filter out trivial cases like ones that collapse to the same words once I strip punctuation and case, I reject anything that equals A or equals the gold B from the training set if that exists, and I require a small positive NLL margin so I do not ship accidental improvements. Among the survivors I choose the smallest positive delta which is the closest worse option since that keeps the difficulty near the boundary, and if nothing survives after a few attempts I fall back to a deterministic change so I still make progress. I append A, a tab, and B, then I flush and optionally fsync so I can resume safely if the job restarts. When I rerun the script it skips lines that already exist which lets me resume from the last good write without scanning everything again.



