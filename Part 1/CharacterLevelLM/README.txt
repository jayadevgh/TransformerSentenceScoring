This is a character level language model. We are using the negative log likelihoods of the sentences to compare which sentence the model likes more. 

Here is an example:
print(avg_nll_string(model, "This is an English sentence."))
1.1000322412561487 

print(avg_nll_string(model, "Th1s is n0t gr8 Engl1sh!!!"))
4.821158752441407

As you can see, a lower negative log likelihood means that the model believes that the sentence is constructed better. 

---------------------------------------------------------------------------------------------------------------------
STATS:
Parameters: 10.95M
Non corrupted file Size: 134,878,148 characters (1M sentences)
"The Inaccurate" Accuracy: 93.59% (Using training data)


Issues:
- Need to add a test split along with the train and val split to training
- Better tokenization
- Add eos, bos tokens 
- cannot access long range cues (256 block size limit)
- currently if input contains weird unicode (another language) -> penalized heavily
- can improve generation, (add temperature etc.)
---------------------------------------------------------------------------------------------------------------------

Here is are some examples of the model making an incorrect prediction:
Top mistakes (model preferred B):

#1 margin=-1.3485 (nll_a=1.7148, nll_b=0.3663)
A: http:\/\/www.mvlib.com\/details\/movie\/adventures-of-don-juan-267964.html 
B: just

#2 margin=-0.8108 (nll_a=2.7555, nll_b=1.9447)
A: (Just joking!)
B: (Just 

#3 margin=-0.5561 (nll_a=1.4505, nll_b=0.8944)
A: I shall be out to lunch.
B: I shall be out to the

#4 margin=-0.5483 (nll_a=1.4048, nll_b=0.8565)
A: Is this progress?
B: Is this unfortunately

#5 margin=-0.5322 (nll_a=2.2814, nll_b=1.7491)
A: I drew mine.
B: I drew it



NOW LETS GO OVER THE CONTENTS OF trial.py
---------------------------------------------------------------------------------------------------------------------

PART 0: Formatting
    For our transformer, we are only using the first sentences (that are not corrupted) for the training data. After converting train.txt to train_utf8.txt in order to access the file in the first place, we then separate the first and second sentences in each line and reroutes all the "first" sentences into a new file called english_corpus.txt. Finally we convert the file into a single string named corpus (which will later be converted to the data tensor)

---------------------------------------------------------------------------------------------------------------------

PART 1: Tokenization
    Tokenization is not complicated since we are going character level. For this I am using the dictionaries itos (int to string) and stoi (string to int) which allows me to easily map every every character to an integer and vice versa. I am also using methods to quickly decode and encode strings or lists of ints. 
    
    Also you may notice that I am using a special character in the form of <unk>. Often times, the corrupted sentences contain extra characters that are either not in the general vocab of the training data or maybe even bytes that don't correspond to any character (making the sentence corrupt). All of these situations map to the <unk> character and usually increase the negative log likelihood.
    
---------------------------------------------------------------------------------------------------------------------

PART 2: Streaming data + batches
    As I mentioned in the previous part, I am now converting the data into tensors and am using a hard 90-10 train-val split. Now let's go over each method. 
    
    get_batch(split)
    This method chooses train_data or val_data based on split (the parameter). Batch_size represents how many sequences are we looking at in parallel and block_size represents what is the maximum context length. From here we obtain a batch_size number starting idxs that are of block_size length. After shifting each sequence to the right by one, we obtain our inputs and targets (x and y) which are both of shape (batch_size, block_size). This means the target at every time step is the next character after the input position. Randomizing start indices exposes the model to diverse contexts without any padding.

    estimate_loss(model, iters=200)
    First off all we are disabling any gradient tracking when running this method which should make the method much faster. Second of all we are turning on eval mode (which turns off dropout so we use all neurons when calculating loss). Then we calculate losses for both training and evaluation. We don't want there to be too big or small of a gap between training and evaluation loss to avoid underfitting and overfitting. We do this by iterating through 200 random mini-batches by default and calculate the mean of the losses among these mini-batches as an estimate to the overall loss. This gives a stochastic (Monte-Carlo) estimate of the true loss over the whole split without scanning it end-to-end. Finally, we return these losses as a dictionary and set the model back into training mode.

---------------------------------------------------------------------------------------------------------------------

PART 3: Transformer Architecture

    Now to the important parts of the model - the transformer architecture. You may know that there are many ways to train Transformers but this is a very simple implementation. This architecture is heavily inspired by a nano-GPT model built by Karpathy in one of his videos. Throughout the summer I have been learning from his Zero to Hero series regarding deep learning. 
    
    This specific transformer is called a decoder only transformer, so there's no encoder component and there's no cross attention block. Our block only has a self-attention and the feed forward. Now let's go through each class one by one. 
    
    Embeddings
    Each character in our vocabulary is represented by a C-dimensional (n_embd) vector in a token embedding table. This table is shaped (vocab_size, n_embd). Similarly, each time step from (0 to T-1) is represented by a C-dimensional vector in a position embedding table so the model understands order. This table is shaped (block_size, n_embd). These tables are then used to convert each batch to their embedding values before adding both vectors. The token embeddings is shaped (B, T, C) while the positional embeddings are only shaped (T, C) because the values in each batch don't effect positional embeddings (only the position matters). Added together, the model sees which char and where it sits.
    
    Head
    The Head class is the smallest functional unit of self-attention. Each head learns how to make tokens "communicate" with each other using a content-based weighted averaging mechanism. Let’s walk through the code.
    A single attention head takes the combined embedding x (token + position) with shape (B, T, n_embd) and runs three linear layers to produce queries Q, keys K, and values V, each shaped (B, T, head_size). It then scores how much each position should listen to every earlier position by taking dot products between Q and K, scales by 1/sqrt(head_size) to keep numbers stable, masks the future so we can’t peek ahead, and applies a row-wise softmax so each row becomes probabilities that sum to 1. Those probabilities are the attention weights. Finally, it forms a weighted average of the value vectors: out = attention_weights @ V, giving (B, T, head_size).
    Why Q/K/V take on their roles comes from how gradients flow during training, not from anything hard-coded. Q only affects which sources a position prefers (so it learns “what this position needs”), K only affects how attractive a token looks to others (so it learns “what this token offers" - to the position we are looking at), and V is the actual information being passed along (so it learns “what to deliver” when attended to). Concretely, for each example in the batch the attention score matrix has shape (T, T), and entry (i, j) is the dot product between the query at position i and the key at position j — a content-based similarity that measures how well what position i is seeking aligns with what position j offers. Softmax is what turns raw similarity scores into a clean, comparable set of weights per position, ensuring the head mixes information as a proper weighted average rather than an arbitrary sum.
    
    MultiHeadAttention
    Multi-head attention runs several independent heads in parallel on the same input x (B, T, n_embd). Each head has its own Q/K/V projections and computes its own masked self-attention, returning (B, T, head_size). With num_heads heads, we choose head_size = n_embd // num_heads (so n_embd should be divisible by num_heads). In the forward pass, we call each head, concatenate their outputs along the channel dimension to get (B, T, head_size * num_heads) which equals (B, T, n_embd), then apply a learned output projection back to n_embd followed by dropout.
    Why do this? A single head can only express one attention pattern at a time. On the other hand, multiple heads let the model look at different things simultaneously. One head might focus on recent context, another on long-range dependencies, another on punctuation or matching brackets, etc. Concatenation preserves each head’s perspective, and the final linear proj mixes these perspectives into a single representation that plugs cleanly into the residual stream. During training, gradients encourage heads to specialize where it helps the loss most. Some redundancy can occur, but the combination typically captures richer structure than a single head of the same total width.
    
    FeedForward
    The feed-forward block is a position-wise MLP that operates on the last (channel) dimension of x independently at every time step. Given input x of shape (B, T, n_embd), it first expands features with a linear layer to 4*n_embd, applies a nonlinearity (GELU), projects back down to n_embd with another linear layer, and then applies dropout. Because these layers act on the channel dimension only, the output shape stays (B, T, n_embd) and no information is exchanged across time steps inside this block.
    Why include it? Self-attention mixes information across positions and the feed-forward network mixes and transforms features within each position, adding nonlinearity and capacity. The 4x expansion gives the MLP a larger intermediate space to compute richer feature interactions before compressing back, and GELU provides smooth, input-dependent gating that tends to work well in Transformers. Dropout regularizes the block’s output. In combination with attention (plus the surrounding residual connections and layer norms in the full Transformer block), this MLP helps refine token representations in a way that improves next-token prediction.
    
    Block
    A Transformer block stacks two sublayers with residual (skip) connections: masked multi-head self-attention and a position-wise feed-forward network. Given x shaped (B, T, n_embd), this “pre-norm” variant first normalizes per token with LayerNorm (ln1) before attention, adds the attention output back to x (x = x + sa(ln1(x))), then repeats the pattern for the MLP (x = x + ff(ln2(x))). LayerNorm operates across the channel dimension of each time step, stabilizing activations with learnable scale and bias and a small epsilon for numerical stability. The attention sublayer mixes information across time (who each token listens to), while the feed-forward sublayer mixes features within a token. Shapes stay (B, T, n_embd) throughout, attention returns (B, T, n_embd) after concatenating heads and projecting, and the MLP maps n_embd -> 4*n_embd -> n_embd. Dropout in attention (on weights or output) and in the MLP regularizes both sublayers.
    Why this structure? Residual connections create an identity highway so information and gradients can flow through many stacked blocks; if a sublayer learns nothing useful, the model can fall back to the skip path. Pre-norm (LayerNorm before each sublayer) tends to train more stably in deep stacks than post-norm because gradients see a normalized path. The masked self-attention enforces causality in decoder-only models, letting each position aggregate only from itself and the past. The MLP then refines those aggregated features with nonlinearity and channel-wise mixing. Together, “attention for token-to-token communication” plus “MLP for local transformation,” wrapped in LayerNorm and residuals, gives a robust, depth-scalable unit: stable statistics, efficient gradient flow, richer representations, and consistent (B, T, n_embd) interfaces that compose cleanly in a stack.
    
    GPTLanguageModel
    This module wires the whole model end to end. Tokens are turned into vectors with a token embedding table (vocab_size, n_embd) and positions are turned into vectors with a position embedding table (block_size, n_embd). These are added to form the combined embedding x of shape (B, T, n_embd); the position term is (T, n_embd) and broadcasts across the batch. The stacked Transformer blocks (nn.Sequential([...])) then refine x with masked self-attention and feed-forward layers, keeping the shape (B, T, n_embd). A final LayerNorm (ln_f) stabilizes features before the output projection. The language-model head (lm_head) is a linear layer to vocab size, producing logits of shape (B, T, vocab_size).
    Weights for linear and embedding layers are initialized from a normal distribution with std 0.02, and linear biases (if present) are zeroed. This is a standard Transformer-friendly init that keeps activations and gradients in a reasonable range at the start of training.
    Given indices idx (B, T), the model computes tok_emb, pos_emb, sums them, runs through the block stack, applies the final LayerNorm, and projects to logits (B, T, V). If targets are provided, logits and targets are flattened to (BT, V) and (BT,) and the cross-entropy loss is computed; otherwise loss is None. The method returns the per-token logits shaped back to (B, T, V) along with the optional loss.
    
    generate
    The generate method uses autoregressive decoding that samples new tokens one at a time. Gradients are disabled with @torch.no_grad() to save memory and compute. At each step, the method crops the current sequence to the last block_size tokens (idx_cond = idx[:, -block_size:]) so inference respects the model’s fixed context window. It runs a forward pass to get logits over the vocabulary for every position, keeps only the last position’s logits (the next-token prediction), converts them to probabilities with softmax, and samples the next token via torch.multinomial (stochastic decoding). The sampled id is appended to the running sequence, and the loop repeats until max_new_tokens have been generated. The result is the original prompt plus the newly generated tokens, returned as a single tensor of ids.
    Currently his path assumes single-sequence generation with no padding needed. I am using multinomial to introduce diversity. Through greedy decoding I could also use argmax instead, and for finer control I plan on adding temperature or top-k filtering before sampling in the future. I made sure the model is in eval mode (model.eval()) so dropout is disabled during generation. This implementation recomputes the full forward pass each step; for long sequences you can speed it up with key/value caching.

---------------------------------------------------------------------------------------------------------------------

PART 4: Training
    
    Here we are finally training the model. After instantiating the model, I optimize it with AdamW using mixed precision. Afterwards, I monitor train/val loss periodically, and save a reusable checkpoint (lm.pt) with both weights and config.
    
    Model initialization + optimization
    model = GPTLanguageModel().to(device) constructs the Transformer defined earlier and moves it to the selected device (CPU or CUDA). The next line prints a quick sanity check of model size by summing .numel() over all parameters and reporting millions of parameters; this is useful to catch configuration mistakes (e.g., wrong n_embd or n_layer) and to gauge expected memory/compute needs before training.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) sets up AdamW, which is Adam with decoupled weight decay. This is the standard choice for Transformers because it keeps the adaptive updates while applying weight decay as a separate, well-behaved regularizer. scaler = torch.cuda.amp.GradScaler(enabled=use_amp) prepares automatic mixed precision: it scales losses dynamically to avoid FP16 underflow, enabling faster training and lower memory use on GPU while keeping numerics stable.
    
    Training Loop
    The loop switches the model into training mode (model.train()), starts a timer, and iterates for max_steps. Each step samples a fresh mini-batch with get_batch('train'), which returns contiguous context windows as inputs xb and their next-character targets yb. The forward pass runs under torch.cuda.amp.autocast(enabled=use_amp) to use mixed precision on CUDA for speed and lower memory, producing a loss. Gradients are cleared with optimizer.zero_grad(set_to_none=True) (more memory-friendly than writing zeros), then scaler.scale(loss).backward() backpropagates a scaled loss to avoid FP16 underflow. scaler.step(optimizer) safely unscales and applies the optimizer update when numerically valid, and scaler.update() adjusts the scaling factor for the next step. This loop performs one optimization step per mini-batch; dropout is active, parameters update each iteration, and the timer (t0) can be used to report throughput or ETA.
    
    Eval logging + Checkpointing
    Every eval_interval steps (and at step 1), the loop calls estimate_loss(model) to compute mean losses on random mini-batches from the train and validation splits without gradient tracking. This switches the model into eval behavior inside that helper (dropout off, BatchNorm not used here), aggregates several batches to smooth noise, then returns to train mode so the outer loop can continue. The code also measures wall-clock time since t0 and prints a compact line with the current step, averaged train loss, averaged val loss, and elapsed minutes. Watching train vs val lets you spot underfitting or overfitting and confirm the run is progressing at a reasonable speed.
    At the end, torch.save({...}, ckpt_file) writes a checkpoint containing the model’s state_dict() (all learned weights) and a small config dictionary with key hyperparameters (vocab size, embedding size, heads, layers, dropout, block size). Keeping config alongside weights makes reloads reproducible and guards against shape mismatches when reconstructing the model later. This saves just the model; if you also want to resume training seamlessly, you can additionally save the optimizer state and, if using AMP, the GradScaler state in the same dict.

---------------------------------------------------------------------------------------------------------------------

PART 5:  Avg-NLL scorer (sliding window)

    Now that we created an entire LM (that can even generate text!!), let's shift focus to the actual problem at hand. This method essentially scores different sentences by looking at block_size sized windows and calculating the negative log-likelihood (NLL). 
    
    This method computes the average NLL of a plain-text string under the model, using a sliding window up to the model’s context length. It first encodes the string to token ids on the current device, then walks through the sequence in chunks of at most block tokens. For each chunk, it builds input ids (all but the last token) and target ids (all but the first), runs a forward pass with gradients disabled, converts logits to log-probabilities with log_softmax, and uses gather to pick the log-prob for each true next token. It accumulates the sum of log-probs and the token count across windows, advancing the window by block - 1 tokens so the last token of a window becomes the first context token of the next window. It returns the negative of the total log-probability divided by the total number of predicted tokens, i.e., mean NLL over the string. A degenerate length check returns a large value for strings with fewer than two tokens.
    
---------------------------------------------------------------------------------------------------------------------

PART 6:  Writing part1.txt

    Starting off the code, I switch to eval mode for stable logits, then stream test pairs from TEST_PATH. For each line, split on the tab into strings a and b; if no tab, default to label A. Then, I compute average NLL for each string with avg_nll_string(model, ·), pick the lower-NLL string as the label (A if a is more likely, else B), and write one label per line to OUT_PATH. Every 10k pairs, I print a progress message (optionally flush to disk). Finally, I finish by reporting the output file path.
    
