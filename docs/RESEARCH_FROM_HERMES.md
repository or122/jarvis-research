# Flow — Research Brief for the Training Effort

**From:** Hermes (research pass over the actual Flow codebase, `~/flow`)
**For:** the Claude Code session training Flow in VS Code
**Date:** 2026-09-12

## How to read this
Every item is scoped to Flow's *real* constraints, read straight from the code:
**~11.07M params, decoder-only, word-level vocab 12,288, block 128, n_embd 256,
n_head 8, n_layer 6; trained on ONE Intel MacBook Pro 16,2 (4 physical / 8
logical cores, no GPU), torch pinned 2.2.2, CPU-bound ~1,500 tok/s and thermally
throttling.** Generic "how to train an LLM" advice is deliberately excluded —
only levers that pay off at *this* size on *this* machine are here.

Each item lists: the change, the file/lines, the evidence, confidence, and how
to measure it with the gates you already have (`eval.py`, `eval_code.py`,
`score.py`). Suggested order is Tier 1 → 3.

The north-star fact that reorders everything (see T1-3): **Flow is
undertrained, not undersized.** So the highest-value work is *throughput and
data*, not more parameters.

---

## Tier 1 — Highest leverage, low risk

### T1-1 · Tie the input and output embeddings
**Change:** after building the layers in `FlowLM.__init__`, add
`self.lm_head.weight = self.token_embedding.weight`, and drop the `lm_head`
bias (`nn.Linear(N_EMBD, vocab_size, bias=False)`).
**File:** `model/model.py` — lines 119 & 125.
**Why it's #1 for Flow specifically:** the two embedding matrices are
`12288×256 = 3,145,728` params *each*. The output matrix (`lm_head`) is a second
copy of what the input embedding already learns. Right now it costs **3,158,016
params — 28.5% of the entire 11.07M model — for no representational gain.** Tying
them:
  - frees ~3.16M params to reinvest (more data passes, or +1 layer if you must);
  - is shown to *improve* perplexity, not just save memory.
**Evidence:** Press & Wolf, "Using the Output Embedding to Improve Language
Models," arXiv:1608.05859 (ACL 2017). Standard in GPT-2/nanoGPT.
**Confidence:** High.
**Gate check:** GATE 1 (untrained loss ≈ ln(12288) = 9.42) still holds — tying
doesn't change the uniform-guess baseline. Watch val loss (Gate 2) drop or hold
at fewer params; expect equal-or-better at the same iters.
**Caveat:** must retrain from scratch (checkpoint shape changes). The vocab
mismatch guard in `sample.py:load()` protects you; also bump/verify any code
that assumes a separate lm_head tensor.

### T1-2 · Replace hand-rolled attention with fused SDPA
**Change:** collapse the per-head Python loop into one batched QKV projection and
call `torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True,
dropout_p=DROPOUT if self.training else 0.0)`. Verified present in your torch
2.2.2 (`hasattr(F,'scaled_dot_product_attention') == True`).
**File:** `model/model.py` — `Head` / `MultiHeadAttention`, lines 21–72.
**Why:** two speedups compound — (a) SDPA uses a fused CPU kernel and never
materializes the `(B, T, T)` score matrix per head; (b) it removes the
`torch.cat([h(x) for h in self.heads])` Python loop over 8 separate modules,
replacing it with one `(B, n_head, T, head_size)` reshape + one call. At
B=32, T=128, this is your hottest loop. Faster steps = more tokens seen in the
same overnight window, which the design doc itself names the single biggest
lever.
**Evidence:** PyTorch docs for `scaled_dot_product_attention` (fused CPU/flash
backends); this is exactly nanoGPT's `F.scaled_dot_product_attention` path.
**Confidence:** High that it's correct and available; Medium on the exact CPU
speedup (benchmark it — see below). Keep the old code behind a flag and confirm
GATE 1 loss is unchanged (same math, so it must be).
**Measure:** the training log already prints `tok/s`. Run 200 iters before/after
with `FLOW_EVAL_EVERY=50` and compare tok/s and wall-clock.

### T1-3 · Reframe the roadmap: Flow is UNDERTRAINED, so don't add params
**Not a code change — a planning correction that should gate T-anything that
adds parameters.**
**The numbers:** Chinchilla's compute-optimal ratio is ~20 tokens per parameter.
11M params → ~**220M** tokens to be "optimally" trained. Flow's CPU cap lets it
*see* ~50M tokens overnight (of 74.5M prepared). Flow is at ~4–5× *below*
compute-optimal — it is firmly **data/compute-limited, not capacity-limited.**
**Implications:**
  - Adding layers/width makes each CPU step slower and pushes you *further* from
    optimal — the opposite of what you want. The design doc's own risk table
    ("training too slow → drop n_layer to 4, block to 64") is the correct
    direction; scaling *up* is not.
  - The freed budget from T1-1 is better spent on **more passes / cleaner data**
    than on more parameters.
  - Best marginal hour: anything in Tier 1–2 that raises tok/s or data quality.
**Evidence:** Hoffmann et al., "Training Compute-Optimal Large Language Models"
(Chinchilla), NeurIPS 2022. Applies as a *direction*, not a promise, at this
scale.
**Confidence:** High on direction (undertrained), Medium on the exact 20× target
for an 11M model on narrow data.

---

## Tier 2 — Training quality

### T2-1 · Pin CPU threads to physical cores and test 4 vs 8
**Change:** at the top of `train.py`, `torch.set_num_threads(4)` (physical core
count on this MacBookPro16,2) and export `OMP_NUM_THREADS=4`. A/B test 4 vs 8.
**Why:** torch currently defaults to 4 here (good), but leaving it implicit is
fragile, and on a 4-core/8-thread chip GEMM throughput usually peaks at
*physical* cores — logical (hyperthread) cores add contention and **heat**,
which is your documented enemy (a 15-min job ran 55 min from throttling).
**Evidence:** PyTorch "Grokking PyTorch Intel CPU performance from first
principles" (pin to physical cores, avoid logical cores, NUMA/affinity).
**Confidence:** High it helps determinism/heat; Medium on raw speed — **measure
tok/s at 4 vs 8** and keep the winner.

### T2-2 · Mask the prompt in the chat fine-tune (completion-only loss)
**Change:** for the `chat_` dataset, compute cross-entropy only on the model's
reply tokens; set the `You: … \nMe:` prompt tokens to `ignore_index=-100` in the
targets. Requires the token prep to emit a prompt/reply boundary (or a mask
built at the `\nMe:` marker).
**Files:** `model/train.py` loss (line 168 → `model(x, y)`), `model/model.py`
`F.cross_entropy` (line 152 — pass `ignore_index=-100`), and
`model/prepare_chat_tokens.py` (emit the boundary/mask).
**Why:** Flow spends its tiny capacity partly on *reproducing the user's turn*.
Masking the prompt focuses every gradient on *learning to reply* — the exact
thing `score.py`'s "Chatting" measures.
**Evidence:** "Instruction Fine-Tuning: Does Prompt Loss Matter?" arXiv:2401.13586.
**Nuance:** the paper finds a *small nonzero* prompt-loss weight can beat full
masking in some setups — so treat full-mask as the strong default and, if you
have time, try a light prompt weight (~0.1) as a variant.
**Confidence:** Medium at this scale — cheap to try, measure with `score.py`
"Chatting" and `eval_code.py`.

### T2-3 · Scaled init on residual-projection layers (GPT-2 trick)
**Change:** in `_init_weights`, init the layers that *write into the residual
stream* — attention output `proj` and the FFN's second `Linear` — with
`std = 0.02 / sqrt(2 * N_LAYER)` instead of a flat 0.02.
**File:** `model/model.py` — `_init_weights`, lines 128–136 (tag those two
linears, e.g. a `._is_resid_proj = True` flag set in `MultiHeadAttention` /
`FeedForward`).
**Why:** without it, residual variance grows with depth; GPT-2/nanoGPT scale
these down so a 6-layer stack trains better-conditioned. Free, standard.
**Evidence:** nanoGPT `model.py` (`c_proj` special init, `0.02/sqrt(2*n_layer)`),
after the GPT-2 implementation.
**Confidence:** High that it's correct/standard; Low-Medium on measurable gain
at 6 layers — but it's ~3 lines and risk-free.

---

## Tier 3 — Inference polish (no retraining)

### T3-1 · Add repetition control to generation
**Change:** in `model.generate` / `sample.stream`, add **no-repeat-ngram**
(block any n-gram, n=3, already seen) and/or a light **repetition penalty** on
recently-emitted tokens, before the softmax. Optionally try **min-p** sampling
alongside or instead of `top_k`.
**Files:** `model/model.py` `generate` (lines 155–179), `model/sample.py`.
**Why:** looping/degeneration is *the* failure mode of small LMs — you already
felt this (Gate 3 in `eval.py` added a "variety ≥ 0.5" check, and `sample.py`
has stop-sequence and `min_new_tokens` machinery for the same reason). This lifts
the "English variety" and "Creativity" scores with zero retraining.
**Evidence:** no-repeat-ngram / repetition & frequency penalties are the standard
degeneration controls; min-p: Nguyen et al. 2024 (ICLR). Caveat: min-p's
evaluation is contested (arXiv:2506.13681), so make **no-repeat-ngram + light
repetition penalty the primary**, min-p an experiment.
**Confidence:** High that repetition control helps; Medium on min-p vs top_k.
**Measure:** `eval.py` Gate 3 variety %, and `score.py` English/Creativity.

---

## Suggested order & measurement loop
1. **T1-1 (weight tying)** + **T1-2 (SDPA)** together, then retrain — biggest
   quality-per-hour and speed-per-hour wins; both need a fresh run anyway.
2. Benchmark tok/s (T1-2) and set threads (T2-1) *before* the long overnight run.
3. Fold in **T2-3** (init) in the same retrain — free.
4. **T2-2** (prompt masking) on the chat fine-tune stage.
5. **T3-1** last — pure inference, iterate fast without retraining.

Keep the existing gates as the scoreboard the whole way:
`eval.py` (3 gates) · `eval_code.py` (4 gates) · `score.py` (/100).
Change one thing, rerun, keep what wins. Nothing here promises "closer to
Claude" — as `score.py` already says honestly, that scale is out of reach on a
laptop. These changes make Flow the *best 11M-param CPU model it can be*, which
is the actual goal.

## Sources (primary)
- Press & Wolf 2017, *Using the Output Embedding to Improve Language Models* — arXiv:1608.05859
- Hoffmann et al. 2022, *Training Compute-Optimal LLMs* (Chinchilla) — NeurIPS 2022
- Eldan & Li 2023, *TinyStories* — arXiv:2305.07759 (Flow's exact size class)
- PyTorch: `scaled_dot_product_attention` docs; *Grokking PyTorch Intel CPU performance*
- Karpathy, nanoGPT `model.py` (weight tying, SDPA path, scaled residual init)
- Shi et al. 2024, *Instruction Fine-Tuning: Does Prompt Loss Matter?* — arXiv:2401.13586
- Nguyen et al. 2024, *Min-p Sampling* (ICLR); critique arXiv:2506.13681
