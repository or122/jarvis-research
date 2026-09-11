"""Train FlowLM.

Run:  .venv/bin/python model/train.py [max_iters]
Overnight, so the Mac doesn't sleep:
      caffeinate -i .venv/bin/python model/train.py 60000

Saves ckpt.pt only when validation loss improves, so an interrupted run still
leaves the best model on disk.
"""
import math
import os
import shutil
import sys
import time

import numpy as np
import torch

from model import BLOCK_SIZE, FlowLM
from tokenizer import DATA, WordTokenizer

BATCH_SIZE = 32
LEARNING_RATE = 6e-4      # measured 1484 tok/s, so CPU time is the constraint;
                          # warmup + gradient clipping make this rate safe
# Both settable from the environment so a speed test can get numbers in
# minutes instead of an hour. A 500-iteration gap between log lines hid a 4x
# throughput collapse for over an hour.
EVAL_INTERVAL = int(os.environ.get("FLOW_EVAL_EVERY", "500"))
EVAL_ITERS = int(os.environ.get("FLOW_EVAL_ITERS", "40"))
WARMUP_ITERS = 200        # ramp the rate up so early steps can't blow up
MIN_LR = LEARNING_RATE / 10
# Cosine decay: high rate early to explore, small rate late to settle. Worth a
# few percent of final loss on a long run, for a handful of lines.
DECAY = os.environ.get("FLOW_DECAY", "1") == "1"
MAX_ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
# Second argument picks the dataset: "" is the base story data, "chat_" is
# the conversation data used for fine-tuning.
PREFIX = sys.argv[2] if len(sys.argv) > 2 else ""
# The cosine schedule's horizon, separate from where THIS run stops. Chunked
# training calls train.py many times with a nearby MAX_ITERS; without this the
# learning rate decayed to its minimum at the end of every chunk and sawtoothed
# back up, so the model never got one long, smooth decay.
TOTAL_ITERS = int(os.environ.get("FLOW_TOTAL_ITERS", "0")) or MAX_ITERS

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt.pt")
# A second copy that is only ever written when the loss genuinely improves on
# everything seen before. ckpt.pt is the working file a run resumes from;
# ckpt_best.pt is the safety net, so a bad chunk can never destroy a good model.
BEST = os.path.join(HERE, "ckpt_best.pt")

torch.manual_seed(1337)

# memmap: the token files stay on disk and pages are read as needed, so a
# 70 MB dataset never has to sit in RAM alongside the model.
train_data = np.memmap(os.path.join(DATA, f"{PREFIX}train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(DATA, f"{PREFIX}val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.from_numpy(np.stack([d[i:i + BLOCK_SIZE].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([d[i + 1:i + BLOCK_SIZE + 1].astype(np.int64) for i in ix]))
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def lr_at(it, total=None):
    """Linear warmup, then cosine decay down to MIN_LR.

    Warmup stops the first few huge steps from wrecking the model; the decay
    lets it take smaller, more careful steps as it converges.
    """
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / WARMUP_ITERS
    if not (DECAY and total):
        return LEARNING_RATE
    if it >= total:
        return MIN_LR
    progress = (it - WARMUP_ITERS) / max(total - WARMUP_ITERS, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


if __name__ == "__main__":
    tok = WordTokenizer.load()
    model = FlowLM(tok.vocab_size)
    n_params = sum(p.numel() for p in model.parameters())

    resumed = ""
    best_val = float("inf")
    start_iter = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location="cpu")
        model.load_state_dict(ck["model"])
        # A different DATASET makes the old best_val incomparable, so it is
        # reset then - but only then. Resetting whenever a prefix was set
        # fired on every chunk of the same dataset, so the first eval of each
        # chunk always overwrote the checkpoint even when it was worse. The
        # model went backwards: chunk 8 reached 3.0243 and the next save put
        # 3.0353 on disk.
        same_dataset = ck.get("dataset", "") == PREFIX
        best_val = ck["val_loss"] if same_dataset else float("inf")
        start_iter = ck["iter"]
        resumed = f"  (resumed from iter {start_iter}, val {best_val:.4f})"

    print(f"parameters: {n_params:,}   vocab: {tok.vocab_size}   "
          f"tokens: {len(train_data):,}   dataset: {PREFIX or 'base'}{resumed}")
    print(f"iters: {MAX_ITERS}   batch: {BATCH_SIZE}x{BLOCK_SIZE} "
          f"= {BATCH_SIZE * BLOCK_SIZE:,} tokens/step\n", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location="cpu")
        if "optim" in ck:
            optimizer.load_state_dict(ck["optim"])
    t0 = time.time()

    for it in range(start_iter, MAX_ITERS + 1):
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
            losses = estimate_loss(model)
            mins = (time.time() - t0) / 60
            done = max(it - start_iter, 1)
            tps = done * BATCH_SIZE * BLOCK_SIZE / max(time.time() - t0, 1)
            star = ""
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(),
                            "optim": optimizer.state_dict(),
                            "vocab_size": tok.vocab_size,
                            "val_loss": best_val,
                            "dataset": PREFIX,
                            "iter": it}, CKPT)
                star = "  <- saved"
                # Mirror to the safety net only on a true all-time best.
                prior = float("inf")
                if os.path.exists(BEST):
                    try:
                        prior = torch.load(BEST, map_location="cpu")["val_loss"]
                    except (RuntimeError, KeyError, EOFError):
                        pass
                if best_val < prior:
                    shutil.copyfile(CKPT, BEST)
            print(f"iter {it:6d}   train {losses['train']:.4f}   "
                  f"val {losses['val']:.4f}   {mins:6.1f} min   "
                  f"{tps:5.0f} tok/s  lr {lr_at(it, TOTAL_ITERS):.1e}{star}",
                  flush=True)

        for g in optimizer.param_groups:
            g["lr"] = lr_at(it, TOTAL_ITERS)

        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Clip gradients: one freak batch can otherwise produce a huge step
        # and send the loss to NaN hours into an unattended run.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    print(f"\nbest val loss: {best_val:.4f}   saved to {CKPT}")
    print(f"GATE (val < 4.5): {'PASS' if best_val < 4.5 else 'FAIL'}")
