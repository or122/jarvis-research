"""Train FlowLM.

Run:  .venv/bin/python model/train.py [max_iters]
Overnight, so the Mac doesn't sleep:
      caffeinate -i .venv/bin/python model/train.py 60000

Saves ckpt.pt only when validation loss improves, so an interrupted run still
leaves the best model on disk.
"""
import os
import sys
import time

import numpy as np
import torch

from model import BLOCK_SIZE, FlowLM
from tokenizer import DATA, WordTokenizer

BATCH_SIZE = 32
LEARNING_RATE = 6e-4      # measured 1484 tok/s, so CPU time is the constraint;
                          # warmup + gradient clipping make this rate safe
EVAL_INTERVAL = 500
EVAL_ITERS = 40
WARMUP_ITERS = 200        # ramp the rate up so early steps can't blow up
MAX_ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
# Second argument picks the dataset: "" is the base story data, "chat_" is
# the conversation data used for fine-tuning.
PREFIX = sys.argv[2] if len(sys.argv) > 2 else ""

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "ckpt.pt")

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


def lr_at(it):
    """Linear warmup, then flat. Warmup stops the first few huge steps from
    wrecking a freshly initialised model."""
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / WARMUP_ITERS
    return LEARNING_RATE


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
        # On a different dataset the old best_val is not comparable, so
        # reset it — otherwise nothing would ever look like an improvement
        # and no checkpoint would be saved.
        best_val = float("inf") if PREFIX else ck["val_loss"]
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
                            "iter": it}, CKPT)
                star = "  <- saved"
            print(f"iter {it:6d}   train {losses['train']:.4f}   "
                  f"val {losses['val']:.4f}   {mins:6.1f} min   "
                  f"{tps:5.0f} tok/s{star}", flush=True)

        for g in optimizer.param_groups:
            g["lr"] = lr_at(it)

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
