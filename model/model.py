"""FlowLM — Flow's language model. A decoder-only transformer, from zero.

Same architecture Or built for Tiny-GPT (Head -> MultiHeadAttention ->
FeedForward -> Block), scaled from 0.8M to ~9M parameters and pointed at a
word-level vocabulary instead of characters.

Run `.venv/bin/python model/model.py` to check the wiring before training.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

# --- hyperparameters (see docs/superpowers/specs/2026-09-09-flow-design.md)
BLOCK_SIZE = 128   # how many tokens of context
N_EMBD = 256      # size of each token's vector
N_HEAD = 8        # attention heads, 32 dims each
N_LAYER = 6       # transformer blocks
DROPOUT = 0.1


class Head(nn.Module):
    """One head of self-attention.

    Every position asks a question (query) and every position advertises what
    it has (key). Matching them says how much each position should listen to
    each earlier one. Then it collects value vectors in those proportions.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        # A lower-triangular matrix of ones. Registered as a buffer, not a
        # parameter: it is fixed furniture, never trained.
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)

        # How much each position should attend to each other position.
        # Scaling by 1/sqrt(head_size) keeps the numbers small enough that
        # softmax stays soft instead of collapsing onto one position.
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)

        # The causal mask: position t may not see t+1. Without this line the
        # model can read the answer it is being asked to predict, loss drops
        # to near zero, and generation produces garbage. It is the single most
        # common bug in a from-scratch GPT.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        return wei @ v   # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """Several heads in parallel. Each can learn a different relationship."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Per-position thinking time.

    Attention moves information between positions; this layer processes it.
    The 4x inner width is the standard transformer ratio.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Communication (attention) then computation (feed-forward).

    Note `x = x + ...`: these are residual connections. The block learns a
    change to add, not a replacement. Without them a 4-layer network trains
    badly; with them gradients reach the early layers intact.
    """

    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention(N_HEAD, N_EMBD // N_HEAD)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # pre-norm: normalize, then transform
        x = x + self.ffwd(self.ln2(x))
        return x


class FlowLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        # Attention has no built-in sense of order, so position is learned
        # separately and added in.
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Small random weights. Too large and the untrained loss starts far
        # above ln(vocab_size) — which is exactly what eval Gate 1 catches.
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)                                  # (B,T,C)
        pos = self.position_embedding(torch.arange(T, device=idx.device))  # (T,C)
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                         # (B,T,vocab)

        if targets is None:
            return logits, None

        # cross_entropy wants (N, classes), so flatten batch and time together.
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        """Predict one token, append it, repeat.

        temperature < 1 sharpens the distribution (safer, more repetitive);
        top_k keeps only the k most likely tokens before sampling. Without
        top_k, the long tail of 8,192 tokens each has a small chance, and
        those rare wrong picks are most of what reads as gibberish.
        """
        for _ in range(max_new_tokens):
            # Position embeddings only exist up to BLOCK_SIZE, so only ever
            # feed the model the last BLOCK_SIZE tokens.
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            # Sample, don't argmax — argmax makes it repeat one phrase forever.
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next.item()


if __name__ == "__main__":
    import math

    torch.manual_seed(1337)
    vocab_size = 8192
    m = FlowLM(vocab_size)
    n_params = sum(p.numel() for p in m.parameters())

    x = torch.randint(0, vocab_size, (4, BLOCK_SIZE))
    logits, loss = m(x, x)
    expected = math.log(vocab_size)

    embed = m.token_embedding.weight.numel() + m.lm_head.weight.numel()
    print(f"parameters:     {n_params:,}  ({embed:,} in embeddings)")
    print(f"logits shape:   {tuple(logits.shape)}  (expected (4, {BLOCK_SIZE}, {vocab_size}))")
    print(f"untrained loss: {loss.item():.4f}")
    print(f"ln({vocab_size}) =         {expected:.4f}")
    # Window around ln(8192)=9.011: an untrained model must guess
    # uniformly over the vocabulary, whatever size that vocabulary is.
    ok = 8.7 < loss.item() < 9.3
    print(f"\nGATE 1 (wiring): {'PASS' if ok else 'FAIL'}")
