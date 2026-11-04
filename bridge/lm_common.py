"""Latent bridge between two tiny character-level GPTs (mlx). Same corpus and vocabulary, different seeds and widths.
A runs blocks 1..K (K = SPLIT) and sends its residual stream r1A (one vector per position) to B:
    r1B_hat = r1A @ M + c
B continues from there with block 2, its final norm and its unembedding. Score: B's next-character accuracy (and
loss) on held-out text, against B alone, A alone and the bridges that do not involve the LLM.
"""
import glob, json, os
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out_lm")
CTX = 64
CHARS = "\n" + "".join(chr(i) for i in range(32, 127))        # 96 symbols; anything else -> '?'
STOI = {ch: i for i, ch in enumerate(CHARS)}
PAIR = {"A": {"d": 64, "seed": 11}, "B": {"d": 96, "seed": 12}}
LAYERS, SPLIT = 4, 2


def corpus():
    """Python standard-library source as text (offline, plenty, same for both models)."""
    import sysconfig
    files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    txt = "".join(open(f, errors="replace").read() for f in files)
    ids = np.array([STOI.get(ch, STOI["?"]) for ch in txt], dtype=np.int32)
    n = len(ids)
    return ids[: int(0.9 * n)], ids[int(0.9 * n):]


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiHeadAttention(d, h)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def __call__(self, x, mask):
        y = self.ln1(x)
        x = x + self.attn(y, y, y, mask)
        return x + self.fc2(nn.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, d, heads=4, layers=LAYERS, vocab=len(CHARS)):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(CTX, d)
        self.blocks = [Block(d, heads) for _ in range(layers)]
        self.lnf, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)

    def mask(self, T):
        return nn.MultiHeadAttention.create_additive_causal_mask(T)

    def r0(self, idx):
        return self.tok(idx) + self.pos(mx.arange(idx.shape[1]))

    def r1(self, idx):
        """residual stream after the first SPLIT blocks (the message A sends)"""
        x, m = self.r0(idx), self.mask(idx.shape[1])
        for b in self.blocks[:SPLIT]:
            x = b(x, m)
        return x

    def from_r1(self, r1):
        x, m = r1, self.mask(r1.shape[1])
        for b in self.blocks[SPLIT:]:
            x = b(x, m)
        return self.head(self.lnf(x))

    def __call__(self, idx):
        return self.from_r1(self.r1(idx))


def batches(ids, bs, rng):
    while True:
        ix = rng.integers(0, len(ids) - CTX - 1, bs)
        x = np.stack([ids[i:i + CTX] for i in ix]); y = np.stack([ids[i + 1:i + CTX + 1] for i in ix])
        yield mx.array(x), mx.array(y)


def train(d, seed, tr, steps=6000, bs=64, lr=1e-3):
    mx.random.seed(seed)
    m = GPT(d)
    opt = optim.AdamW(learning_rate=optim.cosine_decay(lr, steps))
    loss_fn = lambda m, x, y: nn.losses.cross_entropy(m(x), y).mean()
    step = nn.value_and_grad(m, loss_fn)
    g = batches(tr, bs, np.random.default_rng(seed))
    for s in range(steps):
        x, y = next(g)
        loss, grads = step(m, x, y)
        opt.update(m, grads)
        mx.eval(m.parameters(), opt.state)
        if s % 1000 == 0 or s == steps - 1:
            print(f"  d={d} step {s} loss {loss.item():.3f}", flush=True)
    return m


def eval_windows(va, n=400, seed=0):
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(va) - CTX - 1, n)
    return mx.array(np.stack([va[i:i + CTX] for i in ix])), mx.array(np.stack([va[i + 1:i + CTX + 1] for i in ix]))


def score_logits(logits, y):
    acc = (logits.argmax(-1) == y).astype(mx.float32).mean().item()
    loss = nn.losses.cross_entropy(logits, y).mean().item()
    return {"acc": round(acc, 4), "loss": round(loss, 4)}


def bridged(A, B, M, c, x):
    r = A.r1(x) @ mx.array(M, dtype=mx.float32) + mx.array(c, dtype=mx.float32)
    return B.from_r1(r)


def export(m, path):
    """weights as numpy (the only thing the LLM gets), names spelled out."""
    from mlx.utils import tree_flatten
    np.savez(path, **{k: np.array(v) for k, v in tree_flatten(m.parameters())})
