"""Linear bridge trained on B's own next-character loss (gradient descent on M, c only; A and B frozen).
With real text this is the upper bound for a linear bridge; with other text (e.g. LLM-written probes) it is that arm's
bridge.   python lm_taskfit.py <name> [probe_text_file]"""
import json, os, sys
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_common as L


class Bridge(nn.Module):
    def __init__(self, M, c):
        super().__init__()
        self.M, self.c = mx.array(M, dtype=mx.float32), mx.array(c, dtype=mx.float32)

    def __call__(self, r):
        return r @ self.M + self.c


def load_models():
    A, B = L.GPT(L.PAIR["A"]["d"]), L.GPT(L.PAIR["B"]["d"])
    A.load_weights(f"{L.OUT}/A.safetensors"); B.load_weights(f"{L.OUT}/B.safetensors")
    A.freeze(); B.freeze()
    return A, B


def fit(A, B, ids, M0, c0, steps=3000, bs=64, lr=3e-3, seed=0):
    br = Bridge(M0, c0)
    opt = optim.Adam(learning_rate=lr)
    rA_cache = {}
    loss_fn = lambda br, ra, y: nn.losses.cross_entropy(B.from_r1(br(ra)), y).mean()
    step = nn.value_and_grad(br, loss_fn)
    g = L.batches(ids, bs, np.random.default_rng(seed))
    for s in range(steps):
        x, y = next(g)
        loss, grads = step(br, mx.stop_gradient(A.r1(x)), y)
        opt.update(br, grads)
        mx.eval(br.parameters(), opt.state)
    return np.array(br.M), np.array(br.c)


def main():
    name = sys.argv[1]
    A, B = load_models()
    tr, va = L.corpus()
    ids = tr
    if len(sys.argv) > 2:
        txt = open(sys.argv[2], errors="replace").read()
        ids = np.array([L.STOI.get(ch, L.STOI["?"]) for ch in txt], dtype=np.int32)
        print("probe text:", len(ids), "characters")
    dA, dB = L.PAIR["A"]["d"], L.PAIR["B"]["d"]
    z = np.load(f"{L.OUT}/bridge_embed.npz")
    M, c = fit(A, B, ids, z["M"], z["c"])
    x, y = L.eval_windows(va)
    r = L.score_logits(L.bridged(A, B, M, c, x), y)
    print(name, r)
    np.savez(f"{L.OUT}/bridge_{name}.npz", M=M, c=c)
    res = json.load(open(f"{L.OUT}/baselines.json"))
    res[name] = r
    json.dump(res, open(f"{L.OUT}/baselines.json", "w"), indent=1)


if __name__ == "__main__":
    main()
