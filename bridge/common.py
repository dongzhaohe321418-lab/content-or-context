"""Shared pieces of the latent-bridge experiment: data, the model pairs, the stitched forward pass and scoring.

A bridge sends model A's first hidden layer (post-activation, h1A) to model B as B's first-layer PRE-activation:
    z1B_hat = h1A @ M + c          (M: hA x hB, c: hB)
B then applies its own activation and its own remaining layers. Accuracy of B's prediction on the test set is the
"communication" score. Nothing here is shown to the LLM except what render_prompt() writes.
"""
import json
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

PAIRS = {
    # name: (A spec, B spec); spec = (hidden sizes, activation, seed)
    "same":   (((16, 16), "relu", 1), ((16, 16), "relu", 2)),
    "width":  (((16, 16), "relu", 3), ((24, 24), "relu", 4)),
    "hetero": (((16, 16), "relu", 5), ((24, 16), "tanh", 6)),
}
ACT = {"relu": lambda z: np.maximum(z, 0), "tanh": np.tanh}


def data():
    d = load_digits()
    X, y = d.data / 16.0, d.target
    return train_test_split(X, y, test_size=0.33, random_state=0, stratify=y)


def train(spec, Xtr, ytr):
    hs, act, seed = spec
    m = MLPClassifier(hidden_layer_sizes=hs, activation=act, max_iter=3000, random_state=seed, alpha=1e-4)
    m.fit(Xtr, ytr)
    return {"W": [w.copy() for w in m.coefs_], "b": [b.copy() for b in m.intercepts_], "act": act}


def h1(net, X):
    return ACT[net["act"]](X @ net["W"][0] + net["b"][0])


def tail_from_z1(net, z1):
    h = ACT[net["act"]](z1)
    for W, b in zip(net["W"][1:-1], net["b"][1:-1]):
        h = ACT[net["act"]](h @ W + b)
    return h @ net["W"][-1] + net["b"][-1]


def predict(net, X):
    return tail_from_z1(net, X @ net["W"][0] + net["b"][0]).argmax(1)


def stitched_acc(A, B, M, c, X, y):
    return float((tail_from_z1(B, h1(A, X) @ M + c).argmax(1) == y).mean())


def save(net, path):
    np.savez(path, act=net["act"], **{f"W{i}": w for i, w in enumerate(net["W"])},
             **{f"b{i}": b for i, b in enumerate(net["b"])})


def load(path):
    z = np.load(path)
    n = sum(k.startswith("W") for k in z.files)
    return {"W": [z[f"W{i}"] for i in range(n)], "b": [z[f"b{i}"] for i in range(n)], "act": str(z["act"])}


def fmt(a):
    a = np.atleast_2d(a)
    return "\n".join(" ".join(f"{v:.3f}" for v in row) for row in a)


def describe(net, name):
    hs = [w.shape[1] for w in net["W"][:-1]]
    lines = [f"MODEL {name}: 64 inputs -> " + " -> ".join(f"{h} ({net['act']})" for h in hs) + " -> 10 logits",
             f"  forward: h1 = {net['act']}(x @ W1 + b1); "
             + "; ".join(f"h{i+2} = {net['act']}(h{i+1} @ W{i+2} + b{i+2})" for i in range(len(hs) - 1))
             + f"; logits = h{len(hs)} @ W{len(hs)+1} + b{len(hs)+1}"]
    for i, (W, b) in enumerate(zip(net["W"], net["b"])):
        lines.append(f"\n{name}.W{i+1}, shape {W.shape[0]} x {W.shape[1]} (row r = input/unit r of the previous layer):")
        lines.append(fmt(W))
        lines.append(f"\n{name}.b{i+1}, shape {b.shape[0]}:")
        lines.append(fmt(b))
    return "\n".join(lines)


def parse_bridge(text, hA, hB):
    """the last ```json block with {"M": [[...] x hA], "c": [... hB]}"""
    import re
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    for blk in reversed(blocks):
        try:
            d = json.loads(blk)
            M, c = np.array(d["M"], dtype=float), np.array(d["c"], dtype=float)
            if M.shape == (hA, hB) and c.shape == (hB,):
                return M, c
        except Exception:
            continue
    return None
