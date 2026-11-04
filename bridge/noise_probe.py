"""Knowledge-free probe baseline: fit the bridge (ridge) on synthetic inputs that carry no idea of what a digit
looks like: uniform noise, sparse random pixels, and smooth random blobs."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from run_baselines import OUT


def ridge(A, B, X, lam=1e-3):
    H = np.hstack([C.h1(A, X), np.ones((len(X), 1))])
    Z = X @ B["W"][0] + B["b"][0]
    W = np.linalg.solve(H.T @ H + lam * np.eye(H.shape[1]), H.T @ Z)
    return W[:-1], W[-1]


def probes(kind, n, rng):
    if kind == "uniform":
        return rng.random((n, 64))
    if kind == "sparse":
        return (rng.random((n, 64)) < 0.3).astype(float)
    g = rng.normal(size=(n, 8, 8))                      # smooth blobs: blur white noise, rescale to [0, 1]
    k = np.array([0.25, 0.5, 0.25])
    for ax in (1, 2):
        g = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), ax, g)
    g = (g - g.min((1, 2), keepdims=True)) / np.ptp(g, axis=(1, 2), keepdims=True)
    return (g > 0.6).astype(float).reshape(n, 64) * g.reshape(n, 64)


Xtr, Xte, ytr, yte = C.data()
rng = np.random.default_rng(0)
res = {}
for name in C.PAIRS:
    A, B = C.load(f"{OUT}/{name}_A.npz"), C.load(f"{OUT}/{name}_B.npz")
    res[name] = {k: round(C.stitched_acc(A, B, *ridge(A, B, probes(k, 15000, rng)), Xte, yte), 3)
                 for k in ("uniform", "sparse", "blobs")}
    print(name, res[name])
json.dump(res, open(f"{OUT}/noise_probe.json", "w"), indent=1)
