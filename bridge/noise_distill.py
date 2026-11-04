"""Fairer knowledge-free baseline: blob-noise probes, ridge start, then fine-tune (M, c) with L-BFGS so that B's
softmax through the bridge matches B's softmax on the probe itself (the same objective Claude used on its drawings)."""
import json, os, sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from noise_probe import ridge, probes, OUT


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def distill(A, B, X, M0, c0):
    hA, hB = M0.shape
    H, target = C.h1(A, X), softmax(C.tail_from_z1(B, X @ B["W"][0] + B["b"][0]))

    def f(p):
        M, c = p[:hA * hB].reshape(hA, hB), p[hA * hB:]
        logp = np.log(softmax(C.tail_from_z1(B, H @ M + c)) + 1e-12)
        return -(target * logp).sum(1).mean()
    r = minimize(f, np.concatenate([M0.ravel(), c0]), method="L-BFGS-B", options={"maxiter": 300})   # numerical grads
    return r.x[:hA * hB].reshape(hA, hB), r.x[hA * hB:]


Xtr, Xte, ytr, yte = C.data()
rng = np.random.default_rng(1)
res = {}
for name in C.PAIRS:
    A, B = C.load(f"{OUT}/{name}_A.npz"), C.load(f"{OUT}/{name}_B.npz")
    X = probes("blobs", 3000, rng)
    M, c = distill(A, B, X, *ridge(A, B, X))
    res[name] = round(C.stitched_acc(A, B, M, c, Xte, yte), 3)
    print(name, res[name], flush=True)
json.dump(res, open(f"{OUT}/noise_distill.json", "w"), indent=1)
