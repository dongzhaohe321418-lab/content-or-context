"""Train the model pairs, then score every bridge that does not involve the LLM:
random, identity (same width only), two data-free weight-only algorithms, and the data-fitted upper bound."""
import json, os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def cosine_match(A, B):
    """data-free: each B unit copies the A unit whose incoming pixel weights point the same way (one-to-one by the
    Hungarian method when the widths agree), rescaled by the weight norms; offset from the biases."""
    WA, WB = A["W"][0], B["W"][0]
    nA, nB = np.linalg.norm(WA, axis=0), np.linalg.norm(WB, axis=0)
    S = (WA / nA).T @ (WB / nB)                       # hA x hB cosine
    hA, hB = S.shape
    if hA == hB:
        rows, cols = linear_sum_assignment(-S)
        pick = dict(zip(cols, rows))
    else:
        pick = {j: int(S[:, j].argmax()) for j in range(hB)}
    M, c = np.zeros((hA, hB)), B["b"][0].copy()
    for j, i in pick.items():
        s = S[i, j] * nB[j] / nA[i]
        M[i, j] = s
        c[j] -= s * A["b"][0][i]
    return M, c


def weight_lsq(A, B):
    """data-free: pretend h1A were linear in x and solve W1A @ M = W1B over the 64 input directions."""
    M, *_ = np.linalg.lstsq(A["W"][0], B["W"][0], rcond=None)
    return M, B["b"][0] - A["b"][0] @ M


def data_fit(A, B, X, lam=1e-3):
    """upper bound with data: ridge regression of B's first-layer pre-activation on A's h1 over the training set."""
    H = np.hstack([C.h1(A, X), np.ones((len(X), 1))])
    Z = X @ B["W"][0] + B["b"][0]
    W = np.linalg.solve(H.T @ H + lam * np.eye(H.shape[1]), H.T @ Z)
    return W[:-1], W[-1]


def main():
    os.makedirs(OUT, exist_ok=True)
    Xtr, Xte, ytr, yte = C.data()
    rng = np.random.default_rng(0)
    res = {}
    for name, (sa, sb) in C.PAIRS.items():
        A, B = C.train(sa, Xtr, ytr), C.train(sb, Xtr, ytr)
        C.save(A, f"{OUT}/{name}_A.npz"); C.save(B, f"{OUT}/{name}_B.npz")
        hA, hB = A["W"][0].shape[1], B["W"][0].shape[1]
        r = {"A_alone": float((C.predict(A, Xte) == yte).mean()), "B_alone": float((C.predict(B, Xte) == yte).mean())}
        rand = [C.stitched_acc(A, B, rng.normal(0, 1 / np.sqrt(hA), (hA, hB)) * np.linalg.norm(B["W"][0], axis=0).mean(),
                               B["b"][0], Xte, yte) for _ in range(20)]
        r["random(mean of 20)"] = float(np.mean(rand))
        if hA == hB:
            r["identity (no alignment)"] = C.stitched_acc(A, B, np.eye(hA), np.zeros(hB), Xte, yte)
        for k, f in [("cosine-match (data-free)", cosine_match), ("weight-LSQ (data-free)", weight_lsq)]:
            M, c = f(A, B)
            r[k] = C.stitched_acc(A, B, M, c, Xte, yte)
            np.savez(f"{OUT}/{name}_bridge_{k.split()[0]}.npz", M=M, c=c)
        M, c = data_fit(A, B, Xtr)
        r["data-fit ridge (upper bound)"] = C.stitched_acc(A, B, M, c, Xte, yte)
        res[name] = r
        print(name, json.dumps(r, indent=1))
    json.dump(res, open(f"{OUT}/baselines.json", "w"), indent=1)


if __name__ == "__main__":
    main()
