"""Train the two tiny GPTs, then score the bridges that do not involve the LLM."""
import json, os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_common as L


def lsq(Pa, Pb, lam=1e-6):
    """M minimising ||Pa M - Pb||^2 (+ small ridge)."""
    return np.linalg.solve(Pa.T @ Pa + lam * np.eye(Pa.shape[1]), Pa.T @ Pb)


def main():
    os.makedirs(L.OUT, exist_ok=True)
    tr, va = L.corpus()
    print("corpus", len(tr), len(va))
    models = {}
    for k, spec in L.PAIR.items():
        models[k] = L.train(spec["d"], spec["seed"], tr)
        L.export(models[k], f"{L.OUT}/{k}.npz")
        models[k].save_weights(f"{L.OUT}/{k}.safetensors")
    A, B = models["A"], models["B"]
    x, y = L.eval_windows(va)
    res = {"A_alone": L.score_logits(A(x), y), "B_alone": L.score_logits(B(x), y)}
    dA, dB = L.PAIR["A"]["d"], L.PAIR["B"]["d"]
    rng = np.random.default_rng(0)
    res["random"] = L.score_logits(L.bridged(A, B, rng.normal(0, 1 / np.sqrt(dA), (dA, dB)), np.zeros(dB), x), y)
    EA, EB = np.array(A.tok.weight), np.array(B.tok.weight)
    PA, PB = np.array(A.pos.weight), np.array(B.pos.weight)
    UA, UB = np.array(A.head.weight), np.array(B.head.weight)          # vocab x d
    # data-free 1: token + position embeddings are shared "inputs" -> align the layer-0 bases
    M = lsq(np.vstack([EA, PA]), np.vstack([EB, PB]))
    res["embed-anchor LSQ (data-free)"] = L.score_logits(L.bridged(A, B, M, np.zeros(dB), x), y)
    np.savez(f"{L.OUT}/bridge_embed.npz", M=M, c=np.zeros(dB))
    # data-free 2: also ask the unembedding directions to agree (rows of U are read-out directions over the same vocab)
    M2 = lsq(np.vstack([EA, PA, UA]), np.vstack([EB, PB, UB]))
    res["embed+unembed LSQ (data-free)"] = L.score_logits(L.bridged(A, B, M2, np.zeros(dB), x), y)
    # upper bound: ridge regression of r1B on r1A over training windows
    xt, _ = L.eval_windows(tr, n=2000, seed=1)
    RA = np.array(A.r1(xt)).reshape(-1, dA); RB = np.array(B.r1(xt)).reshape(-1, dB)
    H = np.hstack([RA, np.ones((len(RA), 1))])
    Wfit = np.linalg.solve(H.T @ H + 1e-3 * np.eye(dA + 1), H.T @ RB)
    res["data-fit ridge (upper bound)"] = L.score_logits(L.bridged(A, B, Wfit[:-1], Wfit[-1], x), y)
    for k, v in res.items():
        print(f"{k:34s} {v}")
    json.dump(res, open(f"{L.OUT}/baselines.json", "w"), indent=1)


if __name__ == "__main__":
    main()
