"""Paired comparison of two senders into the same receiver on identical windows and donors (comm410_160 vs comm70):
change in content contrast D and in net benefit T, with a hierarchical bootstrap that resamples articles, fitting
seeds and donor draws. Seeds and donors are paired: fit s of either sender uses the same probe-window and minibatch
seeds, and both senders are scored against the same donor draws.   python pb_sender.py -> out_pb/sender_compare.json"""
import json, os
import numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_pb")
a, b = np.load(f"{OUT}/comm410_160_swap.npz"), np.load(f"{OUT}/comm70_swap.npz")
assert (a["article"] == b["article"]).all() and all((a[f"donor_other r{r}"] == b[f"donor_other r{r}"]).all() for r in range(5))
keep = (a["within_article"] > 0) & (a["has_same"] > 0); art = a["article"][keep]; none = a["no context"][keep]
arts = np.unique(art); mem = [np.flatnonzero(art == x) for x in arts]
rng = np.random.default_rng(0); NB = 4000
res = {}
for src in ["real", "Claude", "self-sampled"]:
    def arrays(z):
        m = np.stack([z[f"comm_{src}_s{s}|matched"] for s in range(3)])[:, keep]
        o = np.stack([np.stack([z[f"comm_{src}_s{s}|other r{r}"] for r in range(5)]) for s in range(3)])[:, :, keep]
        return m, o
    (ma, oa), (mb, ob) = arrays(a), arrays(b)
    def stats(m1, o1, m2, o2, ix, s1, s2, r1, r2):
        D = lambda m, o, s, r: (o[s][:, r].mean((0, 1)) - m[s].mean(0))[ix].mean()
        T = lambda m, s: (none - m[s].mean(0))[ix].mean()
        return D(m1, o1, s1, r1) - D(m2, o2, s2, r2), T(m1, s1) - T(m2, s2)
    all_s, all_r, all_ix = np.arange(3), np.arange(5), np.arange(len(art))
    est = stats(ma, oa, mb, ob, all_ix, all_s, all_s, all_r, all_r)
    bs = []
    for _ in range(NB):
        ix = np.concatenate([mem[j] for j in rng.integers(0, len(arts), len(arts))])
        s, r = rng.integers(0, 3, 3), rng.integers(0, 5, 5)
        bs.append(stats(ma, oa, mb, ob, ix, s, s, r, r))
    bs = np.array(bs)
    res[src] = {"delta_content": round(float(est[0]), 4), "delta_content_ci": [round(float(x), 4) for x in np.percentile(bs[:, 0], [2.5, 97.5])],
                "delta_net": round(float(est[1]), 4), "delta_net_ci": [round(float(x), 4) for x in np.percentile(bs[:, 1], [2.5, 97.5])]}
    print(src, res[src])
json.dump(res, open(f"{OUT}/sender_compare.json", "w"), indent=1)
