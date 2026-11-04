"""Equal numbers of transmitted positions: content gain of K text tokens minus that of the real-text comm bridge
restricted to the last K prefix positions (mean over its three fitting seeds), for K in {1, ..., 128}, on all 512 test
windows with the first other-article donor (pb_extra.py). Intervals resample articles (4000 draws) and are conditional
on the saved bridges and that donor; Bonferroni-adjusted over the eight values of K.
    python pb_bandwidth.py -> out_pb/bandwidth_test.json"""
import json, os
import numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_pb")
z, w = np.load(f"{OUT}/comm70_extra.npz"), np.load(f"{OUT}/comm70_swap.npz")
art = w["article"]; arts = np.unique(art); mem = [np.flatnonzero(art == x) for x in arts]
KS = [1, 2, 4, 8, 16, 32, 64, 128]; NB = 4000; alpha = 0.05 / len(KS)
content = lambda k: (z[f"{k} | mismatched"] - z[k]).mean(-1)
diff = {K: content(f"text K={K}") - np.mean([content(f"comm_real_s{s} K={K}") for s in range(3)], 0) for K in KS}
rng = np.random.default_rng(0)
boot = [np.concatenate([mem[j] for j in rng.integers(0, len(arts), len(arts))]) for _ in range(NB)]
res = {}
for K in KS:
    bs = np.array([diff[K][ix].mean() for ix in boot])
    lo, hi = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    res[f"K={K}"] = {"text_minus_bridge": round(float(diff[K].mean()), 6),
                     "ci95": [round(float(x), 6) for x in np.percentile(bs, [2.5, 97.5])],
                     "bonferroni_ci": [round(float(lo), 6), round(float(hi), 6)]}
    print(K, res[f"K={K}"])
res["smallest_lower_bound"] = min(v["bonferroni_ci"][0] for v in res.values())
json.dump(res, open(f"{OUT}/bandwidth_test.json", "w"), indent=1)
