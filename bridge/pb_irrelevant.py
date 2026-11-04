"""Is the harm from an irrelevant message a property of the receiver, of the hidden-state interface, or of the bridge?
Per window, generic(source) - generic(text), with generic = CE(no context) - CE(other-article donor); article bootstrap
(conditional on the saved fits and donor draws). Also the share of windows whose seed- and donor-averaged CE with an
irrelevant message exceeds CE with no context.     python pb_irrelevant.py -> out_pb/irrelevant_contrast.json"""
import json, os
import numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_pb")
out = {}
for grp in ["comm70", "comm410_160"]:
    z = np.load(f"{OUT}/{grp}_swap.npz")
    keep = (z["within_article"] > 0) & (z["has_same"] > 0); art = z["article"][keep]; none = z["no context"][keep]
    arts = np.unique(art); mem = [np.flatnonzero(art == a) for a in arts]
    rng = np.random.default_rng(0)
    draws = [np.concatenate([mem[j] for j in rng.integers(0, len(arts), len(arts))]) for _ in range(4000)]
    oth = lambda tag: np.mean([z[f"{tag}|other r{r}"] for r in range(5)], 0)[keep]
    gen = {"text": none - oth("text"), "reference": none - oth("receiver-state reference")}
    for src in ["real", "Claude", "self-sampled"]:
        gen[src] = np.mean([none - oth(f"comm_{src}_s{s}") for s in range(3)], 0)
    res = {}
    for k, v in gen.items():
        d = v - gen["text"]; bs = [d[ix].mean() for ix in draws]
        res[k] = {"generic": round(float(v.mean()), 4), "minus_text": round(float(d.mean()), 4),
                  "ci": [round(float(x), 4) for x in np.percentile(bs, [2.5, 97.5])],
                  "share_windows_hurt": round(float((v < 0).mean()), 3)}
    out[grp] = res
    print(grp, res)
json.dump(out, open(f"{OUT}/irrelevant_contrast.json", "w"), indent=1)
