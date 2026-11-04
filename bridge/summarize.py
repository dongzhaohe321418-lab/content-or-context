"""One table per experiment from the raw result files (no hand-copied numbers)."""
import json, os, re
import numpy as np

R = os.path.dirname(os.path.abspath(__file__))
base = json.load(open(f"{R}/out/baselines.json"))
noise = json.load(open(f"{R}/out/noise_probe.json"))
llm = json.load(open(f"{R}/out/llm_scores.json"))


def arm(pair, pat):
    v = [r["acc"] for r in llm if r["run"].startswith(pair + "_") and re.search(pat, r["run"]) and r["acc"] is not None]
    return v


print("DIGITS: B's test accuracy through the bridge (A sends h1, B continues)\n")
cols = ["same", "width", "hetero"]
rows = [("A alone", lambda p: [base[p]["A_alone"]]), ("B alone", lambda p: [base[p]["B_alone"]]),
        ("random bridge", lambda p: [base[p]["random(mean of 20)"]]),
        ("cosine unit matching (data-free algorithm)", lambda p: [base[p]["cosine-match (data-free)"]]),
        ("weight-space least squares (data-free algorithm)", lambda p: [base[p]["weight-LSQ (data-free)"]]),
        ("ridge on uniform-noise probes (no domain knowledge)", lambda p: [noise[p]["uniform"]]),
        ("ridge on smooth-blob probes (no domain knowledge)", lambda p: [noise[p]["blobs"]]),
        ("Claude, tool-free, single response (3 runs)", lambda p: arm(p, r"toolfree")),
        ("Claude, code on weights only, no inputs at all (1 run)", lambda p: arm(p, r"sandbox_r9$")),
        ("Claude, code + self-made probes, no data (2 runs)", lambda p: arm(p, r"sandbox_r[01]$")),
        ("ridge on the real training images (upper bound)", lambda p: [base[p]["data-fit ridge (upper bound)"]])]
print(f"{'':56s}" + "".join(f"{c:>18s}" for c in cols))
for name, f in rows:
    cells = []
    for p in cols:
        v = f(p)
        cells.append("—" if not v else (f"{v[0]:.3f}" if len(v) == 1 else f"{np.mean(v):.3f} [{min(v):.2f}-{max(v):.2f}]"))
    print(f"{name:56s}" + "".join(f"{c:>18s}" for c in cells))

lm = json.load(open(f"{R}/out_lm/baselines.json"))
print("\nTINY LMs: B's next-character accuracy on held-out Python (A sends its residual stream after block 2 of 4)\n")
for k, v in lm.items():
    print(f"{k:60s} acc {v['acc']:.3f}  loss {v['loss']:.3f}")
