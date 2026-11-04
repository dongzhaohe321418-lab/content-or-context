"""Score the LLM-designed LM bridges (out/llm/lm_sandbox_r*_bridge.npz) on held-out text."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_common as L
from lm_taskfit import load_models

A, B = load_models()
tr, va = L.corpus()
x, y = L.eval_windows(va)
res = json.load(open(f"{L.OUT}/baselines.json"))
for f in sorted(glob.glob(os.path.join(os.path.dirname(L.OUT), "out", "llm", "lm_sandbox_r*_bridge.npz"))):
    z = np.load(f)
    r = L.score_logits(L.bridged(A, B, z["M"], z["c"], x), y)
    k = "Claude sandbox, designs everything, no corpus (" + os.path.basename(f).split("_")[2] + ")"
    res[k] = r
    print(k, r)
json.dump(res, open(f"{L.OUT}/baselines.json", "w"), indent=1)
