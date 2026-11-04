"""Seed robustness for the tiny-GPT task-loss fits: refit each probe-text condition with batch-order seeds 0-2 (same
start, steps and validation windows as lm_taskfit.py) and report mean and spread. Writes out_lm/seeds.json.
    python lm_seeds.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_common as L
from lm_taskfit import fit, load_models

A, B = load_models()
tr, va = L.corpus()
enc = lambda p: np.array([L.STOI.get(ch, L.STOI["?"]) for ch in open(p, errors="replace").read()], dtype=np.int32)
conds = {"real text (upper bound)": tr,
         "Claude-written probe text": enc(f"{L.OUT}/probes/all.txt"),
         "self-sampled from A and B (no LLM)": enc(f"{L.OUT}/probes/selfsample.txt"),
         "shuffled probe characters (control)": enc(f"{L.OUT}/probes/shuffled.txt")}
z = np.load(f"{L.OUT}/bridge_embed.npz")
x, y = L.eval_windows(va)
res = {}
for name, ids in conds.items():
    for sd in (0, 1, 2):
        M, c = fit(A, B, ids, z["M"], z["c"], seed=sd)
        r = L.score_logits(L.bridged(A, B, M, c, x), y)
        res.setdefault(name, []).append(r["acc"])
        print(f"{name:38s} seed {sd}: {r}", flush=True)
        json.dump(res, open(f"{L.OUT}/seeds.json", "w"), indent=1)
for k, v in res.items():
    print(f"{k:38s} mean {np.mean(v):.4f}  sd {np.std(v, ddof=1):.4f}  {v}")
