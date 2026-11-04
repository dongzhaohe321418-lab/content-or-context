"""Receiver-state reference vs injection depth: inject B's own layer-LB prefix states (B reading the prefix alone) into B's
suffix pass at layer LB, for pythia-160m and pythia-70m; 128 test windows. Diagnostic only (no fitting, no selection).
    python pb_ceiling.py > out_pb/ceiling.txt      (per-window arrays: out_pb/ceiling.npz)"""
import sys, numpy as np, torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import pb
from datasets import load_dataset
wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", revision=pb.REV["Salesforce/wikitext"])
te = pb.ids_of("".join(wt["test"]["text"][:6000]))
P, S = 128, 32
test = pb.disjoint_windows(te, 128, P + S + 1, 123)
ARR = {}
for name in ["pythia-160m", "pythia-70m"]:
    for lb in [1, 2, 3, 4, 6, 8]:
        if name == "pythia-70m" and lb > 5: continue
        pair = pb.Pair(name, name, lb, lb)
        B = pair.B
        if lb == 1:
            a, c = pb.score(lambda s: (pair.b_logits(s[:, :-1])[:, P:], s[:, P+1:]), test); print(name, "full ctx", 100*a.mean(), c.mean())
            ARR[f"{name}|full|acc"], ARR[f"{name}|full|ce"] = a, c
            a, c = pb.score(lambda s: (pair.b_logits(s[:, P:-1]), s[:, P+1:]), test); print(name, "no ctx", 100*a.mean(), c.mean())
            ARR[f"{name}|none|acc"], ARR[f"{name}|none|ce"] = a, c
        def oracle(s, perm=False):
            h = B(s[:, :P], output_hidden_states=True).hidden_states[lb]
            if perm: h = h.roll(1, 0)
            return pb.comm_logits(pair, h, s, P), s[:, P+1:]
        a, c = pb.score(oracle, test); a2, c2 = pb.score(lambda s: oracle(s, True), test)
        ARR[f"{name}|L{lb}|acc"], ARR[f"{name}|L{lb}|ce"] = a, c
        print(name, f"oracle at L{lb}: acc {100*a.mean():.2f} ce {c.mean():.3f} | mismatched acc {100*a2.mean():.2f} ce {c2.mean():.3f}", flush=True)
np.savez(__import__("os").path.join(pb.OUT, "ceiling.npz"), **ARR)
