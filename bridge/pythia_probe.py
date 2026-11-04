"""Fit the pythia-70m -> pythia-160m bridge on a probe text file (same KL objective, start and step count as the
real-text fit in pythia.py), score it on the same wikitext test windows, and report verbatim overlap with the test text.
    python pythia_probe.py <label> <probe.txt>"""
import json, sys
import numpy as np
import torch
from datasets import load_dataset
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pythia as P

label, path = sys.argv[1:3]
rng = np.random.default_rng(0)
wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", revision=P.REV["Salesforce/wikitext"])
test_text = "".join(wt["test"]["text"][:6000])
te_ids = np.array(P.tok(test_text)["input_ids"])
test = P.windows(te_ids, 256, rng)                       # identical windows: same rng call order as pythia.py
probe = open(path).read()
subs = [probe[i:i + 50] for i in range(0, len(probe) - 50, 50)]
print(f"probe {len(probe)} chars; 50-char windows found verbatim in the wikitext test text: "
      f"{sum(s in test_text for s in subs)}/{len(subs)}")
ids = np.array(P.tok(probe)["input_ids"])
seqs = P.windows(ids, 4096, np.random.default_rng(1))
bridges = torch.load(f"{P.OUT}/bridges.pt")
M, c = P.fit(seqs, bridges["M_eu"].numpy(), np.zeros(P.dB))
r = P.score(test, M, c)
print(label, r)
res = json.load(open(f"{P.OUT}/results.json")); res[label] = r
json.dump(res, open(f"{P.OUT}/results.json", "w"), indent=1)
