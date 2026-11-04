"""Retrospective reconstruction of the out-of-domain test windows used by pb_domain.py (same inputs, seeds and selection
rules, no model forward passes): for each window, document index, document name (file name for code, IMDB test index
for reviews) and token offset. The original run saved document ids but not offsets; we verify the document ids against
out_pb/dom_<domain>_comm70_swap.npz, and the offsets follow from the deterministic selection.
    python pb_domain_windows.py   -> out_pb/dom_<domain>_windows.json"""
import glob, json, os, sys, sysconfig
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb
L, N, PER_DOC = 161, 512, 4
for domain in ("code", "reviews"):
    if domain == "code":
        files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
        docs, names = [open(f, errors="replace").read() for f in files], [os.path.basename(f) for f in files]
    else:
        from datasets import load_dataset
        docs = load_dataset("stanfordnlp/imdb", revision=pb.REV["stanfordnlp/imdb"])["test"]["text"]; names = [f"imdb-test-{i}" for i in range(len(docs))]
    rng = np.random.default_rng(123); cands = []
    for di, d in enumerate(docs):
        k = len(pb.ids_of(d)) // L
        if k < 2:
            continue
        for j in sorted(rng.choice(k, min(k, PER_DOC), replace=False)):
            cands.append((di, int(j) * L))
    pick = np.sort(rng.choice(len(cands), min(N, len(cands)), replace=False))
    rows = [{"doc_index": cands[i][0], "doc_name": names[cands[i][0]], "token_offset": cands[i][1]} for i in pick]
    saved = np.load(f"{pb.OUT}/dom_{domain}_comm70_swap.npz")["article"]
    assert (np.array([r["doc_index"] for r in rows]) == saved).all(), "window identities do not match saved run"
    json.dump(rows, open(f"{pb.OUT}/dom_{domain}_windows.json", "w"), indent=0)
    print(domain, len(rows), "windows verified against saved run")
