"""Robustness check for the Pythia bridge: every probe-text condition is fitted with 3 seeds (window draw + batch order)
and scored on a larger, fresh test set (512 wikitext test windows, disjoint draw from the 256 used before). Per-window
accuracies are saved so conditions can be compared with a paired bootstrap over test windows.
Adds the missing fair control: real wikitext of the same size as Claude's 3x text (383k chars).

    python pythia_seeds.py
"""
import json, os, sys, time
import numpy as np
import torch
from datasets import load_dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pythia as P

PR = f"{P.OUT}/probes"
SEEDS = [0, 1, 2]


def per_window_acc(seqs, M, c):
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), 16):
            s = seqs[i:i + 16]; x, y = s[:, :-1], s[:, 1:]
            lg = P.b_logits(x, P.hA(x) @ M + c)
            out.append((lg.argmax(-1) == y).float().mean(1).cpu().numpy())
    return np.concatenate(out)


def ids_of(text):
    return np.array(P.tok(text)["input_ids"])


def main():
    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", revision=P.REV["Salesforce/wikitext"])
    te_ids = ids_of("".join(wt["test"]["text"][:6000]))
    test = P.windows(te_ids, 512, np.random.default_rng(123))
    train = "".join(wt["train"]["text"][:30000])
    claude3x = open(f"{PR}/all3x.txt").read()
    conds = {
        "real wikitext pool (8.8M chars)": ids_of(train),
        "real wikitext 127k chars": ids_of(open(f"{PR}/real_matched.txt").read()),
        "real wikitext 383k chars": ids_of(train[:len(claude3x)]),
        "Claude text 127k chars": ids_of(open(f"{PR}/all.txt").read()),
        "Claude text 383k chars": ids_of(claude3x),
    }
    M0 = torch.load(f"{P.OUT}/bridges.pt")["M_eu"].numpy()
    res, acc = {}, {}
    with torch.no_grad():
        acc["B alone"] = np.concatenate([
            (P.b_logits(test[i:i + 16, :-1]).argmax(-1) == test[i:i + 16, 1:]).float().mean(1).cpu().numpy()
            for i in range(0, len(test), 16)])
    for name in list(conds) + ["self-sampled from A and B"]:
        for sd in SEEDS:
            t0 = time.time()
            if name.startswith("self"):
                samp = torch.cat([P.self_sample(m, 128, 10 * sd + j) for j, m in enumerate([P.A, P.A, P.B, P.B])])
                seqs = samp[:, :P.T + 1]
            else:
                seqs = P.windows(conds[name], 4096, np.random.default_rng(100 + sd))
            M, c = P.fit(seqs, M0, np.zeros(P.dB), seed=sd)
            a = per_window_acc(test, M, c)
            acc[f"{name} | seed {sd}"] = a
            res.setdefault(name, []).append(round(float(a.mean()), 4))
            print(f"{name:34s} seed {sd}: {a.mean():.4f}  ({time.time() - t0:.0f} s)", flush=True)
            np.savez(f"{P.OUT}/seeds_per_window.npz", **acc)
            json.dump(res, open(f"{P.OUT}/seeds.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
