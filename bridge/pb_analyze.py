"""Summaries with paired bootstrap intervals from out_pb/<group>.npz. Seeds are pooled by averaging the per-window
values of the three fits before resampling windows, so intervals reflect test-set sampling of the seed-averaged bridge;
the seed spread is reported separately.

    python pb_analyze.py <group> [<group> ...]   -> out_pb/<group>_summary.json and a printed table
"""
import json, os, re, sys
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_pb")
NB = 4000


def boot(fn, arrays, seed=0):
    """Point estimate and 95% interval of fn(*arrays) under joint resampling of test windows."""
    n = len(arrays[0]); rng = np.random.default_rng(seed)
    est = fn(*arrays)
    bs = [fn(*[a[ix] for a in arrays]) for ix in (rng.integers(0, n, n) for _ in range(NB))]
    return [round(float(est), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def conditions(z):
    """Group keys 'name | seed k|metric' into {name: {metric: [per-seed arrays]}}."""
    out = {}
    for k in z.files:
        cond, metric = k.rsplit("|", 1)
        m = re.match(r"(.*) \| seed (\d+)(.*)$", cond)
        name = (m.group(1) + m.group(3)) if m else cond
        out.setdefault(name, {}).setdefault(metric, []).append(z[k])
    return out


def summarise(group):
    z = np.load(f"{OUT}/{group}.npz")
    C = conditions(z)
    mean = lambda v: np.mean(v, 0)
    res = {}
    for name, d in C.items():
        acc, ce = mean(d["acc"]), mean(d["ce"])
        res[name] = {"n_seeds": len(d["acc"]), "acc": boot(lambda a: 100 * a.mean(), [acc]),
                     "ce": boot(lambda c: c.mean(), [ce])}
        if len(d["acc"]) > 1:
            res[name]["acc_seed_values"] = [round(100 * float(a.mean()), 2) for a in d["acc"]]
    if "B no context (suffix only)" in C:                         # comm groups
        ce0 = mean(C["B no context (suffix only)"]["ce"]); cefull = mean(C["B full context (upper bound)"]["ce"])
        orc = [k for k in C if k.startswith("oracle")]
        for name in C:
            mm = name + " | mismatched message"
            if mm not in C:
                continue
            ce, cem = mean(C[name]["ce"]), mean(C[mm]["ce"])
            acc, accm = mean(C[name]["acc"]), mean(C[mm]["acc"])
            r = res[name]
            r["doc_specific_ce_gain"] = boot(lambda a, b: (b - a).mean(), [ce, cem])           # nats/token
            r["doc_specific_acc_gain"] = boot(lambda a, b: 100 * (a - b).mean(), [acc, accm])
            r["generic_ce_gain"] = boot(lambda a, b: (a - b).mean(), [ce0, cem])
            r["share_of_full_context_ce_gain"] = boot(lambda a, b, f: (b - a).mean() / (b - f).mean(), [ce, ce0, cefull])
        # paired contrasts of document-specific gain between probe texts (same objective, same test windows)
        dsg = {n: mean(C[n + " | mismatched message"]["ce"]) - mean(C[n]["ce"])
               for n in C if n + " | mismatched message" in C}
        cons = {}
        for obj in ("full", "comm"):
            k = {t: f"fit[{obj}] on {t}" for t in ("real wikitext", "Claude text 383k", "self-sampled")}
            k = {t: v for t, v in k.items() if v in dsg}
            pairs = [("Claude text 383k", "self-sampled"), ("real wikitext", "Claude text 383k"),
                     ("real wikitext", "self-sampled")]
            for x, y in pairs:
                if x in k and y in k:
                    cons[f"{obj}: {x} minus {y}"] = boot(lambda p, q: (p - q).mean(), [dsg[k[x]], dsg[k[y]]])
            if "Claude text 383k" in k and "real wikitext" in k:
                cons[f"{obj}: Claude / real ratio"] = boot(lambda p, q: p.mean() / q.mean(),
                                                           [dsg[k["Claude text 383k"]], dsg[k["real wikitext"]]])
        res["_contrasts_doc_specific_ce_gain"] = cons
        for kk, vv in cons.items():
            print(f"  contrast {kk:50s} {vv}")
        res["_note"] = ("doc_specific = mismatched-message CE minus matched-message CE (>0: the bridge carries "
                        "information about this particular prefix); generic = no-context CE minus mismatched CE; "
                        "share_of_full_context = (no-context CE - CE) / (no-context CE - full-context CE)")
    json.dump(res, open(f"{OUT}/{group}_summary.json", "w"), indent=1)
    for k, v in res.items():
        if k.startswith("_"):
            continue
        extra = ""
        if "doc_specific_ce_gain" in v:
            extra = f"  doc-specific dCE {v['doc_specific_ce_gain']}  dAcc {v['doc_specific_acc_gain']}"
        print(f"{k:62s} acc {v['acc']}  ce {v['ce']}{extra}")
    return res


if __name__ == "__main__":
    for g in sys.argv[1:]:
        print("==", g); summarise(g)
