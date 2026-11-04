"""Decompose the log-score gain of a message into generic, article-level and window-specific parts.
Input: out_pb/<group>_swap.npz from pb_mismatch.py (or pbx.py).

Population: recipient windows that lie inside one article and have another such window in the same article; every
component is computed on this one population, so generic + article + window = total exactly. Per window, with suffix
cross-entropy CE (nats/token):
  total            = CE(no context)          - CE(matched)
  generic          = CE(no context)          - CE(other-article donor)
  article-level    = CE(other-article donor) - CE(same-article donor)
  window-specific  = CE(same-article donor)  - CE(matched)
  content          = article-level + window-specific
These are differences in expected log-score through this receiver, not information quantities.

Intervals (4000 draws):
  "ci"      hierarchical bootstrap: resample test articles (all their windows enter together), and, inside each
            replicate, resample fitting seeds and donor draws with replacement; covers article, seed and donor variation
            (seed variation is estimated from only 2-3 fits);
  "ci_cond" conditional on the saved fits and donor draws: articles only.
Ratios are ratios of mean contrasts computed within each replicate.

    python pb_decompose.py <group> [<group> ...]   -> out_pb/<group>_decomp.json
"""
import json, os, re, sys
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_pb")
NB = 4000
NICE = {"text": "Text (prefix as tokens)", "receiver-state reference": "Receiver-state reference",
        "random map": "Random map", "weight-only anchors": "Weight-only anchors"}


def load(group):
    z = np.load(f"{OUT}/{group}_swap.npz")
    R = len([k for k in z.files if k.startswith("text|other r")])
    src = {}
    for k in z.files:
        if not k.endswith("|matched"):
            continue
        tag = k[:-len("|matched")]
        m = re.match(r"(full|comm)_(.+)_s(\d+)$", tag) or re.match(r"(.+)_(\d+k)_s(\d+)$", tag)
        if m and tag.startswith(("full_", "comm_")):
            name = f"{m.group(2)} [{m.group(1)} objective]"
        elif m:
            name = f"{m.group(1)} text {m.group(2)}"
        else:
            name = NICE.get(tag, tag)
        d = src.setdefault(name, {"m": [], "o": [], "s": []})
        d["m"].append(z[k])                                                   # (n,)
        d["o"].append(np.stack([z[f"{tag}|other r{r}"] for r in range(R)]))   # (R, n)
        d["s"].append(np.stack([z[f"{tag}|same r{r}"] for r in range(R)]))
    return z, {k: {kk: np.stack(v) for kk, v in d.items()} for k, d in src.items()}   # m: (S, n); o, s: (S, R, n)


def main(group):
    z, src = load(group)
    keep = (z["within_article"] > 0) & (z["has_same"] > 0)
    split = keep.sum() >= 100                     # too few same-document donors: report generic and content only
    if not split:
        keep = z["within_article"] > 0
    art = z["article"][keep]; none = z["no context"][keep]
    for d in src.values():
        d["m"] = d["m"][:, keep]; d["o"] = d["o"][:, :, keep]; d["s"] = d["s"][:, :, keep]
    arts = np.unique(art); members = [np.flatnonzero(art == a) for a in arts]
    rng = np.random.default_rng(0)
    A = [np.concatenate([members[j] for j in rng.integers(0, len(arts), len(arts))]) for _ in range(NB)]
    nS = max(d["m"].shape[0] for d in src.values()); nR = next(iter(src.values()))["o"].shape[1]
    SEEDS = [rng.integers(0, nS, nS) for _ in range(NB)]
    DONOR = [rng.integers(0, nR, nR) for _ in range(NB)]

    def parts(d, b=None, full=True):
        """per-window (total, generic, article, window) for source d in replicate b (None: point estimate). Without a
        same-document split, 'article' carries the whole content part and 'window' is zero."""
        nseed = d["m"].shape[0]
        if b is None or not full:
            m, o, s = d["m"].mean(0), d["o"].mean((0, 1)), d["s"].mean((0, 1))
        else:
            si = SEEDS[b] % nseed if nseed > 1 else np.zeros(1, int)
            ri = DONOR[b]
            m = d["m"][si].mean(0); o = d["o"][si][:, ri].mean((0, 1)); s = d["s"][si][:, ri].mean((0, 1))
        return (none - m, none - o, o - s, s - m) if split else (none - m, none - o, o - m, 0 * m)

    def stat(fn, names, full):
        est = fn(*[parts(src[n]) for n in names])
        bs = []
        for b in range(NB):
            ix = A[b]
            ps = [tuple(x[ix] for x in parts(src[n], b, full)) for n in names]
            bs.append(fn(*ps))
        return round(float(est), 6), [round(float(np.percentile(bs, 2.5)), 6), round(float(np.percentile(bs, 97.5)), 6)]

    def both(fn, names):
        e, ci = stat(fn, names, True); _, cc = stat(fn, names, False)
        return {"est": e, "ci": ci, "ci_cond": cc}

    res = {}
    comp = {"total": lambda p: p[0].mean(), "generic": lambda p: p[1].mean(), "article_level": lambda p: p[2].mean(),
            "window_specific": lambda p: p[3].mean(), "content": lambda p: (p[2] + p[3]).mean()}
    ref = "Receiver-state reference"
    for name, d in src.items():
        r = res[name] = {"n_seeds": int(d["m"].shape[0]),
                         "ce_matched": round(float(d["m"].mean()), 6)}
        for k, f in comp.items():
            r[k] = both(f, [name])
        if d["m"].shape[0] > 1:
            r["content_per_seed"] = [round(float(((d["o"][i].mean(0)) - d["m"][i]).mean()), 4)
                                     for i in range(d["m"].shape[0])]
        if ref in src:
            r["content_ratio_to_reference"] = both(lambda p, q: (p[2] + p[3]).mean() / (q[2] + q[3]).mean(), [name, ref])
            r["total_ratio_to_reference"] = both(lambda p, q: p[0].mean() / q[0].mean(), [name, ref])
            if d["m"].shape[0] > 1:
                rd = src[ref]
                rc = ((rd["o"].mean((0, 1)) - rd["m"].mean(0))).mean()
                r["content_ratio_per_seed"] = [round(float(((d["o"][i].mean(0)) - d["m"][i]).mean() / rc), 4)
                                               for i in range(d["m"].shape[0])]
    cons = {}
    names = list(src)
    for x in names:
        for y in names:
            fitted = lambda n: n not in set(NICE.values())           # contrasts between fitted bridges only
            if x != y and fitted(x) and fitted(y):
                cons[f"{x} minus {y}"] = {
                    "content": both(lambda p, q: ((p[2] + p[3]) - (q[2] + q[3])).mean(), [x, y]),
                    "generic": both(lambda p, q: (p[1] - q[1]).mean(), [x, y]),
                    "total": both(lambda p, q: (p[0] - q[0]).mean(), [x, y])}
    res["_contrasts"] = cons
    res["_meta"] = {"n_windows": int(keep.sum()), "n_windows_all": int(len(keep)), "n_articles": int(len(arts)),
                    "population": ("recipients inside one article with a same-article donor" if split else
                                   "all recipients inside one document (no same-document split: article_level = content)"),
                    "bootstrap": f"{NB} draws; ci: articles + seeds + donor draws; ci_cond: articles only",
                    "no_context_ce": round(float(none.mean()), 6)}
    json.dump(res, open(f"{OUT}/{group}_decomp.json", "w"), indent=1)
    for k, v in res.items():
        if not k.startswith("_"):
            f = lambda q: f"{q['est']:.3f} {q['ci']}"
            print(f"{k:34s} gen {f(v['generic'])} content {f(v['content'])} "
                  f"ratio {f(v['content_ratio_to_reference']) if 'content_ratio_to_reference' in v else ''}")
    return res


if __name__ == "__main__":
    for g in sys.argv[1:]:
        main(g)
