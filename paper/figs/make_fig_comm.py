"""Figure 1: what a latent message is worth to the receiver (pythia-70m L2 -> pythia-160m L1, prefix 128, suffix 32,
512 test windows from 58 WikiText-103 articles).

Claim: almost all of a message's benefit is content, and a bridge fitted without in-domain data delivers most of the
content that the receiver's own prefix states would; per transmitted position, text remains worth more.
  (a) generic / article-level / window-specific decomposition on one population (recipients inside one article with
      a same-article donor), values and content intervals from pb_decompose.py (five donor draws of each kind;
      hierarchical bootstrap over articles, fitting seeds and donor draws, 4000 draws).
  (b) content gain per suffix position; (c) content gain when only the last K prefix positions are sent
      (pb_extra.py, first other-article donor draw; pointwise 95% bands from 4000 article-clustered draws).

    python make_fig_comm.py <out_pb dir> [<dir with audit_panel_alignment.py>, default: this directory]
"""
import json, os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT_PB = sys.argv[1]
SKILL = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SKILL)
from audit_panel_alignment import require_matplotlib_panel_alignment

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7, "axes.linewidth": 0.6,
    "axes.spines.right": False, "axes.spines.top": False, "legend.frameon": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})
NEUTRAL, NEUTRAL_L = "#4D4D4D", "#CFCECE"
C = {"text": "#272727", "ref": "#767676", "real": "#0F4D92", "llm": "#B64342", "self": "#42949E"}

dec = json.load(open(f"{OUT_PB}/comm70_decomp.json"))
extra = np.load(f"{OUT_PB}/comm70_extra.npz")
art = np.load(f"{OUT_PB}/comm70_swap.npz")["article"]
arts = np.unique(art); members = [np.flatnonzero(art == a) for a in arts]
rng = np.random.default_rng(0)      # bootstrap resampling of real test articles (no simulated data)
DRAWS = [np.concatenate([members[j] for j in rng.integers(0, len(arts), len(arts))]) for _ in range(4000)]

rows = [("Random map", "Random map", NEUTRAL_L), ("Weight-only anchors", "Weight-only anchors", NEUTRAL_L),
        ("Self-samples", "self-sampled [full objective]", C["self"]),
        ("Self-samples", "self-sampled [comm objective]", C["self"]),
        ("LLM-written text", "Claude [full objective]", C["llm"]),
        ("LLM-written text", "Claude [comm objective]", C["llm"]),
        ("Real text", "real [full objective]", C["real"]),
        ("Real text", "real [comm objective]", C["real"]),
        ("Receiver-state reference", "Receiver-state reference", C["ref"]),
        ("Text (prefix as tokens)", "Text (prefix as tokens)", C["text"])]
objective = ["", "", "hand-over", "comm", "hand-over", "comm", "hand-over", "comm", "", ""]

src = {"a": []}
fig = plt.figure(figsize=(5.5, 4.9))                    # drawn at the manuscript text width: no scaling on inclusion
gsa = fig.add_gridspec(1, 1, left=0.35, right=0.985, bottom=0.53, top=0.965)     # long condition labels need room
gsb = fig.add_gridspec(1, 2, left=0.1, right=0.985, bottom=0.085, top=0.43, wspace=0.34)
ax = fig.add_subplot(gsa[0, 0])
y = np.arange(len(rows))[::-1]
for yi, (lab, key, col), obj in zip(y, rows, objective):
    d = dec[key]
    g, art_l, win = d["generic"]["est"], d["article_level"]["est"], d["window_specific"]["est"]
    cont = [d["content"]["est"]] + d["content"]["ci"]
    ax.barh(yi, g, color=col, alpha=0.25, height=0.66, lw=0)
    x0 = max(g, 0)
    ax.barh(yi, art_l, left=x0, color=col, alpha=0.6, height=0.66, lw=0)
    ax.barh(yi, win, left=x0 + art_l, color=col, height=0.66, lw=0)
    end = x0 + cont[0]
    ax.errorbar(end, yi, xerr=[[cont[0] - cont[1]], [cont[2] - cont[0]]], fmt="none", ecolor="#272727",
                elinewidth=0.6, capsize=1.5, capthick=0.6)
    src["a"].append({"condition": lab, "objective": obj, "generic": d["generic"], "article_level": d["article_level"],
                     "window_specific": d["window_specific"], "content": cont})
ax.axvline(0, color="#272727", lw=0.6)
ax.set_yticks(y); ax.set_yticklabels([f"{r[0]} ({o})" if o else r[0] for r, o in zip(rows, objective)])
ax.set_xlabel("CE reduction vs. no context (nats/token)")
ax.set_xlim(-0.8, 1.5)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.legend(handles=[Patch(color=NEUTRAL, alpha=0.25, lw=0, label="Generic"),
                   Patch(color=NEUTRAL, alpha=0.6, lw=0, label="Article-level"),
                   Patch(color=NEUTRAL, lw=0, label="Window-specific")],
          loc="upper right", fontsize=5.5, handlelength=1.2, borderaxespad=0.1)


def cluster_band(d2):
    """d2: (n_windows, m) -> mean and pointwise 95% article-clustered interval over windows."""
    b = np.stack([d2[ix].mean(0) for ix in DRAWS])
    return d2.mean(0), np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)


def content(key, K, seeds=None):
    if seeds:
        return np.mean([extra[f"{key}_s{s} K={K} | mismatched"] - extra[f"{key}_s{s} K={K}"] for s in range(seeds)], 0)
    return extra[f"{key} K={K} | mismatched"] - extra[f"{key} K={K}"]


series = [("Text", "text", None, C["text"], "-"), ("Receiver-state ref.", "oracle", None, C["ref"], "--"),
          ("Real text (comm)", "comm_real", 3, C["real"], "-"), ("LLM text (comm)", "comm_Claude", 3, C["llm"], "-"),
          ("Self-samples (hand-over)", "full_self-sampled", 3, C["self"], "-")]
axb = fig.add_subplot(gsb[0, 0])
pos = np.arange(1, 33)
src["b"] = {}
for lab, key, sd, col, ls in series:
    m, lo, hi = cluster_band(content(key, 128, sd))
    axb.fill_between(pos, np.clip(lo, 0.05, 6), np.clip(hi, 0.05, 6), color=col, alpha=0.15, lw=0)
    axb.plot(pos, m, color=col, ls=ls, lw=1.0, label=lab)
    src["b"][lab] = {"mean": [round(float(v), 4) for v in m], "lo": [round(float(v), 4) for v in lo],
                     "hi": [round(float(v), 4) for v in hi]}
axb.set_yscale("log"); axb.set_ylim(0.05, 6)
axb.set_yticks([0.1, 0.3, 1, 3]); axb.set_yticklabels(["0.1", "0.3", "1", "3"]); axb.minorticks_off()
axb.set_xlim(1, 32); axb.set_xticks([1, 8, 16, 24, 32])
axb.set_xlabel("Suffix position"); axb.set_ylabel("Content gain (nats)")
axb.legend(loc="lower left", fontsize=5.5, handlelength=1.6, borderaxespad=0.1)

axc = fig.add_subplot(gsb[0, 1])
KS = [1, 2, 4, 8, 16, 32, 64, 128]
src["c"] = {}
for lab, key, sd, col, ls in series[:4]:
    ms, los, his = [], [], []
    for K in KS:
        m, lo, hi = cluster_band(content(key, K, sd).mean(1, keepdims=True))
        ms.append(m[0]); los.append(lo[0]); his.append(hi[0])
    axc.fill_between(KS, los, his, color=col, alpha=0.15, lw=0)
    axc.plot(KS, ms, color=col, ls=ls, lw=1.0, marker="o", ms=2.2, label=lab)
    src["c"][lab] = {"K": KS, "mean": [round(float(v), 4) for v in ms],
                     "ci": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(los, his)]}
axc.set_xscale("log", base=2); axc.set_xticks(KS); axc.set_xticklabels([str(k) for k in KS]); axc.minorticks_off()
axc.set_ylim(-0.1, 1.5)
axc.set_xlabel("Prefix positions sent (K)"); axc.set_ylabel("Content gain (nats)")
axc.legend(loc="upper left", fontsize=5.5, handlelength=1.6, borderaxespad=0.1)

for a_, lab in [(ax, "a"), (axb, "b"), (axc, "c")]:
    a_.annotate(lab, xy=(0, 1), xycoords="axes fraction", xytext=(-4, 3), textcoords="offset points",
                fontsize=8, fontweight="bold", ha="right", va="bottom")
fig.canvas.draw()
here = os.path.dirname(os.path.abspath(__file__))
require_matplotlib_panel_alignment(
    fig, json_out=f"{here}/fig_comm.alignment.json", overlay_svg=f"{here}/fig_comm.alignment.svg",
    tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True, panel_ids=["a", "b", "c"])
json.dump(src, open(f"{here}/fig_comm.source.json", "w"), indent=1)
fig.savefig(f"{here}/fig_comm.pdf", bbox_inches="tight")
fig.savefig(f"{here}/fig_comm.svg", bbox_inches="tight")
fig.savefig(f"{here}/fig_comm.tiff", dpi=600, bbox_inches="tight")
fig.savefig(f"{here}/fig_comm.png", dpi=300, bbox_inches="tight")
