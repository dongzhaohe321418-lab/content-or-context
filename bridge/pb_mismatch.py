"""Message-swap controls for the communication setting, evaluated post hoc on saved bridges.

For every test window i and every message source (each saved bridge, the receiver-state reference, the weight-only and
random maps, and the prefix as text) we compute suffix cross-entropy with
  matched:       window i's own prefix;
  other article: the prefix of a window from a different WikiText article, for R independent random donor assignments;
  same article:  the prefix of another window of the same article, for R independent random assignments (windows whose
                 article has no other window are flagged and excluded from same-article contrasts).
Articles are delimited by top-level ' = Title = ' headings. Windows that cross an article boundary are flagged and are
never used as donors. Donor indices are saved.
(This replaces an earlier control that rolled messages by one inside each batch of 16: test windows are sorted by text
position, so 83% of those 'mismatched' donors came from the same article.)

    python pb_mismatch.py <group> <A> <B> <LA> <LB> [R=5]     -> out_pb/<group>_swap.npz and out_pb/<group>_swap.json
npz keys: '<source>|matched', '<source>|other r<k>', '<source>|same r<k>' (per-window CE, nats/token),
          'article', 'has_same', 'within_article', 'no context'.
"""
import glob, json, os, re, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb
from datasets import load_dataset

group, a, b, la, lb = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
R = int(sys.argv[6]) if len(sys.argv) > 6 else 5
P, S, N = 128, 32, 512
L = P + S + 1
lines = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", revision=pb.REV["Salesforce/wikitext"])["test"]["text"][:6000]
text = "".join(lines)
ids = pb.ids_of(text)
test = pb.disjoint_windows(ids, N, L, 123)                              # identical to pb_run.py

art_of_line = np.cumsum([1 if re.match(r"^ = [^=].* = \n$", l) else 0 for l in lines])
offs = np.array([o[0] for o in pb.tok(text, return_offsets_mapping=True)["offset_mapping"]])
assert len(offs) == len(ids)
line_start = np.cumsum([0] + [len(l) for l in lines[:-1]])
pick = np.sort(np.random.default_rng(123).choice(len(ids) // L, N, replace=False))
art_at = lambda t: art_of_line[np.searchsorted(line_start, offs[t], side="right") - 1]
art = art_at(pick * L + P - 1)                                          # article of the last prefix token
within = (art_at(pick * L) == art_at(pick * L + L - 1)).astype(np.float32)
rng = np.random.default_rng(2026)
# donors must lie inside one article (whole prefix and suffix); recipients are all evaluated, and analyses that need a
# clean population restrict to recipients that lie inside one article and have a same-article donor
inside = within > 0
other, same = [], []
for r in range(R):
    o = np.array([rng.choice(np.flatnonzero(inside & (art != art[i]))) for i in range(N)])
    sm = []
    for i in range(N):
        c = np.flatnonzero(inside & (art == art[i]) & (np.arange(N) != i))
        sm.append(rng.choice(c) if len(c) else i)
    assert not np.any(art[o] == art)
    other.append(torch.tensor(o, device=pb.DEV)); same.append(torch.tensor(np.array(sm), device=pb.DEV))
has_same = (same[0].cpu().numpy() != np.arange(N)).astype(np.float32)
audit = {"n_windows": N, "n_articles": int(len(set(art))), "windows_with_same_article_partner": int(has_same.sum()),
         "windows_within_one_article": int(within.sum()), "donor_assignments": R}
print(audit, flush=True)

pair = pb.Pair(a, b, la, lb)
B = pair.B
out = {"article": art, "has_same": has_same, "within_article": within}
for r in range(R):                                                      # donor indices, for audit
    out[f"donor_other r{r}"] = other[r].cpu().numpy(); out[f"donor_same r{r}"] = same[r].cpu().numpy()


@torch.no_grad()
def ce_suffix(lg, s):
    y = s[:, P + 1:]
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").view(y.shape).mean(1)


@torch.no_grad()
def eval_source(name, msgs=None):
    """msgs: (N, P, dB) messages, or None for the text channel (the donor's prefix tokens)."""
    def run(idx):
        res = []
        for i in range(0, N, 16):
            s = test[i:i + 16]
            j = torch.arange(i, min(i + 16, N), device=pb.DEV) if idx is None else idx[i:i + 16]
            if msgs is None:
                lg = pair.b_logits(torch.cat([test[j, :P], s[:, P:-1]], 1))[:, P:]
            else:
                lg = pb.comm_logits(pair, msgs[j], s, P)
            res.append(ce_suffix(lg, s))
        return torch.cat(res).cpu().numpy()
    out[f"{name}|matched"] = run(None)
    for r in range(R):
        out[f"{name}|other r{r}"] = run(other[r])
        out[f"{name}|same r{r}"] = run(same[r])
    mo = np.mean([out[f"{name}|other r{r}"].mean() for r in range(R)])
    ms = np.mean([out[f"{name}|same r{r}"].mean() for r in range(R)])
    print(f"{name:28s} matched {out[f'{name}|matched'].mean():.4f}  other {mo:.4f}  same {ms:.4f}", flush=True)


@torch.no_grad()
def messages(fn):
    return torch.cat([fn(test[i:i + 16]) for i in range(0, N, 16)])


with torch.no_grad():
    out["no context"] = torch.cat([ce_suffix(pair.b_logits(test[i:i + 16, P:-1]), test[i:i + 16])
                                   for i in range(0, N, 16)]).cpu().numpy()
eval_source("text")
eval_source("receiver-state reference", messages(lambda s: B(s[:, :P], output_hidden_states=True).hidden_states[lb]))
torch.manual_seed(0)
for name, br in [("random map", pb.Bridge(torch.randn(pair.dA, pair.dB) / pair.dA ** .5).to(pb.DEV)),
                 ("weight-only anchors", pb.Bridge(pair.anchors()).to(pb.DEV))]:
    eval_source(name, messages(lambda s, br=br: br(pair.hA(s[:, :P]))))
for f in sorted(glob.glob(f"{pb.OUT}/{group}_*_s*.pt")):
    tag = os.path.basename(f)[len(group) + 1:-3]                       # e.g. comm_real_s0
    br = pb.load_bridge(f, pair.dA, pair.dB)
    eval_source(tag, messages(lambda s, br=br: br(pair.hA(s[:, :P]))))
np.savez(f"{pb.OUT}/{group}_swap.npz", **out)
json.dump({"audit": audit, "mean_ce": {k: round(float(v.mean()), 4) for k, v in out.items()
                                       if not k.startswith(("article", "has_same", "within_article", "donor_"))}},
          open(f"{pb.OUT}/{group}_swap.json", "w"), indent=1)
