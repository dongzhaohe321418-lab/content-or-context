"""Out-of-domain evaluation of saved bridges (no refitting). The bridges were fitted for WikiText-like text; here the
test windows come from another domain:
  code:     Python standard-library source files (one file = one document);
  reviews:  IMDB movie reviews (stanfordnlp/imdb test split; one review = one document).
Only documents long enough for at least two non-overlapping 161-token windows are eligible (at most 4 windows each);
the final 512 windows are sampled from all eligible windows, so many recipients (most, for reviews) have no
same-document donor, and pb_decompose.py then reports generic and content parts without the article/window split. Donor controls exactly as in pb_mismatch.py (five other-document and five same-document
draws), and the output has the same format, so pb_decompose.py applies unchanged.

    python pb_domain.py <domain> <group> <A> <B> <LA> <LB>    -> out_pb/dom_<domain>_<group>_swap.npz
"""
import glob, json, os, sys, sysconfig
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb

domain, group, a, b, la, lb = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
P, S, N, R, PER_DOC = 128, 32, 512, 5, 4
L = P + S + 1
if domain == "code":
    docs = [open(f, errors="replace").read() for f in sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))]
elif domain == "reviews":
    from datasets import load_dataset
    docs = load_dataset("stanfordnlp/imdb", revision=pb.REV["stanfordnlp/imdb"])["test"]["text"]
else:
    raise ValueError(domain)
rng = np.random.default_rng(123)
cands = []                                                                  # (doc id, window token ids)
for di, d in enumerate(docs):
    ids = pb.ids_of(d)
    k = len(ids) // L
    if k < 2:
        continue
    for j in sorted(rng.choice(k, min(k, PER_DOC), replace=False)):
        cands.append((di, ids[j * L:(j + 1) * L]))
pick = np.sort(rng.choice(len(cands), min(N, len(cands)), replace=False))
art = np.array([cands[i][0] for i in pick]); test = torch.tensor(np.stack([cands[i][1] for i in pick]), device=pb.DEV)
n = len(test)
other, same = [], []
drng = np.random.default_rng(2026)
for r in range(R):
    other.append(torch.tensor([drng.choice(np.flatnonzero(art != art[i])) for i in range(n)], device=pb.DEV))
    sm = []
    for i in range(n):
        c = np.flatnonzero((art == art[i]) & (np.arange(n) != i))
        sm.append(drng.choice(c) if len(c) else i)
    same.append(torch.tensor(np.array(sm), device=pb.DEV))
has_same = (same[0].cpu().numpy() != np.arange(n)).astype(np.float32)
audit = {"domain": domain, "documents_used": int(len(set(art))), "windows": n, "with_same_doc_partner": int(has_same.sum())}
print(audit, flush=True)

pair = pb.Pair(a, b, la, lb)
B = pair.B
out = {"article": art, "has_same": has_same, "within_article": np.ones(n, np.float32)}
for r in range(R):
    out[f"donor_other r{r}"] = other[r].cpu().numpy(); out[f"donor_same r{r}"] = same[r].cpu().numpy()


@torch.no_grad()
def ce_suffix(lg, s):
    y = s[:, P + 1:]
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").view(y.shape).mean(1)


@torch.no_grad()
def eval_source(name, msgs=None):
    def run(idx):
        res = []
        for i in range(0, n, 16):
            s = test[i:i + 16]
            j = torch.arange(i, min(i + 16, n), device=pb.DEV) if idx is None else idx[i:i + 16]
            lg = pair.b_logits(torch.cat([test[j, :P], s[:, P:-1]], 1))[:, P:] if msgs is None \
                else pb.comm_logits(pair, msgs[j], s, P)
            res.append(ce_suffix(lg, s))
        return torch.cat(res).cpu().numpy()
    out[f"{name}|matched"] = run(None)
    for r in range(R):
        out[f"{name}|other r{r}"] = run(other[r]); out[f"{name}|same r{r}"] = run(same[r])
    print(f"{name:28s} matched {out[f'{name}|matched'].mean():.4f}  other "
          f"{np.mean([out[f'{name}|other r{r}'].mean() for r in range(R)]):.4f}", flush=True)


@torch.no_grad()
def messages(fn):
    return torch.cat([fn(test[i:i + 16]) for i in range(0, n, 16)])


with torch.no_grad():
    out["no context"] = torch.cat([ce_suffix(pair.b_logits(test[i:i + 16, P:-1]), test[i:i + 16])
                                   for i in range(0, n, 16)]).cpu().numpy()
eval_source("text")
eval_source("receiver-state reference", messages(lambda s: B(s[:, :P], output_hidden_states=True).hidden_states[lb]))
eval_source("weight-only anchors", messages(lambda s, br=pb.Bridge(pair.anchors()).to(pb.DEV): br(pair.hA(s[:, :P]))))
for f in sorted(glob.glob(f"{pb.OUT}/{group}_comm_*_s*.pt")):
    tag = os.path.basename(f)[len(group) + 1:-3]
    br = pb.Bridge(torch.zeros(pair.dA, pair.dB)).to(pb.DEV)
    br.load_state_dict(torch.load(f)); br.eval()
    eval_source(tag, messages(lambda s, br=br: br(pair.hA(s[:, :P]))))
np.savez(f"{pb.OUT}/dom_{domain}_{group}_swap.npz", **out)
json.dump(audit, open(f"{pb.OUT}/dom_{domain}_{group}_audit.json", "w"), indent=1)
