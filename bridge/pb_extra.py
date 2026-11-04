"""Post-hoc analyses of saved comm-setting bridges (no further fitting), on the same test windows as pb_run.py.

  1. the receiver-state reference ("oracle": B's own prefix states) with a donor prefix;
  2. per-suffix-position content gain (matched versus a donor from another article);
  3. equal numbers of transmitted positions: the message restricted to the last K prefix positions, against B reading
     the last K prefix tokens as text, for K in {1, 2, 4, 8, 16, 32, 64, 128}, each with its donor control.
Mismatched = the message (or prefix text) of the first other-article donor drawn by pb_mismatch.py (run it first).

    python pb_extra.py <group> <A> <B> <LA> <LB>      e.g.  python pb_extra.py comm70 pythia-70m pythia-160m 2 1
writes out_pb/<group>_extra.npz (per-window, per-position arrays) and out_pb/<group>_extra.json
"""
import glob, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb
from datasets import load_dataset

group, a, b, la, lb = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
P, S = 128, 32
KS = [1, 2, 4, 8, 16, 32, 64, 128]
wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", revision=pb.REV["Salesforce/wikitext"])
LINES = wt["test"]["text"][:6000]
IDS = pb.ids_of("".join(LINES))
test = pb.disjoint_windows(IDS, 512, P + S + 1, 123)                     # as pb_run.py
# donors: the first random other-article draw of pb_mismatch.py (donors lie inside one article), so panels share a control
FAR = torch.tensor(np.load(f"{pb.OUT}/{group}_swap.npz")["donor_other r0"], device=pb.DEV)
pair = pb.Pair(a, b, la, lb)
B = pair.B


@torch.no_grad()
def pos_ce(lg, y):
    """Per-window, per-position cross-entropy (n, S)."""
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").view(y.shape)


@torch.no_grad()
def run(fn, bs=16):
    return torch.cat([fn(test[i:i + bs]) for i in range(0, len(test), bs)]).cpu().numpy()


@torch.no_grad()
def msg_logits(msg, s, K):
    """B's suffix logits given a message for the last K prefix positions only."""
    m = msg[:, P - K:]
    h = torch.cat([m, pb.suffix_states(pair, s, P)], 1)
    dummy = s[:, P - K:-1].clone(); dummy[:, :K] = pb.EOS
    return pair.b_logits(dummy, h)[:, K:]


y = lambda s: s[:, P + 1:]


@torch.no_grad()
def run_msg(msgs, K, mismatched):
    """per-window, per-position CE with the last K positions of each window's message (or of a far window's)."""
    out = []
    for i in range(0, len(test), 16):
        s = test[i:i + 16]
        m = msgs[FAR[i:i + 16]] if mismatched else msgs[i:i + 16]
        out.append(pos_ce(msg_logits(m, s, K), y(s)))
    return torch.cat(out).cpu().numpy()


@torch.no_grad()
def all_msgs(fn):
    return torch.cat([fn(test[i:i + 16]) for i in range(0, len(test), 16)])


out = {}
out["B full context"] = run(lambda s: pos_ce(pair.b_logits(s[:, :-1])[:, P:], y(s)))
out["B no context"] = run(lambda s: pos_ce(pair.b_logits(s[:, P:-1]), y(s)))
pre = test[:, :P]
for K in KS:
    out[f"text K={K}"] = run(lambda s: pos_ce(pair.b_logits(s[:, P - K:-1])[:, K:], y(s)))
    mm = []
    for i in range(0, len(test), 16):
        s = test[i:i + 16]
        with torch.no_grad():
            mm.append(pos_ce(pair.b_logits(torch.cat([pre[FAR[i:i + 16], P - K:], s[:, P:-1]], 1))[:, K:], y(s)))
    out[f"text K={K} | mismatched"] = torch.cat(mm).cpu().numpy()
orc = all_msgs(lambda s: B(s[:, :P], output_hidden_states=True).hidden_states[lb])
for K in KS:
    out[f"oracle K={K}"] = run_msg(orc, K, False)
    out[f"oracle K={K} | mismatched"] = run_msg(orc, K, True)
for f in sorted(glob.glob(f"{pb.OUT}/{group}_*_s*.pt")):
    tag = os.path.basename(f)[len(group) + 1:-3]                    # e.g. comm_real_s0
    br = pb.Bridge(torch.zeros(pair.dA, pair.dB), hidden=0).to(pb.DEV)
    br.load_state_dict(torch.load(f)); br.eval()
    msgs = all_msgs(lambda s: br(pair.hA(s[:, :P])))
    for K in KS:
        out[f"{tag} K={K}"] = run_msg(msgs, K, False)
        out[f"{tag} K={K} | mismatched"] = run_msg(msgs, K, True)
    print(tag, {K: round(float(out[f"{tag} K={K} | mismatched"].mean() - out[f"{tag} K={K}"].mean()), 4)
                for K in KS}, flush=True)
np.savez(f"{pb.OUT}/{group}_extra.npz", **out)
json.dump({k: round(float(v.mean()), 4) for k, v in out.items()}, open(f"{pb.OUT}/{group}_extra.json", "w"), indent=1)
print(json.dumps({k: round(float(v.mean()), 4) for k, v in out.items() if "s1" not in k and "s2" not in k}, indent=1))
