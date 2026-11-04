"""Implementation checks for the communication protocol (pythia-70m L2 -> pythia-160m L1, a saved bridge).

  1. dummy invariance: the tokens that fill B's prefix positions in pass 2 are overwritten at layer L_B, so replacing
     them (EOS -> random tokens) must leave the suffix logits unchanged;
  2. prefix isolation: with the message held fixed, replacing the prefix tokens of the window must leave the suffix
     logits unchanged (B can learn about the prefix only through the message);
  3. causality: replacing suffix token j must leave the logits at earlier suffix positions unchanged.

    python pb_checks.py  -> out_pb/checks.json (maximum absolute logit differences)
"""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb
from datasets import load_dataset

P, S = 128, 32
wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", revision=pb.REV["Salesforce/wikitext"])
test = pb.disjoint_windows(pb.ids_of("".join(wt["test"]["text"][:6000])), 512, P + S + 1, 123)[:32]
pair = pb.Pair("pythia-70m", "pythia-160m", 2, 1)
br = pb.Bridge(torch.zeros(pair.dA, pair.dB)).to(pb.DEV)
br.load_state_dict(torch.load(f"{pb.OUT}/comm70_comm_real_s0.pt")); br.eval()
g = torch.Generator().manual_seed(0)
rand_like = lambda x: torch.randint(0, 50000, x.shape, generator=g).to(pb.DEV)
res = {}
with torch.no_grad():
    msg = br(pair.hA(test[:, :P]))
    base = pb.comm_logits(pair, msg, test, P)
    # 1. dummy invariance: rerun pass 2 with random dummy tokens
    h = torch.cat([msg, pb.suffix_states(pair, test, P)], 1)
    dummy = test[:, :-1].clone(); dummy[:, :P] = rand_like(dummy[:, :P])
    res["dummy_invariance_max_abs_logit_diff"] = float((pair.b_logits(dummy, h)[:, P:] - base).abs().max())
    # 2. prefix isolation: same message, different prefix tokens in the window
    s2 = test.clone(); s2[:, :P] = rand_like(s2[:, :P])
    res["prefix_isolation_max_abs_logit_diff"] = float((pb.comm_logits(pair, msg, s2, P) - base).abs().max())
    # 3. causality: change suffix input token j; logits at suffix positions < j must not move
    diffs = []
    for j in (5, 16, 27):
        s3 = test.clone(); s3[:, P + j] = rand_like(s3[:, P + j])
        out = pb.comm_logits(pair, msg, s3, P)
        diffs.append({"j": j, "earlier_positions_max_diff": float((out[:, :j] - base[:, :j]).abs().max()),
                      "position_j_and_later_max_diff": float((out[:, j:] - base[:, j:]).abs().max())})
    res["causality"] = diffs
    res["logit_scale_reference"] = float(base.abs().mean())
json.dump(res, open(f"{pb.OUT}/checks.json", "w"), indent=1)
print(json.dumps(res, indent=1))
