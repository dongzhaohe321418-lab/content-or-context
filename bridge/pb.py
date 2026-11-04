"""General latent-bridge toolkit for Pythia pairs (shared tokenizer, independent pretraining runs).

A sends its residual stream after layer LA; B's residual stream after layer LB is overwritten (forward hook) at chosen
positions by bridge(h_A). Two settings:

  full  - B sees the tokens, but every position's layer-LB state is replaced by the bridged A state (the setting of
          pythia.py). Measures how much of B's computation survives the hand-over.
  comm  - information asymmetry. A reads a P-token prefix that B never sees; B reads only the S-token suffix (its own
          prefix positions hold EOS tokens) and receives A's prefix states, bridged, at layer LB. Measures whether the
          bridge transmits document-specific information B does not have.

Bridges are fitted by matching B's own next-token distribution (KL) on probe text; nothing is fitted on test text.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, GPTNeoXForCausalLM

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_pb")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
# Hugging Face revisions used in the study (MANIFEST.md); every model and dataset load is pinned to them.
REV = {"EleutherAI/pythia-70m": "a39f36b100fe8a5377810d56c3f4789b9c53ac42",
       "EleutherAI/pythia-160m": "50f5173d932e8e61f858120bcb800b97af589f46",
       "EleutherAI/pythia-410m": "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
       "openai-community/gpt2": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
       "Qwen/Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987",
       "facebook/opt-125m": "27dcfa74d334bc871f3234de431e71c6eeba5dd6",
       "Salesforce/wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3",
       "stanfordnlp/imdb": "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"}
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", revision=REV["EleutherAI/pythia-70m"])
EOS = tok.eos_token_id
_cache = {}


def load(name):
    if name not in _cache:
        m = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/{name}", revision=REV[f"EleutherAI/{name}"],
                                               torch_dtype=torch.float32).to(DEV).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        m._inject = {"h": None, "n": None}

        def hook(mod, inp, out, m=m):
            h, n = m._inject["h"], m._inject["n"]
            if h is None:
                return out
            x = out[0] if isinstance(out, tuple) else out
            x = torch.cat([h, x[:, h.shape[1]:]], 1) if n is not None else h
            return (x,) + tuple(out[1:]) if isinstance(out, tuple) else x

        m._hook = hook
        m._handle = None
        _cache[name] = m
    return _cache[name]


class Pair:
    def __init__(self, a, b, la, lb):
        self.A, self.B, self.la, self.lb = load(a), load(b), la, lb
        self.names = (a, b)
        if self.B._handle is not None:
            self.B._handle.remove()
        self.B._handle = self.B.gpt_neox.layers[lb - 1].register_forward_hook(self.B._hook)
        self.dA, self.dB = self.A.config.hidden_size, self.B.config.hidden_size

    def hA(self, ids):
        return self.A(ids, output_hidden_states=True).hidden_states[self.la]

    def b_logits(self, ids, h=None, prefix=False):
        self.B._inject.update(h=h, n=(h.shape[1] if (h is not None and prefix) else None))
        try:
            return self.B(ids).logits
        finally:
            self.B._inject.update(h=None, n=None)

    def anchors(self):
        """Data-free least-squares map between the two models' token embedding + unembedding matrices."""
        EA = torch.cat([self.A.gpt_neox.embed_in.weight, self.A.get_output_embeddings().weight]).cpu()
        EB = torch.cat([self.B.gpt_neox.embed_in.weight, self.B.get_output_embeddings().weight]).cpu()
        return torch.linalg.lstsq(EA, EB).solution


class Bridge(nn.Module):
    def __init__(self, M0, hidden=0):
        super().__init__()
        dA, dB = M0.shape
        self.M = nn.Parameter(torch.as_tensor(M0, dtype=torch.float32).clone())
        self.c = nn.Parameter(torch.zeros(dB))
        self.mlp = None
        if hidden:
            self.mlp = nn.Sequential(nn.LayerNorm(dA), nn.Linear(dA, hidden), nn.GELU(), nn.Linear(hidden, dB))
            nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h):
        y = h @ self.M + self.c
        return y + self.mlp(h) if self.mlp is not None else y


class LowRankBridge(nn.Module):
    """h @ U @ V + c with rank r, initialised from the truncated SVD of M0."""
    def __init__(self, M0, rank):
        super().__init__()
        U, Sv, Vh = torch.linalg.svd(torch.as_tensor(M0, dtype=torch.float32), full_matrices=False)
        self.U = nn.Parameter((U[:, :rank] * Sv[:rank].sqrt()).clone())
        self.V = nn.Parameter((Sv[:rank].sqrt()[:, None] * Vh[:rank]).clone())
        self.c = nn.Parameter(torch.zeros(Vh.shape[1]))

    def forward(self, h):
        return h @ self.U @ self.V + self.c


def load_bridge(path, dA, dB):
    """Rebuild a saved bridge of any kind (full linear, linear + MLP, low-rank) from its state dict."""
    sd = torch.load(path)
    if "U" in sd:
        br = LowRankBridge(torch.zeros(dA, dB), sd["U"].shape[1])
    else:
        hidden = sd["mlp.1.weight"].shape[0] if "mlp.1.weight" in sd else 0
        br = Bridge(torch.zeros(dA, dB), hidden)
    br.load_state_dict(sd)
    return br.to(DEV).eval()


# ---------------------------------------------------------------- data

def ids_of(text):
    return np.array(tok(text)["input_ids"])


def random_windows(ids, n, length, seed):
    # the last legal start (len(ids) - length) is never drawn; kept so that saved probe windows reproduce
    ix = np.random.default_rng(seed).integers(0, len(ids) - length, n)
    return torch.tensor(np.stack([ids[i:i + length] for i in ix]), device=DEV)


def disjoint_windows(ids, n, length, seed):
    """n non-overlapping windows, sorted by position. Windows of one article are not independent; analyses cluster by article."""
    k = len(ids) // length
    pick = np.sort(np.random.default_rng(seed).choice(k, n, replace=False))
    return torch.tensor(np.stack([ids[i * length:(i + 1) * length] for i in pick]), device=DEV)


# ---------------------------------------------------------------- forward passes for each setting
# full: seq length T+1; inputs x = s[:, :-1], targets s[:, 1:]
# comm: seq length P+S+1; A reads s[:, :P]; B reads only the suffix x_s = s[:, P:P+S]; targets s[:, P+1:P+S+1].
#   Pass 1: B runs on x_s alone (position ids P..P+S-1) and its layer-LB suffix states are kept.
#   Pass 2: B's layers >= LB run on [message for the P prefix positions ; B's own suffix states]. B's layers below LB
#   never see the prefix, so the only route for prefix information is the message.

@torch.no_grad()
def suffix_states(pair, s, P):
    xs = s[:, P:-1]
    pos = torch.arange(P, P + xs.shape[1], device=s.device).expand(len(s), -1)
    return pair.B(xs, position_ids=pos, output_hidden_states=True).hidden_states[pair.lb]


def comm_logits(pair, msg, s, P):
    """B's suffix logits given a message (B-space states for the P prefix positions)."""
    h = torch.cat([msg, suffix_states(pair, s, P)], 1)
    dummy = s[:, :-1].clone(); dummy[:, :P] = EOS          # overwritten at layer LB; only sets the sequence length
    return pair.b_logits(dummy, h)[:, P:]


def bridged_logits(pair, br, s, mode, P=None, msg_perm=None):
    if mode == "full":
        x = s[:, :-1]
        return pair.b_logits(x, br(pair.hA(x)))
    msg = br(pair.hA(s[:, :P]))
    if msg_perm is not None:
        msg = msg[msg_perm]
    return comm_logits(pair, msg, s, P)


def targets(s, mode, P=None):
    return s[:, 1:] if mode == "full" else s[:, P + 1:]


def b_reference(pair, s, mode, P=None):
    """B's own distribution with the information the bridge is meant to supply: all tokens visible."""
    lg = pair.b_logits(s[:, :-1])
    return lg if mode == "full" else lg[:, P:]


# ---------------------------------------------------------------- fit and score

def fit(pair, probe, mode, M0, P=None, hidden=0, steps=600, bs=16, lr=1e-3, seed=0, log=print, checkpoints=None,
        rank=None):
    """checkpoints: {step: callback(bridge)} evaluated after that many steps. rank: low-rank bridge instead."""
    torch.manual_seed(seed)
    br = (LowRankBridge(M0, rank) if rank else Bridge(M0, hidden)).to(DEV)
    opt = torch.optim.Adam(br.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for st in range(steps):
        s = probe[torch.randint(0, len(probe), (bs,), generator=g)]
        with torch.no_grad():
            tgt = torch.log_softmax(b_reference(pair, s, mode, P), -1)
        logp = torch.log_softmax(bridged_logits(pair, br, s, mode, P), -1)
        loss = F.kl_div(logp, tgt, log_target=True, reduction="batchmean") / logp.shape[1]
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 200 == 0 or st == steps - 1:
            log(f"    step {st} KL/token {loss.item():.4f}")
        if checkpoints and st + 1 in checkpoints:
            br.eval(); checkpoints[st + 1](br); br.train()
    return br.eval()


@torch.no_grad()
def per_window(lg, y):
    """Per-window mean accuracy and mean cross-entropy (nats/token)."""
    acc = (lg.argmax(-1) == y).float().mean(1)
    ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").view(y.shape).mean(1)
    return acc.cpu().numpy(), ce.cpu().numpy()


@torch.no_grad()
def score(fn, test, bs=16):
    """fn(s) -> logits aligned with the targets produced by tgt(s); returns per-window arrays."""
    accs, ces = [], []
    for i in range(0, len(test), bs):
        s = test[i:i + bs]
        lg, y = fn(s)
        a, c = per_window(lg, y)
        accs.append(a); ces.append(c)
    return np.concatenate(accs), np.concatenate(ces)


def paired_ci(a, b=None, n=4000, seed=0):
    """Mean of a (or of a-b) with a 95% bootstrap interval over test windows."""
    d = a if b is None else a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def window_controls(lines, ids, n, length, seed, P):
    """Message permutations for the mismatched-message controls on disjoint_windows(ids, n, length, seed).

    far[i]:  window i + n/2 (mod n), always from a different WikiText article (checked);
    same[i]: the nearest other window from the same article, or i itself when the article has no other window
             (has_same[i] = 0; such windows are excluded from same-article contrasts).
    Articles are delimited by top-level ' = Title = ' headings; windows are assigned by their last prefix token."""
    import re
    text = "".join(lines)
    art_of_line = np.cumsum([1 if re.match(r"^ = [^=].* = \n$", l) else 0 for l in lines])
    offs = np.array([o[0] for o in tok(text, return_offsets_mapping=True)["offset_mapping"]])
    assert len(offs) == len(ids)
    line_start = np.cumsum([0] + [len(l) for l in lines[:-1]])
    pick = np.sort(np.random.default_rng(seed).choice(len(ids) // length, n, replace=False))
    art = art_of_line[np.searchsorted(line_start, offs[pick * length + P - 1], side="right") - 1]
    far = (np.arange(n) + n // 2) % n
    assert not np.any(art == art[far]), "far control must pair different articles"
    same = np.arange(n)
    for i in range(n):
        cand = [j for j in range(n) if j != i and art[j] == art[i]]
        if cand:
            same[i] = min(cand, key=lambda j: abs(j - i))
    return {"far": torch.tensor(far, device=DEV), "same": torch.tensor(same, device=DEV),
            "has_same": (same != np.arange(n)).astype(np.float32), "article": art}
