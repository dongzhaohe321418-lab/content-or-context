"""Cross-family latent communication: sender A and receiver B may have different architectures and tokenizers.

Windows are defined in B's tokenisation: P_B prefix tokens, S suffix tokens, one target token beyond. A reads the
prefix *text* (B's prefix tokens decoded), re-tokenised with A's tokenizer and truncated to its last P_A tokens, where
P_A is fixed per pair so that batches need no padding. B reads only its S suffix tokens (pass 1, position ids
P_A..P_A+S-1); its layers > L_B then run on [bridge(A's layer-L_A prefix states) ; B's layer-L_B suffix states].
Bridges are fitted with the communication objective only (KL to B's full-context suffix distribution), which needs
no token alignment between A and B. The weight-only map uses tokens whose strings exist in both vocabularies.

    python pbx.py <name> <A hf id> <B hf id> [seeds=0,1]
"""
import json, os, re, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pb import Bridge, DEV, OUT, REV

P_B, S, NTEST, NPROBE, R = 128, 32, 512, 4096, 5
SMOKE = os.environ.get("PBX_SMOKE") == "1"                              # tiny run to check the plumbing
if SMOKE:
    NTEST, NPROBE, R = 48, 64, 1
PR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_pythia", "probes")


def layers_of(m):
    for path in ("gpt_neox.layers", "transformer.h", "model.decoder.layers", "model.layers"):
        obj = m
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise ValueError(type(m))


class Model:
    def __init__(self, hf_id):
        self.tok = AutoTokenizer.from_pretrained(hf_id, revision=REV[hf_id])
        self.m = AutoModelForCausalLM.from_pretrained(hf_id, revision=REV[hf_id], dtype=torch.float32).to(DEV).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.d = self.m.config.hidden_size
        self.inject = None
        self.handle = None
        self.pad = self.tok.eos_token_id if self.tok.eos_token_id is not None else 0

    def hook_at(self, lb):
        if self.handle is not None:
            self.handle.remove()

        def hook(mod, inp, out):
            if self.inject is None:
                return out
            return (self.inject,) + tuple(out[1:]) if isinstance(out, tuple) else self.inject
        self.handle = layers_of(self.m)[lb - 1].register_forward_hook(hook)

    def hidden(self, ids, layer, pos=None):
        kw = {"position_ids": pos} if pos is not None else {}
        return self.m(ids, output_hidden_states=True, **kw).hidden_states[layer]

    def logits(self, ids, h=None):
        self.inject = h
        try:
            return self.m(ids).logits
        finally:
            self.inject = None


def anchor_map(A, B):
    """Least squares between the embedding (and, if untied, unembedding) rows of tokens with identical strings."""
    va, vb = A.tok.get_vocab(), B.tok.get_vocab()
    shared = sorted(set(va) & set(vb))
    ia = torch.tensor([va[t] for t in shared]); ib = torch.tensor([vb[t] for t in shared])
    EA = A.m.get_input_embeddings().weight.cpu()[ia]; EB = B.m.get_input_embeddings().weight.cpu()[ib]
    UA = A.m.get_output_embeddings().weight.cpu()[ia]; UB = B.m.get_output_embeddings().weight.cpu()[ib]
    M = torch.linalg.lstsq(torch.cat([EA, UA]), torch.cat([EB, UB])).solution
    return M, len(shared)


class Pair:
    def __init__(self, A, B, la, lb, PA):
        self.A, self.B, self.la, self.lb, self.PA = A, B, la, lb, PA
        B.hook_at(lb)

    def msg_A(self, a_ids, br):
        return br(self.A.hidden(a_ids, self.la))

    @torch.no_grad()
    def suffix_states(self, b, offset):
        """B's layer-L_B states for the suffix alone, at the positions it occupies after an offset-long message."""
        xs = b[:, P_B:-1]
        pos = torch.arange(offset, offset + S, device=DEV).expand(len(b), -1)
        return self.B.hidden(xs, self.lb, pos)

    def comm_logits(self, msg, b):
        h = torch.cat([msg, self.suffix_states(b, msg.shape[1])], 1)
        dummy = torch.full((len(b), h.shape[1]), self.B.pad, device=DEV)
        return self.B.logits(dummy, h)[:, msg.shape[1]:]

    def full_context(self, b):
        return self.B.logits(b[:, :-1])[:, P_B:]


def make_windows(A, B, text_or_ids, n, seed, PA=None, disjoint=False):
    """B-token windows of P_B+S+1 and A's last PA tokens of the same prefix text. Returns a_ids, b_ids, starts."""
    ids = np.array(B.tok(text_or_ids)["input_ids"]) if isinstance(text_or_ids, str) else text_or_ids
    L = P_B + S + 1
    rng = np.random.default_rng(seed)
    if disjoint:
        starts = np.sort(rng.choice(len(ids) // L, n, replace=False)) * L
    else:
        starts = rng.integers(0, len(ids) - L, n)   # excludes the last legal start; kept for reproducibility
    b = np.stack([ids[s:s + L] for s in starts])
    pre_text = B.tok.batch_decode(b[:, :P_B])
    a_tok = [A.tok(t, add_special_tokens=False)["input_ids"] for t in pre_text]
    if PA is None:
        PA = min(len(t) for t in a_tok)
    keep = np.array([len(t) >= PA for t in a_tok])
    a = np.stack([t[-PA:] for t, k in zip(a_tok, keep) if k])
    return torch.tensor(a, device=DEV), torch.tensor(b[keep], device=DEV), starts[keep], PA


def fit(pair, a, b, M0, steps=600, bs=16, lr=1e-3, seed=0, log=print):
    torch.manual_seed(seed)
    br = Bridge(M0).to(DEV)
    opt = torch.optim.Adam(br.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for st in range(steps):
        ix = torch.randint(0, len(b), (bs,), generator=g)
        with torch.no_grad():
            tgt = torch.log_softmax(pair.full_context(b[ix]), -1)
        logp = torch.log_softmax(pair.comm_logits(pair.msg_A(a[ix], br), b[ix]), -1)
        loss = F.kl_div(logp, tgt, log_target=True, reduction="batchmean") / logp.shape[1]
        opt.zero_grad(); loss.backward(); opt.step()
        if st % 200 == 0 or st == steps - 1:
            log(f"    step {st} KL/token {loss.item():.4f}")
    return br.eval()


@torch.no_grad()
def ce(lg, y):
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").view(y.shape).mean(1)


def self_text(model, n, length, seed):
    torch.manual_seed(seed)
    start = torch.full((n, 1), model.tok.bos_token_id if model.tok.bos_token_id is not None else model.pad, device=DEV)
    with torch.no_grad():
        o = model.m.generate(start, attention_mask=torch.ones_like(start), do_sample=True, temperature=1.0, top_k=0,
                             max_new_tokens=length, min_new_tokens=length, pad_token_id=model.pad)
    return "\n".join(model.tok.batch_decode(o[:, 1:], skip_special_tokens=True))


def main():
    name, a_id, b_id = sys.argv[1:4]
    skip = f"{OUT}/x_skip.txt"                                            # pairs dropped to fit the compute budget
    if os.path.exists(skip) and name in open(skip).read().split():
        print(f"{name}: skipped (listed in x_skip.txt)"); return
    lean = os.path.exists(f"{OUT}/x_lean.txt") and name in open(f"{OUT}/x_lean.txt").read().split()
    seeds = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0,1").split(",")]
    log_f = open(f"{OUT}/x_{name}.log", "a")
    def log(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
    A, B = Model(a_id), Model(b_id)
    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", revision=REV["Salesforce/wikitext"])
    lines = wt["test"]["text"][:6000]; text = "".join(lines)
    enc = B.tok(text, return_offsets_mapping=True)
    ids = np.array(enc["input_ids"]); offs = np.array([o[0] for o in enc["offset_mapping"]])
    a_te, b_te, starts, PA = make_windows(A, B, ids, NTEST, 123, disjoint=True)
    n = len(b_te)
    art_of_line = np.cumsum([1 if re.match(r"^ = [^=].* = \n$", l) else 0 for l in lines])
    line_start = np.cumsum([0] + [len(l) for l in lines[:-1]])
    art_at = lambda t: art_of_line[np.searchsorted(line_start, offs[t], side="right") - 1]
    art = art_at(starts + P_B - 1)
    within = (art_at(starts) == art_at(starts + P_B + S)).astype(np.float32)
    rng = np.random.default_rng(2026)
    inside = within > 0                                                   # donors must lie inside one article
    other = [torch.tensor([rng.choice(np.flatnonzero(inside & (art != art[i]))) for i in range(n)], device=DEV)
             for _ in range(R)]
    same = []
    for _ in range(R):
        sm = []
        for i in range(n):
            c = np.flatnonzero(inside & (art == art[i]) & (np.arange(n) != i))
            sm.append(rng.choice(c) if len(c) else i)
        same.append(torch.tensor(sm, device=DEV))
    has_same = (same[0].cpu().numpy() != np.arange(n)).astype(np.float32)
    val_text = "".join(wt["validation"]["text"][:6000])
    a_va, b_va, _, _ = make_windows(A, B, val_text, 32 if SMOKE else 256, 7, PA=PA, disjoint=True)
    train_text = "".join(wt["train"]["text"][:30000])
    claude = open(f"{PR}/all3x.txt").read()
    log(f"== {name}: A={a_id} B={b_id}  P_A={PA}  test windows={n}  articles={len(set(art))}")
    out = {"article": art, "has_same": has_same}
    for k in range(R):
        out[f"donor_other r{k}"] = other[k].cpu().numpy(); out[f"donor_same r{k}"] = same[k].cpu().numpy()

    def evaluate(tag, msg_fn, a_set, b_set, swaps=True):
        with torch.no_grad():
            msgs = torch.cat([msg_fn(a_set[i:i + 16], b_set[i:i + 16]) for i in range(0, len(b_set), 16)])
            def run(idx):
                r = []
                for i in range(0, len(b_set), 16):
                    j = torch.arange(i, min(i + 16, len(b_set)), device=DEV) if idx is None else idx[i:i + 16]
                    bb = b_set[i:i + 16]
                    r.append(ce(pair.comm_logits(msgs[j], bb), bb[:, P_B + 1:]))
                return torch.cat(r).cpu().numpy()
            res = {"matched": run(None)}
            if swaps:
                for k in range(R):
                    res[f"other r{k}"] = run(other[k]); res[f"same r{k}"] = run(same[k])
        if swaps:
            for k, v in res.items():
                out[f"{tag}|{k}"] = v
            log(f"  {tag:40s} matched {res['matched'].mean():.4f}  other "
                f"{np.mean([res[f'other r{k}'].mean() for k in range(R)]):.4f}  same "
                f"{np.mean([res[f'same r{k}'].mean() for k in range(R)]):.4f}")
        return res["matched"].mean()

    M0, n_shared = anchor_map(A, B)
    log(f"  shared token strings: {n_shared}")
    # layer selection on validation windows with a short real-text fit (as for Pythia); L_B = 1
    nA = A.m.config.num_hidden_layers
    cands = sorted({1, 2, 4}) if lean else sorted({1, 2, 3, max(1, nA // 6), max(1, nA // 3)})
    real_a, real_b, _, _ = make_windows(A, B, train_text, NPROBE, 100, PA=PA)
    best = None
    for la in cands:
        pair = Pair(A, B, la, 1, PA)
        br = fit(pair, real_a, real_b, M0, steps=5 if SMOKE else 300, log=log)
        v = evaluate("val", lambda a, b, br=br: pair.msg_A(a, br), a_va, b_va, swaps=False)
        log(f"  validation: L_A={la} -> L_B=1  CE {v:.4f}")
        if best is None or v < best[0]:
            best = (v, la)
    la = best[1]
    log(f"  chose L_A={la}, L_B=1")
    pair = Pair(A, B, la, 1, PA)
    with torch.no_grad():
        out["no context"] = torch.cat([ce(B.logits(b_te[i:i + 16, P_B:-1]), b_te[i:i + 16, P_B + 1:])
                                       for i in range(0, n, 16)]).cpu().numpy()
        out["text|matched"] = torch.cat([ce(pair.full_context(b_te[i:i + 16]), b_te[i:i + 16, P_B + 1:])
                                         for i in range(0, n, 16)]).cpu().numpy()
        for k in range(R):
            for kind, idx in (("other", other[k]), ("same", same[k])):
                out[f"text|{kind} r{k}"] = torch.cat([ce(B.logits(torch.cat([b_te[idx[i:i + 16], :P_B],
                                                                            b_te[i:i + 16, P_B:-1]], 1))[:, P_B:],
                                                         b_te[i:i + 16, P_B + 1:]) for i in range(0, n, 16)]).cpu().numpy()
    log(f"  no context {out['no context'].mean():.4f}   text {out['text|matched'].mean():.4f}")
    evaluate("receiver-state reference", lambda a, b: B.hidden(b[:, :P_B], 1), a_te, b_te)
    torch.manual_seed(0)
    evaluate("random map", lambda a, b, br=Bridge(torch.randn(A.d, B.d) / A.d ** .5).to(DEV): pair.msg_A(a, br), a_te, b_te)
    evaluate("weight-only anchors", lambda a, b, br=Bridge(M0).to(DEV): pair.msg_A(a, br), a_te, b_te)
    for sd in seeds:
        selfs = "" if lean else self_text(A, 128, 200, 10 * sd) + "\n" + self_text(B, 128, 200, 10 * sd + 1)
        sources = (("real", train_text), ("Claude", claude)) if lean else \
            (("real", train_text), ("Claude", claude), ("self-sampled", selfs))
        for pname, txt in sources:
            t0 = time.time()
            pa, pbb, _, _ = make_windows(A, B, txt, NPROBE, 100 + sd, PA=PA)
            br = fit(pair, pa, pbb, M0, steps=5 if SMOKE else 600, seed=sd, log=log)
            evaluate(f"comm_{pname}_s{sd}", lambda a, b, br=br: pair.msg_A(a, br), a_te, b_te)
            torch.save(br.state_dict(), f"{OUT}/x_{name}_comm_{pname}_s{sd}.pt")
            log(f"    ({time.time() - t0:.0f} s)")
            np.savez(f"{OUT}/x_{name}_swap.npz", within_article=within, **out)
    np.savez(f"{OUT}/x_{name}_swap.npz", within_article=within, **out)
    json.dump({"A": a_id, "B": b_id, "L_A": la, "L_B": 1, "P_A": PA, "n_test": n, "shared_tokens": n_shared},
              open(f"{OUT}/x_{name}_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
