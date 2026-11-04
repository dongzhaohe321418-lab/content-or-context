"""Latent bridge between two real pretrained LMs: pythia-70m (A, width 512, 6 layers) -> pythia-160m (B, width 768,
12 layers). Separate pretraining runs, shared tokenizer. A sends its residual stream after layer LA; B's residual
stream after layer LB is replaced (forward hook) by  h_A @ M + c  and B finishes the forward pass.
Every fitted bridge minimises KL(B on the text || B through the bridge) on some probe text; what differs is the text:
real text (upper bound) or text sampled from A and B themselves (data-free).

    python pythia.py
"""
import json, os, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_pythia")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
LA, LB, T = 3, 6, 128
torch.manual_seed(0)

# Hugging Face revisions used in the study (MANIFEST.md)
REV = {"EleutherAI/pythia-70m": "a39f36b100fe8a5377810d56c3f4789b9c53ac42",
       "EleutherAI/pythia-160m": "50f5173d932e8e61f858120bcb800b97af589f46",
       "Salesforce/wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3"}
tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", revision=REV["EleutherAI/pythia-70m"])
A = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m", revision=REV["EleutherAI/pythia-70m"], torch_dtype=torch.float32).to(DEV).eval()
B = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-160m", revision=REV["EleutherAI/pythia-160m"], torch_dtype=torch.float32).to(DEV).eval()
for p in list(A.parameters()) + list(B.parameters()):
    p.requires_grad_(False)
dA, dB = A.config.hidden_size, B.config.hidden_size
INJECT = {"h": None}


def hook(mod, inp, out):
    if INJECT["h"] is None:
        return out
    return (INJECT["h"],) + tuple(out[1:]) if isinstance(out, tuple) else INJECT["h"]


B.gpt_neox.layers[LB - 1].register_forward_hook(hook)


def hA(ids):
    return A(ids, output_hidden_states=True).hidden_states[LA]


def b_logits(ids, h=None):
    INJECT["h"] = h
    try:
        return B(ids).logits
    finally:
        INJECT["h"] = None


def windows(text_ids, n, rng):
    ix = rng.integers(0, len(text_ids) - T - 1, n)
    return torch.tensor(np.stack([text_ids[i:i + T + 1] for i in ix]), device=DEV)


def score(seqs, M=None, c=None, which="bridge"):
    accs, losses = [], []
    with torch.no_grad():
        for i in range(0, len(seqs), 16):
            s = seqs[i:i + 16]; x, y = s[:, :-1], s[:, 1:]
            if which == "A":
                lg = A(x).logits
            elif which == "B":
                lg = b_logits(x)
            else:
                lg = b_logits(x, hA(x) @ M + c)
            accs.append((lg.argmax(-1) == y).float().mean().item())
            losses.append(torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1)).item())
    return {"acc": round(float(np.mean(accs)), 4), "loss": round(float(np.mean(losses)), 4)}


def fit(probe_seqs, M0, c0, steps=600, bs=16, lr=1e-3, seed=0):
    M = torch.tensor(M0, device=DEV, dtype=torch.float32, requires_grad=True)
    c = torch.tensor(c0, device=DEV, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([M, c], lr=lr)
    g = torch.Generator().manual_seed(seed)
    for s in range(steps):
        x = probe_seqs[torch.randint(0, len(probe_seqs), (bs,), generator=g)][:, :-1]
        with torch.no_grad():
            target = torch.log_softmax(b_logits(x), -1); h = hA(x)
        logp = torch.log_softmax(b_logits(x, h @ M + c), -1)
        loss = torch.nn.functional.kl_div(logp, target, log_target=True, reduction="batchmean") / x.shape[1]
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 200 == 0:
            print(f"    step {s} KL/token {loss.item():.3f}", flush=True)
    return M.detach(), c.detach()


def self_sample(model, n, seed):
    torch.manual_seed(seed)
    start = torch.full((n, 1), tok.eos_token_id, device=DEV)
    with torch.no_grad():
        out = model.generate(start, do_sample=True, temperature=1.0, top_k=0, max_new_tokens=T, min_new_tokens=T,
                             pad_token_id=tok.eos_token_id)
    return out[:, 1:T + 2] if out.shape[1] >= T + 2 else out[:, :T + 1]


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", revision=REV["Salesforce/wikitext"])
    te_ids = np.array(tok("".join(wt["test"]["text"][:6000]))["input_ids"])
    tr_ids = np.array(tok("".join(wt["train"]["text"][:30000]))["input_ids"])
    test = windows(te_ids, 256, rng)
    res = {"A alone (pythia-70m)": score(test, which="A"), "B alone (pythia-160m)": score(test, which="B")}
    res["random bridge"] = score(test, torch.randn(dA, dB, device=DEV) / dA ** 0.5, torch.zeros(dB, device=DEV))
    EA, EB = A.gpt_neox.embed_in.weight, B.gpt_neox.embed_in.weight
    M_emb = torch.linalg.lstsq(EA.cpu(), EB.cpu()).solution.to(DEV)
    res["token-embedding anchors, least squares (data-free)"] = score(test, M_emb, torch.zeros(dB, device=DEV))
    UA, UB = A.get_output_embeddings().weight, B.get_output_embeddings().weight
    M_eu = torch.linalg.lstsq(torch.cat([EA, UA]).cpu(), torch.cat([EB, UB]).cpu()).solution.to(DEV)
    res["embedding + unembedding anchors, least squares (data-free)"] = score(test, M_eu, torch.zeros(dB, device=DEV))
    print(json.dumps(res, indent=1), flush=True)
    t0 = time.time()
    real = windows(tr_ids, 4096, rng)
    print("fit on real text"); M, c = fit(real, M_eu.cpu().numpy(), np.zeros(dB))
    res["KL fit on real wikitext (upper bound)"] = score(test, M, c)
    print(res["KL fit on real wikitext (upper bound)"], round(time.time() - t0), "s", flush=True)
    print("sampling probe text from A and B")
    samp = torch.cat([self_sample(A, 128, 1), self_sample(A, 128, 2), self_sample(B, 128, 3), self_sample(B, 128, 4)])
    samp = samp[:, :T + 1]
    print("  example:", repr(tok.decode(samp[0][:40])), flush=True)
    M2, c2 = fit(samp, M_eu.cpu().numpy(), np.zeros(dB))
    res["KL fit on text sampled from A and B (data-free, no corpus)"] = score(test, M2, c2)
    for k, v in res.items():
        print(f"{k:62s} {v}")
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
    torch.save({"M_eu": M_eu.cpu(), "M_real": M.cpu(), "c_real": c.cpu(), "M_self": M2.cpu(), "c_self": c2.cpu()},
               f"{OUT}/bridges.pt")


if __name__ == "__main__":
    main()
