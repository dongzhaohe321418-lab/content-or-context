"""Experiment groups for the paper. Each group writes out_pb/<group>.json (summaries) and out_pb/<group>.npz
(per-test-window accuracy and cross-entropy for every condition, for paired bootstrap comparisons).

    python pb_run.py <group> [<group> ...]

Groups
  sanity      160m -> 160m identity bridge must reproduce B's own full-context numbers exactly (code check).
  comm70      information-asymmetric communication, pythia-70m reads the prefix, pythia-160m predicts the suffix.
  diag        hand-over (full) setting: optimisation length, MLP capacity, layer pairing (descriptive diagnostic).
  comm160_410 / comm410_160   the comm setting at larger scale, weak->strong and strong->weak.
  textscale   Claude-written vs equal-size real text as probe text, up to ~1M characters (full setting).
"""
import json, os, re, sys, time
import numpy as np
import torch
from datasets import load_dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pb

os.makedirs(pb.OUT, exist_ok=True)
PR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_pythia", "probes")
P, S, T = 128, 32, 128
NTEST, NPROBE = 512, 4096

TEST_LINES = TEST_IDS = VAL_IDS = TRAIN_TEXT = TRAIN_IDS = None


def load_corpus():
    """WikiText is loaded only for groups that use it, so sweep70free runs without the corpus."""
    global TEST_LINES, TEST_IDS, VAL_IDS, TRAIN_TEXT, TRAIN_IDS
    if TEST_IDS is not None:
        return
    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", revision=pb.REV["Salesforce/wikitext"])
    TEST_LINES = wt["test"]["text"][:6000]
    TEST_IDS = pb.ids_of("".join(TEST_LINES))
    VAL_IDS = pb.ids_of("".join(wt["validation"]["text"][:6000]))      # layer selection only; never used for reporting
    TRAIN_TEXT = "".join(wt["train"]["text"][:30000])
    TRAIN_IDS = pb.ids_of(TRAIN_TEXT)
CLAUDE3X = open(f"{PR}/all3x.txt").read()


class Group:
    def __init__(self, name):
        self.name, self.res, self.arr = name, {}, {}
        self.log = open(f"{pb.OUT}/{name}.log", "a")

    def p(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True); self.log.write(s + "\n"); self.log.flush()

    def add(self, key, acc, ce, extra=None):
        self.arr[f"{key}|acc"], self.arr[f"{key}|ce"] = acc, ce
        self.res[key] = {"acc": round(100 * float(acc.mean()), 2), "ce": round(float(ce.mean()), 4), **(extra or {})}
        self.p(f"  {key:60s} acc {self.res[key]['acc']:6.2f}  ce {self.res[key]['ce']:.4f}")
        json.dump(self.res, open(f"{pb.OUT}/{self.name}.json", "w"), indent=1)
        np.savez(f"{pb.OUT}/{self.name}.npz", **self.arr)


def self_sample(pair, n_each, length, seed):
    """Unconditional samples from A and B themselves (data-free, no LLM)."""
    out = []
    for j, m in enumerate([pair.A, pair.A, pair.B, pair.B]):
        torch.manual_seed(10 * seed + j)
        start = torch.full((n_each, 1), pb.EOS, device=pb.DEV)
        pair.B._inject.update(h=None, n=None)
        with torch.no_grad():
            o = m.generate(start, do_sample=True, temperature=1.0, top_k=0, max_new_tokens=length,
                           min_new_tokens=length, pad_token_id=pb.EOS)
        out.append(o[:, 1:length + 1])
    return torch.cat(out)


def probe_sets(pair, length, seed):
    return {"real wikitext": pb.random_windows(TRAIN_IDS, NPROBE, length, 100 + seed),
            "Claude text 383k": pb.random_windows(pb.ids_of(CLAUDE3X), NPROBE, length, 100 + seed),
            "self-sampled": self_sample(pair, 128, length, seed)}


# ------------------------------------------------------------------ comm setting

def comm_refs(g, pair, test, ctrl):
    A, B = pair.A, pair.B
    g.add("B full context (upper bound)", *pb.score(lambda s: (pair.b_logits(s[:, :-1])[:, P:], s[:, P + 1:]), test))
    g.add("B no context (suffix only)", *pb.score(lambda s: (pair.b_logits(s[:, P:-1]), s[:, P + 1:]), test))
    g.add("A full context", *pb.score(lambda s: (A(s[:, :-1]).logits[:, P:], s[:, P + 1:]), test))
    g.add("A no context (suffix only)", *pb.score(lambda s: (A(s[:, P:-1]).logits, s[:, P + 1:]), test))
    # receiver-state reference: B's own layer-LB states for the prefix (B reading the prefix alone), injected the same way
    with torch.no_grad():
        msgs = torch.cat([B(test[i:i + 16, :P], output_hidden_states=True).hidden_states[pair.lb]
                          for i in range(0, len(test), 16)])
    for suffix, idx in [("", None), (" | mismatched message", ctrl["far"]), (" | same-article message", ctrl["same"])]:
        accs, ces = [], []
        for i in range(0, len(test), 16):
            s = test[i:i + 16]
            m = msgs[i:i + 16] if idx is None else msgs[idx[i:i + 16]]
            a, c = pb.per_window(pb.comm_logits(pair, m, s, P), s[:, P + 1:])
            accs.append(a); ces.append(c)
        g.add("receiver-state reference: B's own prefix states" + suffix, np.concatenate(accs), np.concatenate(ces))
    # text channel with another article's / another same-article window's prefix
    for suffix, idx in [(" | mismatched message", ctrl["far"]), (" | same-article message", ctrl["same"])]:
        accs, ces = [], []
        for i in range(0, len(test), 16):
            s = test[i:i + 16]
            x = torch.cat([test[idx[i:i + 16], :P], s[:, P:-1]], 1)
            a, c = pb.per_window(pair.b_logits(x)[:, P:], s[:, P + 1:])
            accs.append(a); ces.append(c)
        g.add("B full context (upper bound)" + suffix, np.concatenate(accs), np.concatenate(ces))


def comm_bridge(g, pair, test, key, br, ctrl=None):
    """Matched message plus mismatched controls. With ctrl (pb.window_controls): ' | mismatched message' takes the
    message of a window from a different article and ' | same-article message' that of another window of the same
    article. Without ctrl (validation sweeps only): messages rolled by one within each batch."""
    if ctrl is None:
        g.add(key, *pb.score(lambda s: (pb.bridged_logits(pair, br, s, "comm", P), s[:, P + 1:]), test))
        perm = lambda s: torch.roll(torch.arange(len(s), device=pb.DEV), 1)
        g.add(key + " | batch-rolled message", *pb.score(
            lambda s: (pb.bridged_logits(pair, br, s, "comm", P, msg_perm=perm(s)), s[:, P + 1:]), test))
        return
    with torch.no_grad():
        msgs = torch.cat([br(pair.hA(test[i:i + 16, :P])) for i in range(0, len(test), 16)])
    for suffix, idx in [("", None), (" | mismatched message", ctrl["far"]), (" | same-article message", ctrl["same"])]:
        accs, ces = [], []
        for i in range(0, len(test), 16):
            s = test[i:i + 16]
            m = msgs[i:i + 16] if idx is None else msgs[idx[i:i + 16]]
            a, c = pb.per_window(pb.comm_logits(pair, m, s, P), s[:, P + 1:])
            accs.append(a); ces.append(c)
        g.add(key + suffix, np.concatenate(accs), np.concatenate(ces))


def run_commsweep(gname, a, b, las, lbs, steps=300, data_free=False):
    """Choose the (LA, LB) pair on wikitext *validation* windows with a real-text comm-objective fit.
    data_free: fit on the 24 LLM-written articles used in the main experiments and validate on the other 40
    LLM-written articles, so that no corpus text enters selection either."""
    g = Group(gname)
    if data_free:
        files = sorted((f for f in os.listdir(PR) if f.startswith("w") and f.endswith(".txt")), key=lambda f: int(f[1:-4]))
        heldout = "\n".join(open(f"{PR}/{f}").read() for f in files if int(f[1:-4]) >= 26)
        val = pb.disjoint_windows(pb.ids_of(heldout), 256, P + S + 1, 7)
        real = pb.random_windows(pb.ids_of(CLAUDE3X), NPROBE, P + S + 1, 100)
    else:
        val = pb.disjoint_windows(VAL_IDS, 256, P + S + 1, 7)
        real = pb.random_windows(TRAIN_IDS, NPROBE, P + S + 1, 100)
    for lb in lbs:
        for la in las:
            pair = pb.Pair(a, b, la, lb)
            if la == las[0]:
                def oracle(s):
                    h = pair.B(s[:, :P], output_hidden_states=True).hidden_states[lb]
                    return pb.comm_logits(pair, h, s, P), s[:, P + 1:]
                g.add(f"oracle at B L{lb}", *pb.score(oracle, val))
            br = pb.fit(pair, real, "comm", pair.anchors(), P=P, steps=steps, log=g.p)
            comm_bridge(g, pair, val, f"A L{la} -> B L{lb}, fit[comm] {'LLM text' if data_free else 'real'} {steps} steps", br)
    return g


def run_comm(gname, a, b, la, lb, seeds=(0, 1, 2), objectives=("full", "comm"), steps=600, comm_self=False,
             only=None, save_as=None):
    g = Group(gname)
    pair = pb.Pair(a, b, la, lb)
    g.p(f"== {gname}: {a} L{la} -> {b} L{lb}, prefix {P}, suffix {S}")
    test = pb.disjoint_windows(TEST_IDS, NTEST, P + S + 1, 123)
    ctrl = pb.window_controls(TEST_LINES, TEST_IDS, NTEST, P + S + 1, 123, P)
    g.arr["has_same_article_partner"] = ctrl["has_same"]
    comm_refs(g, pair, test, ctrl)
    M0 = pair.anchors()
    torch.manual_seed(0)
    comm_bridge(g, pair, test, "random bridge", pb.Bridge(torch.randn(pair.dA, pair.dB) / pair.dA ** .5).to(pb.DEV), ctrl)
    comm_bridge(g, pair, test, "embedding anchors, no fit (data-free)", pb.Bridge(M0).to(pb.DEV), ctrl)
    for sd in seeds:
        sets = probe_sets(pair, P + S + 1, sd)
        for obj in objectives:
            for name, probe in sets.items():
                if obj == "comm" and name == "self-sampled" and not comm_self:
                    continue
                if only is not None and name not in only:
                    continue
                t0 = time.time()
                pr = probe[:, :T + 1] if obj == "full" else probe
                br = pb.fit(pair, pr, obj, M0, P=P, steps=steps, seed=sd, log=g.p)
                comm_bridge(g, pair, test, f"fit[{obj}] on {name} | seed {sd}", br, ctrl)
                g.p(f"    ({time.time() - t0:.0f} s)")
                torch.save(br.state_dict(), f"{pb.OUT}/{save_as or gname}_{obj}_{name.split()[0]}_s{sd}.pt")
    return g


# ------------------------------------------------------------------ full setting diagnostics

def full_refs(g, pair, test):
    g.add("A alone", *pb.score(lambda s: (pair.A(s[:, :-1]).logits, s[:, 1:]), test))
    g.add("B alone", *pb.score(lambda s: (pair.b_logits(s[:, :-1]), s[:, 1:]), test))


def full_bridge(g, pair, test, key, br):
    g.add(key, *pb.score(lambda s: (pb.bridged_logits(pair, br, s, "full"), s[:, 1:]), test))


def run_diag():
    g = Group("diag")
    test = pb.disjoint_windows(TEST_IDS, NTEST, T + 1, 321)
    real = pb.random_windows(TRAIN_IDS, NPROBE, T + 1, 100)
    pair = pb.Pair("pythia-70m", "pythia-160m", 3, 6)
    full_refs(g, pair, test)
    M0 = pair.anchors()
    for hidden in (0, 2048):
        t0 = time.time()
        kind = "MLP-2048" if hidden else "linear"
        ck = {n: (lambda br, n=n: full_bridge(g, pair, test, f"L3->L6 {kind} {n} steps", br)) for n in (600, 1500, 3000)}
        pb.fit(pair, real, "full", M0, hidden=hidden, steps=3000, log=g.p, checkpoints=ck)
        g.p(f"    ({time.time() - t0:.0f} s)")
    for la, lb in [(1, 2), (2, 4), (4, 8), (5, 10), (6, 12)]:
        pair = pb.Pair("pythia-70m", "pythia-160m", la, lb)
        br = pb.fit(pair, real, "full", M0, steps=600, log=g.p)
        full_bridge(g, pair, test, f"L{la}->L{lb} linear 600 steps", br)


def run_textscale():
    """Comm setting (70m L2 -> 160m L1, comm objective): Claude-written vs equal-size real text, up to ~1M chars."""
    g = Group("textscale")
    pair = pb.Pair("pythia-70m", "pythia-160m", 2, 1)
    test = pb.disjoint_windows(TEST_IDS, NTEST, P + S + 1, 123)          # same windows as comm70
    ctrl = pb.window_controls(TEST_LINES, TEST_IDS, NTEST, P + S + 1, 123, P)
    g.arr["has_same_article_partner"] = ctrl["has_same"]
    comm_refs(g, pair, test, ctrl)
    M0 = pair.anchors()
    files = sorted((f for f in os.listdir(PR) if f.startswith("w") and f.endswith(".txt")), key=lambda f: int(f[1:-4]))
    claude_all = "\n".join(open(f"{PR}/{f}").read() for f in files)
    g.p(f"Claude text total {len(claude_all)} chars from {len(files)} files")
    for sd in (0, 1):
        for n in (383_000, len(claude_all)):
            for name, text in [("Claude", claude_all[:n]), ("real", TRAIN_TEXT[:n])]:
                t0 = time.time()
                probe = pb.random_windows(pb.ids_of(text), NPROBE, P + S + 1, 100 + sd)
                br = pb.fit(pair, probe, "comm", M0, P=P, steps=600, seed=sd, log=g.p)
                comm_bridge(g, pair, test, f"{name} text {n // 1000}k chars | seed {sd}", br, ctrl)
                torch.save(br.state_dict(), f"{pb.OUT}/textscale_{name}_{n // 1000}k_s{sd}.pt")
                g.p(f"    ({time.time() - t0:.0f} s)")


def run_sanity():
    g = Group("sanity")
    pair = pb.Pair("pythia-160m", "pythia-160m", 6, 6)
    test = pb.disjoint_windows(TEST_IDS, 64, P + S + 1, 123)
    ident = pb.Bridge(torch.eye(pair.dA)).to(pb.DEV)
    g.add("B full context", *pb.score(lambda s: (pair.b_logits(s[:, :-1])[:, P:], s[:, P + 1:]), test))
    comm_bridge(g, pair, test, "identity bridge B->B (= oracle message; below full context by design)", ident)
    t = pb.disjoint_windows(TEST_IDS, 64, T + 1, 321)
    g.add("B alone (full setting)", *pb.score(lambda s: (pair.b_logits(s[:, :-1]), s[:, 1:]), t))
    full_bridge(g, pair, t, "identity bridge, full setting (must equal B alone)", ident)


if __name__ == "__main__":
    for grp in sys.argv[1:]:
        if grp != "sweep70free":
            load_corpus()
        if grp == "sanity":
            run_sanity()
        elif grp == "sweep70":
            run_commsweep("sweep70", "pythia-70m", "pythia-160m", [1, 2, 3, 4, 5, 6], [1, 2])
        elif grp == "sweep70free":
            run_commsweep("sweep70free", "pythia-70m", "pythia-160m", [1, 2, 3, 4, 5, 6], [1, 2], data_free=True)
        elif grp == "comm70":
            run_comm("comm70", "pythia-70m", "pythia-160m", 2, 1)   # (LA, LB) chosen on validation (sweep70)
        elif grp == "comm70self":
            # comm-objective self-sampled bridges for the main pair (missing from comm70); saved alongside comm70's
            run_comm("comm70self", "pythia-70m", "pythia-160m", 2, 1, objectives=("comm",), comm_self=True,
                     only=("self-sampled",), save_as="comm70")
        elif grp == "variants":
            # probe-text variants on the same 24 topic areas: Claude encyclopedia text (files w26-w49), GPT
            # encyclopedia text (same prompt), Claude fiction/dialogue (off-register); comm objective, seeds 0-1
            g = Group("variants"); pair = pb.Pair("pythia-70m", "pythia-160m", 2, 1); M0 = pair.anchors()
            test = pb.disjoint_windows(TEST_IDS, NTEST, P + S + 1, 123)
            read = lambda d, pre, idx: "\n".join(open(f"{d}/{pre}{i}.txt").read() for i in idx
                                                  if os.path.exists(f"{d}/{pre}{i}.txt"))
            texts = {"claudewiki": read(PR, "w", range(26, 50)),
                     "gptwiki": read(PR + "_gpt", "g", range(24)),
                     "claudefiction": read(PR + "_fiction", "f", range(24))}
            n_eq = min(len(t) for t in texts.values())                 # equal character budget for every source
            texts = {k: t[:n_eq] for k, t in texts.items()}
            for k, t in texts.items():
                g.p(f"{k}: {len(t)} chars, {len(pb.ids_of(t))} tokens")
            for sd in (0, 1):
                for k, t in texts.items():
                    t0 = time.time()
                    br = pb.fit(pair, pb.random_windows(pb.ids_of(t), NPROBE, P + S + 1, 100 + sd), "comm", M0, P=P,
                                seed=sd, log=g.p)
                    g.add(f"{k} | seed {sd} | matched", *pb.score(
                        lambda s: (pb.bridged_logits(pair, br, s, "comm", P), s[:, P + 1:]), test))
                    torch.save(br.state_dict(), f"{pb.OUT}/variants_comm_{k}_s{sd}.pt")
                    g.p(f"    ({time.time() - t0:.0f} s)")
        elif grp == "capacity":
            # bridge capacity in the communication setting (70m L2 -> 160m L1, comm objective, seed 0)
            g = Group("capacity"); pair = pb.Pair("pythia-70m", "pythia-160m", 2, 1); M0 = pair.anchors()
            test = pb.disjoint_windows(TEST_IDS, NTEST, P + S + 1, 123)
            for pname, ids in [("real", TRAIN_IDS), ("Claude", pb.ids_of(CLAUDE3X))]:
                probe = pb.random_windows(ids, NPROBE, P + S + 1, 100)
                for kind, kw in [("rank16", {"rank": 16}), ("rank64", {"rank": 64}), ("mlp2048", {"hidden": 2048})]:
                    t0 = time.time()
                    br = pb.fit(pair, probe, "comm", M0, P=P, seed=0, log=g.p, **kw)
                    g.add(f"{pname} {kind} | matched", *pb.score(
                        lambda s: (pb.bridged_logits(pair, br, s, "comm", P), s[:, P + 1:]), test))
                    torch.save(br.state_dict(), f"{pb.OUT}/capacity_comm_{pname}-{kind}_s0.pt")
                    g.p(f"    ({time.time() - t0:.0f} s)")
        elif grp == "diag":
            run_diag()
        elif grp == "sweep410_160":
            run_commsweep(grp, "pythia-410m", "pythia-160m", [2, 4, 6, 8, 12, 16], [1])
        elif grp == "sweep160_410":
            run_commsweep(grp, "pythia-160m", "pythia-410m", [1, 2, 4], [1, 2])
        elif grp.startswith("auto:"):
            # auto:<name>:<sweep group>:<A>:<B>:<seeds>  -- (LA, LB) = lowest validation CE in out_pb/<sweep>.json;
            # comm objective only, on real, LLM-written and self-sampled text
            _, name, sweep, a, b, sds = grp.split(":")
            sw = json.load(open(f"{pb.OUT}/{sweep}.json"))
            best = min((v["ce"], k) for k, v in sw.items() if k.startswith("A L") and " | " not in k)[1]   # matched condition only
            la, lb = (int(x) for x in re.match(r"A L(\d+) -> B L(\d+)", best).groups())
            print(f"{name}: chose LA={la}, LB={lb} from {sweep} ({best})", flush=True)
            run_comm(name, f"pythia-{a}", f"pythia-{b}", la, lb, seeds=tuple(int(x) for x in sds.split(",")),
                     objectives=("comm",), comm_self=True)
        elif grp.startswith("comm:"):
            # comm:<name>:<A>:<B>:<LA>:<LB>:<seeds comma-separated>, layers chosen beforehand on validation
            _, name, a, b, la, lb, sds = grp.split(":")
            run_comm(name, f"pythia-{a}", f"pythia-{b}", int(la), int(lb), seeds=tuple(int(x) for x in sds.split(",")))
        elif grp == "textscale":
            run_textscale()
