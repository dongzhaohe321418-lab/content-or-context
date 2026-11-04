"""Population and provenance audits quoted in the paper's appendices.
  - out_pb/general_audit.json: eligible IMDB reviews, domain populations, GPT-2 prefix re-tokenisation statistics;
  - out_lm/probe_overlap_audit.json: verbatim overlap of the LLM-written Python probe text with the stdlib corpus.
    python pb_audits.py"""
import glob, json, os, sys, sysconfig
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/out_pb"
sys.path.insert(0, HERE)
from pb import REV
tp = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m", revision=REV["EleutherAI/pythia-70m"]); tg = AutoTokenizer.from_pretrained("openai-community/gpt2", revision=REV["openai-community/gpt2"])
rev = load_dataset("stanfordnlp/imdb", revision=REV["stanfordnlp/imdb"])["test"]["text"]
elig = sum(len(tp(r)["input_ids"]) // 161 >= 2 for r in rev)
out = {"reviews_total": len(rev), "reviews_eligible_ge_2_windows": elig, "share": round(elig / len(rev), 4)}
for g in ["dom_code_comm70", "dom_reviews_comm70"]:
    z = np.load(f"{OUT}/{g}_swap.npz"); d = json.load(open(f"{OUT}/{g}_decomp.json"))["_meta"]
    out[g] = {"windows": int(len(z["article"])), "documents": int(len(set(z["article"]))),
              "with_same_doc": int(z["has_same"].sum()), "decomp_windows": d["n_windows"], "decomp_docs": d["n_articles"]}
wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", revision=REV["Salesforce/wikitext"]); text = "".join(wt["test"]["text"][:6000])
ids = np.array(tp(text)["input_ids"]); L = 161
starts = np.sort(np.random.default_rng(123).choice(len(ids) // L, 512, replace=False)) * L
la = np.array([len(tg(t, add_special_tokens=False)["input_ids"]) for t in tp.batch_decode(np.stack([ids[s:s + 128] for s in starts]))])
out["gpt2_prefix_tokens"] = {"min": int(la.min()), "median": float(np.median(la)), "max": int(la.max()),
                             "truncated_windows": int((la > la.min()).sum()), "mean_kept_share": round(float((la.min() / la).mean()), 3)}
tq = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", revision=REV["Qwen/Qwen2.5-0.5B"])
qids = np.array(tq(text)["input_ids"])
qs = np.sort(np.random.default_rng(123).choice(len(qids) // L, 512, replace=False)) * L
qpre = tq.batch_decode(np.stack([qids[s:s + 128] for s in qs]))
ql = np.array([len(tg(t, add_special_tokens=False)["input_ids"]) for t in qpre])
dig = np.array([sum(c.isdigit() for c in t) / len(t) for t in qpre])
out["gpt2_into_qwen_prefix_tokens"] = {"min": int(ql.min()), "median": float(np.median(ql)), "max": int(ql.max()),
                                       "mean_kept_share": round(float((ql.min() / ql).mean()), 3),
                                       "digit_share_shortest": round(float(dig[ql.argmin()]), 3),
                                       "digit_share_median": round(float(np.median(dig)), 3),
                                       "corr_length_digits": round(float(np.corrcoef(ql, dig)[0, 1]), 3)}
# cross-family probe/validation filtering (windows whose re-tokenised sender prefix is shorter than P_A are dropped)
claude = open(f"{HERE}/out_pythia/probes/all3x.txt").read()
train_text = "".join(wt["train"]["text"][:30000]); val_text = "".join(wt["validation"]["text"][:6000])
filt = {}
for name, a_id, b_id in [("gpt2_pythia160", "openai-community/gpt2", "EleutherAI/pythia-160m"),
                         ("gpt2_qwen05", "openai-community/gpt2", "Qwen/Qwen2.5-0.5B"),
                         ("opt125_gpt2", "facebook/opt-125m", "openai-community/gpt2")]:
    ta, tb = AutoTokenizer.from_pretrained(a_id, revision=REV[a_id]), AutoTokenizer.from_pretrained(b_id, revision=REV[b_id])
    PA = json.load(open(f"{OUT}/x_{name}_meta.json"))["P_A"]
    def kept(text, n, seed, disjoint=False):
        ids_ = np.array(tb(text)["input_ids"]); r = np.random.default_rng(seed)
        st = np.sort(r.choice(len(ids_) // L, n, replace=False)) * L if disjoint else r.integers(0, len(ids_) - L, n)
        pre = tb.batch_decode(np.stack([ids_[s_:s_ + 128] for s_ in st]))
        return int(sum(len(ta(t, add_special_tokens=False)["input_ids"]) >= PA for t in pre))
    filt[name] = {"P_A": PA, "validation_256": kept(val_text, 256, 7, True)}
    for sd in (0, 1):
        filt[name][f"seed{sd}"] = {"real_4096": kept(train_text, 4096, 100 + sd), "llm_4096": kept(claude, 4096, 100 + sd)}
out["cross_family_filtering"] = filt
json.dump(out, open(f"{OUT}/general_audit.json", "w"), indent=1)
std = "".join(open(f, errors="replace").read() for f in sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py"))))
res = {}
for name in ["p0", "p2"]:
    t = open(f"{HERE}/out_lm/probes/{name}.txt").read(); subs = [t[i:i + 50] for i in range(0, len(t) - 50, 50)]
    hit = sum(s in std for s in subs); train = std[: int(0.9 * len(std))]      # lm_common.corpus(): first 90% trains
    hit_tr = sum(s in train for s in subs)
    res[name] = {"chars": len(t), "windows50": len(subs), "verbatim_in_stdlib": hit, "share": round(hit / len(subs), 4),
                 "verbatim_in_training_split": hit_tr, "share_training": round(hit_tr / len(subs), 4)}
allt = open(f"{HERE}/out_lm/probes/all.txt").read()
res["non_code_share_of_all"] = round((len(open(f"{HERE}/out_lm/probes/p1.txt").read()) + len(open(f"{HERE}/out_lm/probes/p3.txt").read())) / len(allt), 4)
json.dump(res, open(f"{HERE}/out_lm/probe_overlap_audit.json", "w"), indent=1)
print(out, res)
