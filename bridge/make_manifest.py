"""Write bridge/MANIFEST.md: analysis entry points, pinned model / dataset revisions (from the local Hugging Face
cache), and SHA-256 of every script, fitted bridge and result file the paper uses. Superseded outputs are listed
separately so they cannot be mistaken for current results.

    python make_manifest.py
"""
import glob, hashlib, os

HERE = os.path.dirname(os.path.abspath(__file__))
HUB = os.path.expanduser("~/.cache/huggingface/hub")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stdlib_identity():
    import sys, sysconfig
    files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    h = hashlib.sha256()
    for f in files:
        h.update(os.path.basename(f).encode()); h.update(open(f, "rb").read())
    return f"Python {sys.version.split()[0]} standard library, {len(files)} top-level .py files, SHA-256 {h.hexdigest()[:16]}"


def revisions():
    out = []
    for d in sorted(glob.glob(f"{HUB}/models--*") + glob.glob(f"{HUB}/datasets--*")):
        ref = os.path.join(d, "refs", "main")
        rev = open(ref).read().strip() if os.path.exists(ref) else "?"
        out.append(f"| {os.path.basename(d).replace('--', '/', 1).replace('--', '/')} | `{rev}` |")
    return out


ENTRY = """## Entry points (in order)

| Step | Command | Output |
|---|---|---|
| Layer selection, WikiText validation | `python pb_run.py sweep70` | `out_pb/sweep70.*` |
| Layer selection, corpus-free | `python pb_run.py sweep70free` | `out_pb/sweep70free.*` |
| Main fits (hand-over and comm objectives) | `python pb_run.py comm70` then `python pb_run.py comm70self` | `out_pb/comm70_*.pt`, `comm70*.json/.npz` |
| Donor controls (v3) | `python pb_mismatch.py comm70 pythia-70m pythia-160m 2 1 5` | `out_pb/comm70_swap.npz/.json` |
| Decomposition + bootstrap | `python pb_decompose.py comm70` | `out_pb/comm70_decomp.json` |
| Table | `python pb_tables.py comm70 ../paper/tab_comm70.tex` | `paper/tab_comm70.tex` |
| Per-position / truncation | `python pb_extra.py comm70 pythia-70m pythia-160m 2 1` | `out_pb/comm70_extra.*` |
| Receiver-state reference vs depth | `python pb_ceiling.py > out_pb/ceiling.txt` | `out_pb/ceiling.txt` |
| Implementation checks | `python pb_checks.py` | `out_pb/checks.json` |
| Hand-over diagnosis | `python pb_run.py diag` | `out_pb/diag.*` |
| Larger Pythia pairs | `python pb_run.py sweep410_160 sweep160_410 auto:comm410_160:sweep410_160:410m:160m:0,1,2 auto:comm160_410:sweep160_410:160m:410m:0,1`; `python pb_mismatch.py comm410_160 pythia-410m pythia-160m 6 1 5`; `python pb_mismatch.py comm160_410 pythia-160m pythia-410m 4 1 5`; `python pb_decompose.py comm410_160 comm160_410` | `out_pb/comm410_160*`, `comm160_410*` |
| Text scaling | `python pb_run.py textscale`; `python pb_mismatch.py textscale pythia-70m pythia-160m 2 1 5`; `python pb_decompose.py textscale` | `out_pb/textscale*` |
| Cross-family pairs | `python pbx.py gpt2_pythia160 openai-community/gpt2 EleutherAI/pythia-160m 0,1`; `python pbx.py gpt2_qwen05 openai-community/gpt2 Qwen/Qwen2.5-0.5B 0,1`; `python pbx.py opt125_gpt2 facebook/opt-125m openai-community/gpt2 0,1` (with `out_pb/x_lean.txt` and `x_skip.txt` as kept); `python pb_decompose.py x_gpt2_pythia160 x_gpt2_qwen05 x_opt125_gpt2` | `out_pb/x_*` |
| Irrelevant text vs latent | `python pb_irrelevant.py` | `out_pb/irrelevant_contrast.json` |
| Sender comparison (paired seeds and donors) | `python pb_sender.py` | `out_pb/sender_compare.json` |
| Out-of-domain | `python pb_domain.py code comm70 pythia-70m pythia-160m 2 1`; `python pb_domain.py reviews comm70 pythia-70m pythia-160m 2 1`; `python pb_decompose.py dom_code_comm70 dom_reviews_comm70` | `out_pb/dom_*` |
| Probe-text variants | `python pb_run.py variants`; `python pb_mismatch.py variants pythia-70m pythia-160m 2 1 5`; `python pb_decompose.py variants` | `out_pb/variants*` |
| Capacity | `python pb_run.py capacity`; `python pb_mismatch.py capacity pythia-70m pythia-160m 2 1 5`; `python pb_decompose.py capacity` | `out_pb/capacity*` |
| Generalisation table | `python pb_gen_table.py ../paper/tab_general.tex` | `paper/tab_general.tex` |
| Domain window identities | `python pb_domain_windows.py` | `out_pb/dom_*_windows.json` |
| Population audits | `python pb_audits.py` | `out_pb/general_audit.json`, `out_lm/probe_overlap_audit.json` |
| Text vs bridge per transmitted position (Bonferroni) | `python pb_bandwidth.py` | `out_pb/bandwidth_test.json` |
| Figure 1 | `python ../paper/figs/make_fig_comm.py out_pb` (alignment helper included in `paper/figs/`) | `paper/figs/fig_comm.*` |
| Paper | `cd ../paper && tectonic -X compile main.tex` | `paper/main.pdf` |

The run scripts actually used are kept as `out_pb/queue*.sh`. The small-testbed harness (`harness.py`,
`run_toolfree.sh`) additionally needs a sandbox directory (`BRIDGE_SCRATCH`), per-run task files
(`sandbox_task*.md`) and machine-specific paths; that set-up is only partly documented. Run Python with `-I`. Hardware: one Apple M5 Pro (Metal backend, float32). Environments: `requirements-lock.txt`
(analysis, Python 3.12) and `../paper/figs/requirements-fig-lock.txt` (figure, Python 3.9). Every model and dataset
load is pinned to the revisions below through explicit revision dictionaries (`pb.REV`, `pythia.REV`).
"""

SUPERSEDED = """## Superseded outputs (not used by the paper)

- `comm70.json/.npz` keys `... | mismatched message`: the first donor control (messages rolled within each batch of 16;
  83% of donors from the same article). Matched values in these files are still valid and identical to `comm70_swap`.
- `comm70_summary.json` (pb_analyze.py), `batchroll_comm70_extra.*`: analyses built on that control.
- `v2_comm70_swap.*`, `v2_comm70_decomp.json`: five random donors without whole-article restriction; superseded by v3.
- `farshift_comm70_extra.*`: per-position analysis with a deterministic half-shift donor; superseded.
- `textscale_aborted_batchroll.*`: an aborted text-scaling run with the first control.
- `unpaired_sender_compare.json`: sender comparison whose bootstrap drew seeds and donors independently for the two
  senders; superseded by the paired `sender_compare.json`.
- Legacy output keys: early outputs (`comm70.json/.npz`, `sweep70*`, `ceiling.txt`) label the receiver-state reference
  "oracle message ... (protocol ceiling)" or "oracle"; these are the same quantity, renamed in the paper.
- `sanity.*` uses the first control for its comm entries; only its hand-over identity check is cited.
- `variants_aborted*`, `textscale_aborted*`: runs stopped before completion (incomplete probe files / trimmed design).
- Fiction files f18, f20, f21, f23 were regenerated after interrupted calls (`regen_fiction.sh`); failed attempts were
  overwritten and are not archived. The GPT corpus was generated with `codex exec --sandbox read-only` (model not pinned,
  transcripts not kept).
- Cross-family self-sample probe text (pbx.py) was generated in memory and not saved; its filtering counts are therefore
  not available (real-text and LLM-text counts are in out_pb/general_audit.json).
"""


def main():
    lines = ["# Manifest", "", ENTRY, "## Revisions (local Hugging Face cache refs at manifest time)", "",
             "Models and datasets were loaded from this cache; it was not updated during the study, but these refs were",
             "recorded afterwards rather than at each run.", "", "| Repository | Revision |", "|---|---|"] + revisions() + [
             "", f"Code domain input: {stdlib_identity()}.", "", "## File hashes (full SHA-256)", "",
             "| File | SHA-256 |", "|---|---|"]
    rec = lambda d: sorted(f for f in glob.glob(f"{HERE}/{d}/**/*", recursive=True) if "__pycache__" not in f)
    files = sorted(glob.glob(f"{HERE}/*.py")) + sorted(glob.glob(f"{HERE}/*.sh")) + sorted(glob.glob(f"{HERE}/*.md"))
    files = [f for f in files if not f.endswith("REPORT.md")]                    # early notes, not released
    files += [f"{HERE}/requirements-lock.txt"]
    files += rec("out_pb") + rec("out_pythia") + rec("out_lm") + rec("out")          # results, logs, probes, bridges
    files += [f for f in rec("../paper") if not f.endswith((".aux", ".bbl", ".blg", ".out", ".bak", ".log"))
              and not os.path.basename(f).startswith(("review_", "refs_raw", "main_iclr", "iclr2026"))]  # sources, PDF, figures
    files = [f for f in files if os.path.isfile(f) and not f.endswith("MANIFEST.md")]
    for f in files:
        lines.append(f"| `{os.path.relpath(f, HERE)}` | `{sha(f)}` |")
    lines += ["", SUPERSEDED]
    open(f"{HERE}/MANIFEST.md", "w").write("\n".join(lines) + "\n")
    print(f"wrote MANIFEST.md with {len(files)} files")


if __name__ == "__main__":
    main()
