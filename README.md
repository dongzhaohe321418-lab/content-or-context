# Content or Context? Measuring Latent Communication Between Language Models

Code, results, fitted bridges and LaTeX sources for the preprint

> Zhaohe Dong. *Content or Context? Measuring Latent Communication Between Language Models, with Bridges Fitted
> on LLM-Written Text.* 2026. ([paper/main.pdf](paper/main.pdf))

A sender model reads a 128-token prefix and passes its hidden states through a learned bridge to a receiver model
that reads only the 32-token suffix. Donor messages from other articles and from the same article separate the
gain into a generic part, an article-level part and a window-specific part. Bridges fitted only on LLM-written text
recover 82% [79, 85] of the content gain of a bridge fitted on real text (70M → 160M Pythia).

## Layout

| Path | Contents |
|---|---|
| `bridge/pb*.py` | Protocol, bridges, fitting, donor controls, bootstrap and tables for the main Pythia suite |
| `bridge/pbx.py` | Cross-family pairs (GPT-2, Qwen2.5, OPT, Pythia) |
| `bridge/out_pb/` | Logs, per-window score arrays (`.npz`), summaries (`.json`), fitted bridges (`.pt`) and the run scripts used (`queue*.sh`) |
| `bridge/out_lm/`, `bridge/out_pythia/`, `bridge/out/` | Earlier small-testbed and Pythia-seed experiments (Appendix of the paper), with probe texts |
| `bridge/MANIFEST.md` | Entry points in order, pinned model and dataset revisions, superseded outputs, and full SHA-256 of every released file |
| `paper/` | LaTeX sources (`main.tex`), PDF and Figure 1 with its plotting script and QA files |

## Reproducing

Analysis environment: Python 3.12 with `bridge/requirements-lock.txt`; figure environment: Python 3.9 with
`paper/figs/requirements-fig-lock.txt`. Experiments ran on one Apple M5 Pro (Metal backend, float32). Every model
and dataset is loaded at the Hugging Face revision recorded in `pb.REV` / `pythia.REV`.

```bash
cd bridge
python -I pb_decompose.py comm70          # recompute the main decomposition from the saved arrays
python -I pb_tables.py comm70 ../paper/tab_comm70.tex
```

The full sequence of commands, from layer selection to the paper, is in `bridge/MANIFEST.md`. The saved arrays support
numerical reanalysis of every reported result without refitting. The small-testbed harness (`harness.py`,
`run_toolfree.sh`) needs a sandbox set-up that is only partly documented.

Not included: the Hugging Face model weights and datasets (downloaded at the pinned revisions), self-sample probe
text (generated in memory), and transcripts of the second LLM that wrote part of the probe text. See the Code
paragraph of the Reproducibility appendix in the paper.

## Licence

Code: MIT (see `LICENSE`). Results, probe texts and the paper: CC BY 4.0. WikiText-103 and IMDB are not
redistributed; result files refer to them by document index only.
