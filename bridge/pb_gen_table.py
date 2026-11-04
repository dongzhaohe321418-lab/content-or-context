"""Summary table of the generalisation experiments, generated from the <group>_decomp.json files so that every number
is rounded once. One row per setting; for each probe source, the content gain and the net gain (total reduction in CE
relative to no context), both relative to the receiver-state reference of that setting, with 95% intervals.

    python pb_gen_table.py <out .tex>
"""
import json, os, sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_pb")
SETTINGS = [  # (label, group, objective tag used in that group's condition names)
    ("Pythia 70M$\\to$160M (main)", "comm70", "comm objective"),
    ("Pythia 410M$\\to$160M", "comm410_160", "comm objective"),
    ("Pythia 160M$\\to$410M", "comm160_410", "comm objective"),
    ("GPT-2$\\to$Pythia 160M", "x_gpt2_pythia160", "comm objective"),
    ("GPT-2$\\to$Qwen2.5 0.5B", "x_gpt2_qwen05", "comm objective"),
    ("OPT 125M$\\to$GPT-2", "x_opt125_gpt2", "comm objective"),
    ("Code (70M$\\to$160M)", "dom_code_comm70", "comm objective"),
    ("Reviews (70M$\\to$160M)", "dom_reviews_comm70", "comm objective"),
]
SOURCES = [("real", "Real text"), ("Claude", "LLM text"), ("self-sampled", "Self-samples")]


def fmt(q, d=2):
    s = lambda x: f"{x:.{d}f}".replace("-", "$-$")
    return f"{s(q['est'])} [{s(q['ci'][0])}, {s(q['ci'][1])}]"


def main(out):
    lines = ["\\begin{tabular}{llccc}", "\\toprule",
             "Setting & Probe text & Ref.\\ content & Content ratio & Net ratio \\\\", "\\midrule"]
    first = True
    for label, group, obj in SETTINGS:
        p = f"{OUT}/{group}_decomp.json"
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        if not first:
            lines.append("\\addlinespace")
        first = False
        ref = f"{r['Receiver-state reference']['content']['est']:.2f}"
        rows = [(name, f"{key} [{obj}]") for key, name in SOURCES if f"{key} [{obj}]" in r]
        for i, (name, k) in enumerate(rows):
            lines.append(" & ".join([label if i == 0 else "", name, ref if i == 0 else "",
                                     fmt(r[k]["content_ratio_to_reference"]), fmt(r[k]["total_ratio_to_reference"])])
                         + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
