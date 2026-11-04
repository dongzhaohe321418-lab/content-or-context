"""Generate the paper's decomposition tables and quoted numbers directly from out_pb/<group>_decomp.json, so that
every reported value is rounded once from unrounded estimates.

    python pb_tables.py <group> <out .tex>
"""
import json, os, sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_pb")
ROWS = [("Random map", "--", "Random map"), ("Weight-only anchors", "--", "Weight-only anchors"),
        None,
        ("Self-samples", "hand-over", "self-sampled [full objective]"),
        ("Self-samples", "comm", "self-sampled [comm objective]"),
        ("LLM-written text", "hand-over", "Claude [full objective]"),
        ("LLM-written text", "comm", "Claude [comm objective]"),
        ("Real text", "hand-over", "real [full objective]"),
        ("Real text", "comm", "real [comm objective]"),
        None,
        ("Receiver-state reference", "--", "Receiver-state reference"),
        ("Text (prefix as tokens)", "--", "Text (prefix as tokens)")]


def num(x, d=3):
    s = f"{x:.{d}f}"
    if s.startswith("-0") and float(s) == 0:
        s = s[1:]
    return s.replace("-", "$-$")


def iv(q, d=3):
    return f"{num(q['est'], d)} [{num(q['ci'][0], d)}, {num(q['ci'][1], d)}]"


def main(group, out):
    r = json.load(open(f"{OUT}/{group}_decomp.json"))
    lines = []
    for row in ROWS:
        if row is None:
            lines.append("\\midrule"); continue
        lab, obj, key = row
        if key not in r:
            continue
        v = r[key]
        ref = key == "Receiver-state reference"
        cr = "1" if ref else iv(v["content_ratio_to_reference"], 2)
        tr = "1" if ref else iv(v["total_ratio_to_reference"], 2)
        name = f"{lab} ({obj})" if obj != "--" else lab
        lines.append(" & ".join([name, num(v["total"]["est"]), num(v["generic"]["est"]), iv(v["content"]), cr, tr])
                     + " \\\\")
    head = ["\\begin{tabular}{lccccc}", "\\toprule",
            "Message source (objective) & Total & Generic & Content & Content ratio & Net ratio \\\\", "\\midrule"]
    open(out, "w").write("\n".join(head + lines + ["\\bottomrule", "\\end{tabular}"]) + "\n")
    print("\n".join(lines))
    print("no-context CE", num(r["_meta"]["no_context_ce"]), "| windows", r["_meta"]["n_windows"],
          "| articles", r["_meta"]["n_articles"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
