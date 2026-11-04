"""Score every LLM bridge in out/llm/<pair>_<arm>_r<k>.json (claude -p JSON output) on the test set."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def main():
    Xtr, Xte, ytr, yte = C.data()
    rows = []
    for f in sorted(glob.glob(f"{OUT}/llm/*.json")):
        tag = os.path.basename(f)[:-5]
        name = tag.split("_")[0]
        A, B = C.load(f"{OUT}/{name}_A.npz"), C.load(f"{OUT}/{name}_B.npz")
        hA, hB = A["W"][0].shape[1], B["W"][0].shape[1]
        try:
            d = json.load(open(f))
        except json.JSONDecodeError:            # still running
            print(f"{tag:28s} (not finished)")
            continue
        br = C.parse_bridge(d.get("result", ""), hA, hB)
        if br is None and os.path.exists(f"{OUT}/llm/{tag}_bridge.npz"):      # sandbox arm writes its bridge to a file
            z = np.load(f"{OUT}/llm/{tag}_bridge.npz"); br = (z["M"], z["c"])
        acc = None if br is None else C.stitched_acc(A, B, *br, Xte, yte)
        rows.append({"run": tag, "acc": acc, "turns": d.get("num_turns"), "out_tokens": d["usage"].get("output_tokens")})
        print(f"{tag:28s} acc={'PARSE FAIL' if acc is None else f'{acc:.3f}'}  turns={d.get('num_turns')}")
    json.dump(rows, open(f"{OUT}/llm_scores.json", "w"), indent=1)


if __name__ == "__main__":
    main()
