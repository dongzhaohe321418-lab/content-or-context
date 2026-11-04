"""Code-allowed arm, run like the authors' Experiments 3/6/8: the model has no tools of its own. Each turn it replies
with one ```python block (or, when done, a final answer without code); we run the block under macOS sandbox-exec
(no network, no reads of the workspace, writes only inside its folder) with a numpy/scipy-only Python, and send the
printed output back as the next message. The conversation is replayed in full every turn (stateless claude -p).

    python harness.py <pair> <run> [--turns 12] [--effort high]
"""
import argparse, json, os, re, subprocess, time

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = os.environ["BRIDGE_SCRATCH"]
PY = f"{S}/sbenv/bin/python"
EMPTY = f"{S}/empty"
FORBIDDEN = re.compile(os.environ.get("BRIDGE_FORBID", r"sklearn|load_digits|datasets|urllib|requests|socket|pip |curl|wget|\.\./|/Users/|site-packages"))
BLOCK = bool(os.environ.get("BRIDGE_BLOCK"))      # refuse to run a block that hits FORBIDDEN

PROTOCOL = """
HOW THIS WORKS. You have no tools of your own. Each of your replies must be EITHER exactly one ```python block (we
run it with the Python above, working directory = your folder, a fresh process each time, 120 s limit; files you
write persist; we send back what it prints, truncated to 6,000 characters) OR, once bridge.npz is saved, your final
answer with no code block. You have at most {turns} code runs. Keep each reply short: brief reasoning, then the block.
"""


def sandbox_profile(folder):
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-read* (subpath "{W}"))
(deny file-write*)
(allow file-write* (subpath "{folder}") (subpath "/dev") (subpath "/private/var/folders"))
"""


def ask(prompt, effort):
    r = subprocess.run(
        ["claude", "-p", "--model", "claude-opus-5-5", "--effort", effort, "--tools", "", "--system-prompt",
         "Answer the user's request.", "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands",
         "--no-session-persistence", "--output-format", "json"],
        input=prompt, capture_output=True, text=True, cwd=EMPTY, timeout=3600)
    d = json.loads(r.stdout)
    return d["result"], d


def run_code(code, folder):
    prof = os.path.join(folder, ".sb")
    open(prof, "w").write(sandbox_profile(folder))
    open(os.path.join(folder, "_turn.py"), "w").write(code)
    try:
        r = subprocess.run(["sandbox-exec", "-f", prof, PY, "-I", "_turn.py"], cwd=folder, capture_output=True,
                           text=True, timeout=120)
        out = (r.stdout + ("\n[stderr]\n" + r.stderr if r.stderr.strip() else ""))
    except subprocess.TimeoutExpired:
        out = "[timed out after 120 s]"
    return out[-6000:] if len(out) > 6000 else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pair"); ap.add_argument("run")
    ap.add_argument("--turns", type=int, default=12); ap.add_argument("--effort", default="high")
    a = ap.parse_args()
    folder = f"{S}/sbx/{a.pair}_r{a.run}"
    log = f"{W}/bridge/out/llm/{a.pair}_sandbox_r{a.run}"
    task = open(f"{folder}/TASK.md").read().replace(
        "Work only inside this folder. When bridge.npz is saved, reply with a short description of your method.",
        "Work only inside this folder.") + PROTOCOL.format(turns=a.turns)
    convo, runs, turns = [], 0, []
    while True:
        prompt = task + "".join(f"\n\n=== YOUR REPLY {i+1} ===\n{r}\n\n=== OUTPUT ===\n{o}" for i, (r, o) in enumerate(convo))
        prompt += "\n\nYour next reply:" if convo else ""
        t0 = time.time()
        reply, meta = ask(prompt, a.effort)
        blocks = re.findall(r"```python\s*\n(.*?)```", reply, flags=re.S)
        rec = {"turn": len(convo) + 1, "reply": reply, "wall_s": round(time.time() - t0), "out_tokens": meta["usage"].get("output_tokens")}
        if not blocks:
            rec["final"] = True
            turns.append(rec)
            break
        code = blocks[0]
        rec["code"], rec["forbidden_hits"] = code, FORBIDDEN.findall(code)
        if BLOCK and rec["forbidden_hits"]:
            out = f"[not run: the block uses {sorted(set(rec['forbidden_hits']))}, which the rules forbid]"
        else:
            out = run_code(code, folder) if runs < a.turns else "[no code runs left: give your final answer now]"
        runs += 1
        rec["output"] = out
        turns.append(rec)
        convo.append((reply, out))
        print(f"turn {rec['turn']}: {len(code)} chars code, {len(out)} chars out, hits={rec['forbidden_hits']}", flush=True)
        if runs > a.turns + 1:
            break
    with open(log + ".jsonl", "w") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
    bp = f"{folder}/bridge.npz"
    if os.path.exists(bp):
        import shutil
        shutil.copy(bp, log + "_bridge.npz")
    json.dump({"result": turns[-1]["reply"], "num_turns": len(turns), "usage": {"output_tokens": sum(t["out_tokens"] or 0 for t in turns)}},
              open(log + ".json", "w"))
    print("done:", len(turns), "turns, bridge saved" if os.path.exists(bp) else "NO bridge")


if __name__ == "__main__":
    main()
