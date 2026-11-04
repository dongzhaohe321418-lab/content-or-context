"""Knowledge-free probe text: sample sequences from A and B themselves (temperature 1, start = newline), no corpus,
no LLM-written text. Writes out_lm/probes/selfsample.txt for lm_taskfit.py."""
import os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_common as L
from lm_taskfit import load_models

A, B = load_models()
mx.random.seed(0)
out = []
for m in (A, B):
    for _ in range(4):                                   # 4 x 256 sequences of 64 characters per model
        idx = mx.zeros((256, 1), dtype=mx.int32)        # newline
        for t in range(L.CTX - 1):
            nxt = mx.random.categorical(m(idx)[:, -1, :])
            idx = mx.concatenate([idx, nxt[:, None].astype(mx.int32)], axis=1)
        out += ["".join(L.CHARS[i] for i in row) for row in np.array(idx).tolist()]
open(f"{L.OUT}/probes/selfsample.txt", "w").write("".join(out))
print(len(out), "sequences;", sum(map(len, out)), "characters; sample:", repr(out[0][:120]))
