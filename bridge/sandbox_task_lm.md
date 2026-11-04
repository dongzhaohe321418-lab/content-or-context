TASK. Two small character-level language models, A and B, were trained separately (different random initialisations
and different widths: A has width 64, B has width 96) on the same text: Python source code (standard-library style
modules). Both predict the next character. They share the same 96-character vocabulary: index 0 is the newline "\n",
and index i (1..95) is the character chr(31 + i), i.e. the printable ASCII characters from " " (32) to "~" (126). Any
other character was mapped to "?". Context length is 64 characters.

Files in this folder: A.npz and B.npz (numpy arrays, names below; d = 64 for A, 96 for B; 4 heads; 4 blocks).
Forward pass of a model on a sequence of token ids t[0..T-1] (T <= 64):
  r = tok.weight[t] + pos.weight[0:T]                              (tok.weight: 96 x d, pos.weight: 64 x d)
  for each block k = 0..3 (names blocks.k.*):
      y = LN(r; ln1.weight, ln1.bias)
      q = y @ attn.query_proj.weight.T ; k_ = y @ attn.key_proj.weight.T ; v = y @ attn.value_proj.weight.T  (no biases)
      split q, k_, v into 4 heads of size d/4 (head h = columns h*d/4 .. (h+1)*d/4 - 1)
      per head: a = softmax(q_h @ k_h.T / sqrt(d/4) + causal mask) @ v_h ; concatenate heads in order -> o
      r = r + o @ attn.out_proj.weight.T
      y = LN(r; ln2.weight, ln2.bias)
      r = r + gelu(y @ fc1.weight.T + fc1.bias) @ fc2.weight.T + fc2.bias     (gelu = exact erf form)
  logits = LN(r; lnf.weight, lnf.bias) @ head.weight.T + head.bias
  LN(x; w, b) = (x - mean(x)) / sqrt(var(x) + 1e-5) * w + b   over the last axis (biased variance)
  causal mask: position i attends to positions 0..i only.

We want A and B to communicate without any text: A reads a text, runs its first TWO blocks (k = 0, 1) and sends its
residual stream r2A (one 64-vector per position) to B. B never sees the text. A "bridge" (M, c) turns A's message into
B's residual stream at the same point:
    r2B_hat = r2A @ M + c          (M: 64 x 96, c: 96), applied at every position
then B runs its blocks 2 and 3, its final norm and its head as usual. The bridge is scored by B's next-character
accuracy on unseen Python text (B alone gets about 63%; A alone about 63%; a random bridge about 6%).

Design the bridge. Save it as bridge.npz in this folder with arrays M (shape 64 x 96) and c (shape 96).

RULES
- Python with numpy and scipy is at: PYTHON_PATH (use it to run your code). Nothing else may be installed.
- You have NO training text: do not open, read, list or import-and-inspect any file outside this folder (that includes
  the Python standard library's source files: no inspect/getsource, linecache, glob, os.listdir/os.walk, sysconfig,
  __file__, importlib.resources), and no internet. Code that does so is refused without running.
- You may write probe text yourself from your own knowledge if you find that useful.
- Work only inside this folder. When bridge.npz is saved, reply with a short description of your method.
