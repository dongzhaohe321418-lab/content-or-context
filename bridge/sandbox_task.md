TASK. Two small neural networks, A and B, were trained separately (different random initialisations, and possibly
different widths and activations) on the same task: classifying 8 x 8 grayscale images of handwritten digits into the
10 classes 0-9. Input x has 64 numbers: x[r*8 + c] is the pixel in row r (0 = top) and column c (0 = left), scaled to
[0, 1] (0 = background, 1 = full ink). Logit k is the score of digit k.

Files in this folder: A.npz and B.npz. Each holds W0, b0, W1, b1, W2, b2 (numpy arrays) and act ('relu' or 'tanh').
Forward pass of a model: h1 = act(x @ W0 + b0); h2 = act(h1 @ W1 + b1); logits = h2 @ W2 + b2.

We want A and B to communicate without any text: A reads an image, computes its first hidden layer h1A, and sends that
vector to B. B never sees the image. A "bridge" (M, c) turns A's message into the PRE-activation of B's first layer:
    z1B_hat = h1A @ M + c
then B continues exactly as if z1B_hat were its own x @ W0 + b0 (it applies its activation and its remaining layers),
and B's argmax logit is the answer. The bridge is scored by B's accuracy on unseen digit images (B alone gets about
95-97%; a random bridge gets about 10%).

Design the bridge. Save it as bridge.npz in this folder with arrays M (shape hA x hB) and c (shape hB).

RULES
- Python with numpy and scipy is at: PYTHON_PATH (use it to run your code). Nothing else may be installed.
- You have NO data: do not load, download or reconstruct any dataset of digit images (no sklearn, no internet, no files
  outside this folder). You may construct synthetic inputs yourself from first principles if you find that useful,
  but not by copying or recalling real dataset samples.
- Work only inside this folder. When bridge.npz is saved, reply with a short description of your method.
