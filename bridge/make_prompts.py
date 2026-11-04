"""Write the tool-free prompt for each pair (the only thing the LLM sees: the task and both models' weights)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

TASK = """TASK. Two small neural networks, A and B, were trained separately (different random initialisations, and possibly
different widths and activations) on the same task: classifying 8 x 8 grayscale images of handwritten digits into the
10 classes 0-9. Input x has 64 numbers: x[r*8 + c] is the pixel in row r (0 = top) and column c (0 = left), scaled to
[0, 1] (0 = background, 1 = full ink). Logit k is the score of digit k.

We want A and B to communicate without any text: A reads an image, computes its first hidden layer h1A, and sends that
vector to B. B never sees the image. A "bridge" (M, c) turns A's message into the PRE-activation of B's first layer:
    z1B_hat = h1A @ M + c          (M has {hA} rows and {hB} columns, c has {hB} entries)
then B continues exactly as if z1B_hat were its own x @ W1 + b1: it applies its activation and its remaining layers,
and B's argmax logit is the answer. The bridge is scored by B's accuracy on unseen digit images (B alone gets about
95-97%; a random bridge gets about 10%).

Design the bridge from the weights alone. You have no data, no tools, no code execution and no calculator: this is a
single response. Reason about what the weights show, then give the bridge.

Answer in this order:
Reasoning: <how A's and B's first-layer units relate, and how you built M and c>
Bridge: one ```json block, exactly {{"M": [[...{hB} numbers...], ... {hA} rows ...], "c": [...{hB} numbers...]}}

THE WEIGHTS
{weights}
"""


def main():
    for name in C.PAIRS:
        A, B = C.load(f"{OUT}/{name}_A.npz"), C.load(f"{OUT}/{name}_B.npz")
        hA, hB = A["W"][0].shape[1], B["W"][0].shape[1]
        w = C.describe(A, "A") + "\n\n" + C.describe(B, "B")
        txt = TASK.format(hA=hA, hB=hB, weights=w)
        open(f"{OUT}/prompt_{name}.txt", "w").write(txt)
        print(name, hA, hB, len(txt), "chars")


if __name__ == "__main__":
    main()
