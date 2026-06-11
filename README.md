# adze

> A small, readable deep learning framework built from scratch — reverse-mode autograd, a PyTorch-like `nn` library, and a CPU/CUDA backend.

`adze` is a minimal, PyTorch-like deep learning framework written from scratch: a reverse-mode autograd engine, an `nn`/optimizer library, and a CPU/CUDA backend, small enough to read end to end. It exists to understand how modern frameworks work under the hood — not to compete with them.

*(An adze is a hand tool for carving a shape out of a rough block. It also has "AD" — automatic differentiation — hiding in the name.)*

## Status

🚧 In active development. Built roughly in the order of CMU's *Deep Learning Systems* course (10-714 / dlsyscourse), with [d2l.ai](https://d2l.ai) as a reference.

**Phase 1 — core framework** (in progress)
- [ ] Tensor / NDArray with strides, views, broadcasting
- [ ] Reverse-mode autograd engine (computational graph, topological backward, gradient accumulation)
- [ ] `nn` library — modules, parameters, initialization
- [ ] Optimizers — SGD, Adam
- [ ] CPU backend (NumPy-backed)
- [ ] CUDA backend (strided NDArray, hand-written kernels)
- [ ] End-to-end: train an MLP → CNN → small Transformer

**Phase 2 — performance layer** (planned, after Phase 1)
- [ ] Triton fused kernels (fused softmax / layernorm / matmul+bias+activation)
- [ ] A minimal graph-level fusion pass (the "interpreter → compiler" step)
- [ ] FP8 / MX low-precision GEMM on Blackwell, benchmarked against fp16

## Design

A tensor is `(data, shape, strides, offset)`; reshape / permute / broadcast are views (stride tricks), and broadcasting's backward is summation. Autograd is define-by-run: each op carries a `compute()` (forward) and a `gradient()` (vector–Jacobian product), and `backward()` walks the graph in reverse topological order.

Hand-written, on purpose: the autograd engine, the VJPs of the core ops, the strided NDArray semantics, and a tiled / shared-memory GEMM on the GPU. Library calls (cuBLAS, etc.) are used only after a teaching implementation exists.

## Usage

```python
import adze
from adze import Tensor
import adze.nn as nn

x = Tensor.randn(32, 784)
w = Tensor.randn(784, 128, requires_grad=True)
loss = (x @ w).relu().sum()
loss.backward()
print(w.grad.shape)  # (784, 128)

model = nn.Sequential(
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10),
)
```

## Testing

Correctness rests on two oracles, not a fixed test set:

- **Finite differences** — every op's analytic gradient (VJP) is checked against a numerical gradient `(f(x+ε) − f(x−ε)) / 2ε`.
- **Differential testing vs PyTorch** — the same inputs run through `adze` and `torch`; forward outputs and gradients must agree within tolerance.

Layered as: per-op (gradient checks + structural tests, including the shared-node "diamond" gradient-accumulation case) → kernel layer (GPU vs CPU vs torch; CUDA-event timing) → end-to-end convergence (a small model trains to a loss curve matching PyTorch).

## Install

```bash
git clone https://github.com/<you>/adze
cd adze
pip install -e .
```

## Non-goals

Not a PyTorch competitor. On raw speed, op coverage, and hardware support, the big frameworks win — that's the wrong game. `adze` optimizes for being small enough to fully understand and hack, and for serving as the substrate for the kernel work in Phase 2.

## Acknowledgements

Structured after CMU 10-714 *Deep Learning Systems* (dlsyscourse), with [d2l.ai](https://d2l.ai) as a reference. Course assignment solutions are kept private; the public work here is the framework design and the Phase-2 kernel layer.

## License

MIT