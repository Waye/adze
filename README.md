# adze

> A small, readable deep learning framework built from scratch — from tensors and reverse-mode autodiff to neural networks, CPU/CUDA execution, CNNs, sequence models, and Transformers.

`adze` is a minimal, PyTorch-like deep learning framework built from first principles.

The goal is not to compete with PyTorch. It is to understand how modern deep learning systems work by implementing the important pieces in a form small enough to read, test, profile, and modify end to end.

The project follows the learning path of [CMU 10-414/714: Deep Learning Systems](https://dlsyscourse.org/), while keeping each implementation deliberately compact.

Rather than implementing only the minimum needed for one model, `adze` aims to cover the foundations of a modern deep learning framework:

* tensor storage, strides, views, and broadcasting,
* reverse-mode automatic differentiation,
* neural-network abstractions and optimizers,
* normalization and regularization,
* CPU and CUDA execution,
* convolutional networks,
* recurrent sequence models,
* Transformers and autoregressive modeling,
* basic large-model training techniques,
* serialization, fine-tuning, and inference.

Two small end-to-end workloads serve as the primary validation targets:

* **Small ResNet / CIFAR-10** — general deep learning and computer vision.
* **TinyGPT / TinyStories** — Transformers and autoregressive language modeling.

The same tensor, autograd, `nn`, optimizer, and backend abstractions should support both.

*(An adze is a hand tool used to carve a shape out of a rough block. The name also hides “AD” — automatic differentiation.)*

---

## Why build another framework?

Production frameworks already solve deep learning extremely well.

`adze` exists for a different reason:

```text
use a framework
      ↓
understand a framework
      ↓
build a framework
      ↓
understand what happens underneath the model
```

A training loop such as

```python
loss = model(x)
loss.backward()
optimizer.step()
```

hides a large system underneath:

```text
Tensor
  ↓
operators
  ↓
computation graph
  ↓
reverse-mode autodiff
  ↓
module / parameter system
  ↓
optimizer
  ↓
device dispatch
  ↓
CPU / CUDA kernels
```

`adze` makes those layers explicit.

---

# Architecture

```text
                           adze
                            │
                     Model workloads
            ┌───────────────┼───────────────┐
            │               │               │
           CNN         Sequence models   Transformer
            │               │               │
      Small ResNet       RNN / LSTM       TinyGPT
       CIFAR-10                           TinyStories
            │               │               │
            └───────────────┴───────────────┘
                            │
                         adze.nn
                            │
                        adze.optim
                            │
                         Autograd
                            │
                          Tensor
                            │
                    Operator dispatch
                            │
                   ┌────────┴────────┐
                  CPU              CUDA
```

The model implementations should contain no special framework logic.

They are workloads built on top of the same reusable infrastructure.

---

# Scope

The project follows a simple rule:

> **Cover the important ideas broadly, but implement each one as simply as possible.**

The goal is therefore not:

```text
few features
+
large models
```

but:

```text
many fundamental concepts
+
small understandable implementations
```

This keeps `adze` useful both as a framework project and as preparation for deeper work on language models.

---

# Roadmap

## M0 — ML fundamentals

Before automatic differentiation, implement several pieces manually.

* [ ] Softmax regression
* [ ] Cross-entropy loss
* [ ] Manual SGD
* [ ] Two-layer neural network
* [ ] Manual backpropagation
* [ ] Numerical gradient checking

The goal is to understand what the autograd engine will eventually automate.

---

# M1 — Tensor / NDArray

Build the fundamental data structure of the framework.

Conceptually:

```text
Tensor
│
├── storage
├── shape
├── strides
├── offset
├── dtype
└── device
```

Implement:

* [ ] storage
* [ ] shape
* [ ] strides
* [ ] offsets
* [ ] indexing
* [ ] slicing
* [ ] reshape
* [ ] transpose / permute
* [ ] broadcasting
* [ ] reductions
* [ ] contiguous tensors
* [ ] non-contiguous tensors
* [ ] CPU / CUDA device metadata

Views should be preferred over copies where possible.

For example:

```python
x = Tensor.randn(4, 8)

y = x.transpose(0, 1)
z = x.reshape(2, 16)
```

should primarily manipulate tensor metadata rather than copy the underlying data.

---

# M2 — Reverse-mode automatic differentiation

Build a define-by-run computation graph.

```text
x ───┐
     op ─── op ─── loss
w ───┘               │
                     │
                  backward
                     │
                     ▼
                 gradients
```

Each differentiable operator defines:

```text
forward computation
        +
vector-Jacobian product
```

Implement:

* [ ] computation graph construction
* [ ] reverse topological traversal
* [ ] vector-Jacobian products
* [ ] gradient accumulation
* [ ] shared subgraphs
* [ ] broadcasting backward
* [ ] `requires_grad`
* [ ] detach / no-grad behavior
* [ ] `Tensor.backward()`

Example target API:

```python
x = Tensor.randn(32, 128)

w = Tensor.randn(
    128,
    64,
    requires_grad=True,
)

y = (x @ w).relu()

loss = y.mean()

loss.backward()

print(w.grad)
```

---

# M3 — Neural-network abstractions

Turn the tensor and autodiff engine into a usable neural-network library.

## Module system

* [ ] `Module`
* [ ] `Parameter`
* [ ] recursive parameter discovery
* [ ] `train()`
* [ ] `eval()`
* [ ] initialization utilities

## Layers

* [ ] `Linear`
* [ ] `Sequential`
* [ ] `Flatten`
* [ ] `ReLU`
* [ ] `GELU`
* [ ] `Dropout`
* [ ] `BatchNorm`
* [ ] `LayerNorm`
* [ ] `Embedding`

## Loss functions

* [ ] MSE
* [ ] cross entropy

## Optimizers

* [ ] SGD
* [ ] Momentum
* [ ] Adam
* [ ] AdamW

Example target API:

```python
import adze.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
```

---

# M4 — CPU backend

Separate tensor semantics from execution.

```text
Python API
    │
    ▼
 Operator
    │
    ▼
 Dispatch
    │
    ▼
 CPU kernel
```

Implement representative CPU operations:

* [ ] elementwise operations
* [ ] broadcasting
* [ ] reductions
* [ ] matrix multiplication
* [ ] convolution

A simple reference implementation may use NumPy where useful, while selected operations can later be implemented explicitly for comparison.

---

# M5 — CUDA backend

Add GPU execution without changing the user-facing tensor API.

```text
                   Operator
                      │
                   Dispatch
                ┌─────┴─────┐
                │           │
               CPU         CUDA
                │           │
             kernels      kernels
```

Implement:

* [ ] device memory
* [ ] host ↔ device transfer
* [ ] elementwise kernels
* [ ] reduction kernels
* [ ] strided tensor access
* [ ] naive GEMM
* [ ] tiled/shared-memory GEMM
* [ ] convolution

The purpose is to understand:

* thread and block organization,
* memory coalescing,
* shared memory,
* synchronization,
* tiling,
* arithmetic intensity,
* and the effect of tensor layout on performance.

---

# M6 — Convolutional networks

Add the main primitives required by computer vision.

* [ ] `Conv2d`
* [ ] padding
* [ ] stride
* [ ] pooling
* [ ] normalization
* [ ] residual connections

## Validation: Small ResNet / CIFAR-10

```text
Image
  │
Conv2d
  │
Normalization
  │
Residual blocks
  │
Pooling
  │
Linear
  │
Cross entropy
```

The model is intentionally small.

The purpose is not to maximize CIFAR-10 accuracy. It is to verify that a non-trivial convolution-heavy model can be trained entirely through `adze`.

Eventually, increasing the network to ResNet-18 should require changing the model definition rather than the framework itself.

---

# M7 — Sequence modeling

Implement recurrent sequence models before moving to Transformers.

## RNN

* [ ] `RNNCell`
* [ ] multi-step recurrence
* [ ] hidden state
* [ ] batching
* [ ] backpropagation through time

## LSTM

* [ ] `LSTMCell`
* [ ] cell state
* [ ] gating

A small character-level language-modeling example is sufficient.

The goal is to understand recurrent computation graphs and why Transformer-style architectures later became attractive.

---

# M8 — Transformers

Implement the Transformer primitives directly using `adze`.

```text
tokens
  │
Embedding
  │
Transformer block
  │
  ├── LayerNorm
  ├── causal self-attention
  └── MLP
  │
LM head
```

Implement:

* [ ] token embeddings
* [ ] positional information
* [ ] Q / K / V projections
* [ ] scaled dot-product attention
* [ ] causal masking
* [ ] softmax
* [ ] multi-head attention
* [ ] LayerNorm
* [ ] GELU
* [ ] residual connections
* [ ] Transformer blocks

---

# M9 — TinyGPT / TinyStories

Use the Transformer implementation to build a small autoregressive language model.

```text
TinyStories
     │
  tokenizer
     │
   tokens
     │
  TinyGPT
     │
 next-token
 prediction
```

Example target model:

```python
model = TinyGPT(
    vocab_size=vocab_size,
    dim=256,
    layers=4,
    heads=4,
)
```

Training should use ordinary framework APIs:

```python
logits = model(tokens)

loss = cross_entropy(
    logits,
    targets,
)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Implement:

* [ ] causal language-model loss
* [ ] training loop
* [ ] checkpointing
* [ ] autoregressive generation
* [ ] temperature sampling
* [ ] top-k sampling

This workload is the bridge between `adze` and future language-model work.

---

# M10 — Training systems

Cover the basic ideas used when models become too large for naïve training.

Small experiments are enough.

* [ ] gradient accumulation
* [ ] mixed-precision training
* [ ] loss scaling
* [ ] activation-memory measurement
* [ ] gradient checkpointing

Study and, where practical, demonstrate small versions of:

* data parallelism,
* tensor parallelism,
* pipeline parallelism,
* optimizer-state memory,
* parameter / activation / gradient memory costs.

The goal is understanding rather than building a distributed production runtime.

---

# M11 — Generative models

Use the same framework for one additional generative workload.

Possible minimal implementations:

* [ ] VAE

or

* [ ] GAN

Only one is necessary.

This stage exists primarily to verify that the framework abstractions generalize beyond classifiers and autoregressive language models.

---

# M12 — Model customization

Add basic model lifecycle functionality.

* [ ] `state_dict()`
* [ ] save parameters
* [ ] load parameters
* [ ] freeze parameters
* [ ] replace model heads
* [ ] fine-tune pretrained parameters

Example:

```python
model.load_state_dict(weights)

for p in model.backbone.parameters():
    p.requires_grad = False
```

This provides the foundation for later work such as fine-tuning and parameter-efficient adaptation.

---

# M13 — Inference and deployment basics

Keep deployment deliberately lightweight.

Implement:

* [ ] serialization
* [ ] model loading
* [ ] evaluation mode
* [ ] inference-only execution
* [ ] simple command-line inference

For example:

```bash
python train.py

python generate.py \
    --checkpoint checkpoints/tinygpt.adze
```

The goal is to understand the transition:

```text
training
   ↓
checkpoint
   ↓
load
   ↓
eval
   ↓
inference
```

rather than building a production serving platform.

---

# Testing

Framework correctness is treated as a first-class feature.

## Numerical gradients

Compare analytical gradients against finite differences:

```text
           f(x + ε) - f(x - ε)
grad ≈     ───────────────────
                    2ε
```

---

## Differential testing against PyTorch

Use the same inputs for both frameworks.

```text
       same input
      /          \
   adze         PyTorch
     │             │
  forward       forward
     │             │
 backward      backward
      \           /
        compare
```

Compare:

* forward values,
* gradients,
* parameter updates,
* selected training curves.

---

## Autograd structure tests

Include graphs that test:

* branching,
* shared nodes,
* repeated tensor use,
* gradient accumulation,
* broadcasting,
* non-scalar intermediate tensors.

For example:

```text
       x
      / \
     a   b
      \ /
       y
```

must correctly accumulate both gradient paths into `x`.

---

## Backend parity

Where an operator has multiple implementations:

```text
CPU
 │
 ├──── compare ──── CUDA
 │
 └──── compare ──── PyTorch
```

---

# Benchmarks

Correctness comes first, but performance should be measured rather than guessed.

## Operator benchmarks

* [ ] elementwise operations
* [ ] reductions
* [ ] matrix multiplication
* [ ] Conv2d
* [ ] softmax
* [ ] LayerNorm

## End-to-end benchmarks

* [ ] Small ResNet training throughput
* [ ] TinyGPT tokens / second
* [ ] CPU vs CUDA
* [ ] selected comparisons with PyTorch eager mode

The goal is not:

> make adze faster than PyTorch.

The useful question is:

> where is adze slower, and what system behavior explains the difference?

---

# Planned repository structure

```text
adze/
│
├── adze/
│   ├── tensor.py
│   ├── autograd.py
│   │
│   ├── ops/
│   │   ├── elementwise.py
│   │   ├── reduction.py
│   │   ├── matmul.py
│   │   └── conv.py
│   │
│   ├── nn/
│   │   ├── module.py
│   │   ├── linear.py
│   │   ├── normalization.py
│   │   ├── convolution.py
│   │   ├── recurrent.py
│   │   └── attention.py
│   │
│   ├── optim/
│   │   ├── sgd.py
│   │   └── adam.py
│   │
│   └── backend/
│       ├── cpu/
│       └── cuda/
│
├── examples/
│   ├── softmax/
│   ├── manual_backprop/
│   ├── resnet_cifar10/
│   ├── char_rnn/
│   ├── tinygpt_tinystories/
│   └── generative/
│
├── benchmarks/
│
└── tests/
```

---

# Learning path

Development roughly follows the progression of CMU 10-414/714 *Deep Learning Systems*:

```text
ML fundamentals
       │
manual backprop
       │
automatic differentiation
       │
optimization
       │
NN abstractions
       │
normalization / regularization
       │
tensor runtime
       │
CPU / CUDA
       │
convolution
       │
sequence models
       │
Transformers
       │
large-model training concepts
       │
generative models
       │
model customization
       │
inference / deployment
```

The implementations are independently structured for `adze`; the public repository is not intended to distribute course assignment solutions.

---

# From adze to smallm

`adze` focuses on the **deep learning system underneath a model**.

A future project, `smallm`, can then focus on the **language model itself**.

```text
               adze
                 │
        deep learning systems
                 │
     ┌───────────┼───────────┐
     │           │           │
  autograd    runtime      CUDA
     │           │           │
     └───────────┼───────────┘
                 │
           Transformer
                 │
              TinyGPT
                 │
                 ▼
               smallm
                 │
       language-model systems
                 │
      architecture / training
      inference / adaptation
      reasoning experiments
```

The intended progression is:

> first understand the framework that trains the model, then build the model on top of that understanding.

---

# Design principles

## Breadth with small implementations

Cover the important concepts without turning every component into a production system.

## Framework first

CNNs, RNNs, and Transformers must all use ordinary `adze` APIs.

Model-specific behavior should not leak into the tensor runtime or autograd engine.

## Correctness before performance

```text
correct
   ↓
tested
   ↓
profiled
   ↓
optimized
```

## Views before copies

Tensor layout and stride semantics are part of the framework, not an implementation detail to hide.

## Understand important mechanisms explicitly

Core educational components should be implemented directly before introducing optimized shortcuts.

These include:

* reverse-mode autodiff,
* VJPs,
* broadcasting gradients,
* tensor views and strides,
* module / parameter discovery,
* optimizer updates,
* operator dispatch,
* representative CPU/CUDA kernels,
* convolution,
* recurrence,
* attention.

---

# Later systems experiments

Once the complete learning path is functional, `adze` can also serve as a sandbox for more advanced systems work:

* fused kernels,
* Triton kernels,
* operator fusion,
* graph-level optimization,
* kernel profiling,
* mixed-precision execution,
* low-precision GEMM,
* memory planning,
* alternative attention implementations.

These are intentionally downstream of the core framework.

---

# Non-goals

`adze` is not intended to replace PyTorch, JAX, or TensorFlow.

It does not prioritize:

* broad production operator coverage,
* production serving,
* large distributed clusters,
* every hardware backend,
* ecosystem compatibility,
* state-of-the-art training throughput.

Its purpose is to make the foundations of modern deep learning systems understandable from end to end.

---

# References

* [CMU 10-414/714 — Deep Learning Systems](https://dlsyscourse.org/)
* [Deep Learning Systems Lectures](https://dlsyscourse.org/lectures/)
* [Dive into Deep Learning](https://d2l.ai/)
* PyTorch documentation and source code, used as behavioral references where appropriate.

---

## License

MIT
