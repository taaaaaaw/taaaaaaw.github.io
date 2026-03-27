---
title: "Computation Graph"
date: 2026-03-27
description: "A detailed explanation of computation graphs — how they represent neural network calculations, and how they make backpropagation efficient."
categories:
    - Deep-Learning
tags:
    - deep-learning
    - backpropagation
    - computation-graph
    - derivatives
draft: false
---

## What Is a Computation Graph?

A computation graph is a way to **visually represent a mathematical expression** as a series of simple steps. Each node in the graph represents one operation, and the edges show how values flow between operations.

Deep learning frameworks like TensorFlow and PyTorch build computation graphs internally — this is what enables automatic differentiation (autograd) and makes backpropagation efficient.

---

## A Simple Example

Let's say we have:

$$J = 3(a + bc)$$

Where $a = 5$, $b = 3$, $c = 2$.

We break this into small steps:

```
Step 1: u = bc      = 3 × 2 = 6
Step 2: v = a + u   = 5 + 6 = 11
Step 3: J = 3v      = 3 × 11 = 33
```

As a graph:

```
b=3 ──┐
       ×──► u=6 ──┐
c=2 ──┘            +──► v=11 ──┐
                                 ×3 ──► J=33
a=5 ───────────────┘
```

Each arrow carries a value forward — this is the **forward pass**.

---

## Forward Pass

The forward pass computes the output by moving **left to right** through the graph, calculating each node's value from its inputs.

For a neural network:

```
Input x
    ↓
z = wx + b          ← linear step
    ↓
a = σ(z)            ← activation
    ↓
J = Loss(a, y)      ← cost
```

Every intermediate value ($z$, $a$) is stored during the forward pass — they will be needed in the backward pass.

---

## Backward Pass (Backpropagation)

The backward pass computes gradients by moving **right to left** through the graph, applying the chain rule at each node.

Going back to our example $J = 3(a + bc)$:

### Step 1: $\frac{\partial J}{\partial J} = 1$

Always start with 1 — J's rate of change with respect to itself.

### Step 2: $\frac{\partial J}{\partial v}$

$$J = 3v \quad \Rightarrow \quad \frac{\partial J}{\partial v} = 3$$

### Step 3: $\frac{\partial J}{\partial u}$ and $\frac{\partial J}{\partial a}$

$$v = a + u \quad \Rightarrow \quad \frac{\partial v}{\partial u} = 1, \quad \frac{\partial v}{\partial a} = 1$$

By chain rule:

$$\frac{\partial J}{\partial u} = \frac{\partial J}{\partial v} \cdot \frac{\partial v}{\partial u} = 3 \cdot 1 = 3$$

$$\frac{\partial J}{\partial a} = \frac{\partial J}{\partial v} \cdot \frac{\partial v}{\partial a} = 3 \cdot 1 = 3$$

### Step 4: $\frac{\partial J}{\partial b}$ and $\frac{\partial J}{\partial c}$

$$u = bc \quad \Rightarrow \quad \frac{\partial u}{\partial b} = c = 2, \quad \frac{\partial u}{\partial c} = b = 3$$

By chain rule:

$$\frac{\partial J}{\partial b} = \frac{\partial J}{\partial u} \cdot \frac{\partial u}{\partial b} = 3 \cdot 2 = 6$$

$$\frac{\partial J}{\partial c} = \frac{\partial J}{\partial u} \cdot \frac{\partial u}{\partial c} = 3 \cdot 3 = 9$$

### Full Graph with Gradients

```
                    forward →
                    ← backward

b=3 ──┐  ∂J/∂b=6
       ×──► u=6 ──┐  ∂J/∂u=3
c=2 ──┘  ∂J/∂c=9   +──► v=11 ──┐  ∂J/∂v=3
                                   ×3 ──► J=33
a=5 ───────────────┘  ∂J/∂a=3
```

---

## Computation Graph for Logistic Regression

Let's trace through a single training example of logistic regression.

**Given:** input $x$, weight $w$, bias $b$, true label $y$

**Forward pass:**

```
x, w ──► z = wx + b ──► a = σ(z) ──► L = -[y·log(a) + (1-y)·log(1-a)]
```

**Values:**

| Node | Formula | Example value |
|------|---------|---------------|
| $z$ | $wx + b$ | $wx + b$ |
| $a$ | $\sigma(z)$ | between 0 and 1 |
| $L$ | $-[y \log a + (1-y)\log(1-a)]$ | scalar loss |

**Backward pass (derivatives we need):**

$$\frac{\partial L}{\partial a} = -\frac{y}{a} + \frac{1-y}{1-a}$$

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial z} = a - y$$

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w} = (a - y) \cdot x$$

$$\frac{\partial L}{\partial b} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial b} = a - y$$

The result $(a - y)$ appears in every gradient — this is the **prediction error**, and it flows back through the entire graph.

---

## Computation Graph for a Neural Network Layer

For a single hidden layer with $n$ neurons:

**Forward pass:**

```
X ──► Z = WX + b ──► A = g(Z) ──► Z² = W²A + b² ──► A² ──► J
```

**Backward pass (one layer):**

$$dZ = \frac{\partial J}{\partial Z}$$

$$dW = \frac{1}{m} dZ \cdot X^T$$

$$db = \frac{1}{m} \sum dZ$$

$$dA_{prev} = W^T \cdot dZ$$

Each layer receives a gradient from the layer above ($dA$), computes its own gradients ($dW$, $db$), and passes a gradient to the layer below ($dA_{prev}$).

This is how gradients flow from the output all the way back to the first layer.

---

## Why Computation Graphs Are Efficient

### Without a computation graph

To compute $\frac{\partial J}{\partial w}$ directly, you would need to derive one massive formula analytically. For a deep network with millions of parameters, this is completely impractical.

### With a computation graph

Break the computation into small local operations. Each node only needs to know:
1. Its own local derivative (how its output changes with respect to its inputs)
2. The gradient flowing in from the right ($\frac{\partial J}{\partial \text{output}}$)

Then it computes the gradient flowing left:

$$\frac{\partial J}{\partial \text{input}} = \frac{\partial J}{\partial \text{output}} \cdot \frac{\partial \text{output}}{\partial \text{input}}$$

This is just one multiplication per node. For a network with $L$ layers and $n$ parameters per layer, backpropagation runs in $O(Ln)$ — linear in the number of parameters.

### Reusing intermediate values

The forward pass stores intermediate values ($z$, $a$ for each layer). The backward pass reuses these — no redundant computation.

---

## How PyTorch and TensorFlow Use Computation Graphs

Modern deep learning frameworks build the computation graph **automatically** as you write code:

```python
# PyTorch example
import torch

x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

z = w * x + b       # PyTorch records this operation
a = torch.sigmoid(z)  # and this
J = -torch.log(a)   # and this → computation graph built

J.backward()        # backpropagation: traverse graph right to left

print(w.grad)       # ∂J/∂w — computed automatically
print(b.grad)       # ∂J/∂b — computed automatically
```

You never manually compute derivatives — the framework traces every operation, builds the graph, and runs backpropagation for you.

---

## Static vs Dynamic Graphs

| | Static Graph (TensorFlow 1.x) | Dynamic Graph (PyTorch, TF 2.x) |
|---|---|---|
| When built | Before execution (define then run) | During execution (define by run) |
| Debugging | Harder | Easier (standard Python debugging) |
| Performance | Slightly faster (can optimise ahead of time) | Slightly slower |
| Flexibility | Less flexible | More flexible (e.g. variable-length inputs) |

Modern frameworks (PyTorch, TensorFlow 2.x) use **dynamic graphs** by default — the graph is built on the fly as your code runs.

---

## Key Terms

| Term | Meaning |
|------|---------|
| **Computation graph** | A graph where nodes are operations and edges are values flowing between them |
| **Forward pass** | Left-to-right traversal; computes output and stores intermediate values |
| **Backward pass** | Right-to-left traversal; computes gradients using the chain rule |
| **Local gradient** | The derivative of a node's output with respect to its own input |
| **Upstream gradient** | The gradient flowing in from the right ($\frac{\partial J}{\partial \text{output}}$) |
| **Backpropagation** | Efficient computation of all gradients by traversing the graph backwards |
| **Autograd** | Automatic differentiation — frameworks build and traverse the graph automatically |
| **Dynamic graph** | Graph built during execution; used by PyTorch and TensorFlow 2.x |
