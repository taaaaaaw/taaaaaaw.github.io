---
title: "Building Blocks of Deep Neural Networks"
date: 2026-04-05
description: "The core building blocks of deep neural networks — layers, forward propagation, backward propagation, and how they work together during training."
categories:
    - Deep-Learning
tags:
    - neural-networks
    - forward-propagation
    - backpropagation
draft: false
---

## The Core Building Block: A Single Layer

Every deep neural network is built by stacking layers. Each layer $l$ has two fundamental operations:

- **Forward function** — takes input $a^{[l-1]}$, computes output $a^{[l]}$
- **Backward function** — takes gradient from the next layer, computes gradients to pass back and update parameters

These two functions are the building blocks that make training possible.

---

## Forward Propagation

In each layer $l$, forward propagation computes:

$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$

$$a^{[l]} = g^{[l]}(z^{[l]})$$

Where:
- $W^{[l]}$ = weight matrix for layer $l$
- $b^{[l]}$ = bias vector for layer $l$
- $g^{[l]}$ = activation function for layer $l$ (e.g. ReLU, sigmoid)
- $a^{[0]} = x$ = the input to the network

**What gets cached:**

During forward propagation, each layer caches $z^{[l]}$ and $a^{[l]}$. These cached values are needed later during backpropagation to compute gradients efficiently.

---

## Backward Propagation

Backward propagation computes three things for each layer $l$, given the incoming gradient $da^{[l]}$:

$$dz^{[l]} = da^{[l]} \cdot g^{[l]'}(z^{[l]})$$

$$dW^{[l]} = \frac{1}{m} dz^{[l]} \cdot a^{[l-1]T}$$

$$db^{[l]} = \frac{1}{m} \sum dz^{[l]}$$

$$da^{[l-1]} = W^{[l]T} \cdot dz^{[l]}$$

Where:
- $dz^{[l]}$ = gradient of the loss with respect to $z^{[l]}$
- $dW^{[l]}$, $db^{[l]}$ = gradients used to update parameters
- $da^{[l-1]}$ = gradient passed back to the previous layer

This is why the cache from forward propagation matters — $z^{[l]}$ is needed to compute $g^{[l]'}(z^{[l]})$, and $a^{[l-1]}$ is needed to compute $dW^{[l]}$.

---

## The Full Picture: One Training Step

For a network with $L$ layers, one training step looks like this:

```
Input x
   ↓
[Layer 1] → forward → cache z[1], a[1]
   ↓
[Layer 2] → forward → cache z[2], a[2]
   ↓
  ...
   ↓
[Layer L] → forward → cache z[L], a[L]
   ↓
Compute Loss: J = L(a[L], y)
   ↓
[Layer L] → backward → dW[L], db[L], da[L-1]
   ↓
  ...
   ↓
[Layer 2] → backward → dW[2], db[2], da[1]
   ↓
[Layer 1] → backward → dW[1], db[1]
   ↓
Update all W[l] and b[l] with gradient descent
```

Forward goes left to right, backward goes right to left. Each layer is a self-contained block with its own forward and backward function.

---

## Parameters vs Hyperparameters

Each layer has its own set of **parameters** that get updated during training:

| Parameter | Shape | Description |
|---|---|---|
| $W^{[l]}$ | $(n^{[l]}, n^{[l-1]})$ | Weight matrix |
| $b^{[l]}$ | $(n^{[l]}, 1)$ | Bias vector |

Where $n^{[l]}$ = number of units in layer $l$.

The **hyperparameters** you set before training:

| Hyperparameter | Description |
|---|---|
| $L$ | Number of layers |
| $n^{[l]}$ | Number of units per layer |
| $g^{[l]}$ | Choice of activation function per layer |
| $\alpha$ | Learning rate |
| Iterations | Number of gradient descent steps |

---

## Why This Modular Design Matters

Because each layer has a clean forward and backward function, you can:

- **Stack any number of layers** — the interface between layers is always $a^{[l]}$ and $da^{[l]}$
- **Mix activation functions** — use ReLU in hidden layers, sigmoid in the output layer
- **Debug layer by layer** — check the shapes and values at each cache point

This modular structure is exactly how deep learning frameworks like PyTorch and TensorFlow are built under the hood. Each layer is a module with a `forward()` and `backward()` method.
