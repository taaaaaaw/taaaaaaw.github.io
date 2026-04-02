---
title: "Activation Functions"
date: 2026-04-02
description: "A detailed explanation of activation functions in deep learning — what they are, why they are needed, and the most common types including sigmoid, tanh, ReLU, and more."
categories:
    - Deep-Learning
tags:
    - deep-learning
    - activation-functions
    - neural-network
draft: false
---

## What Is an Activation Function?

An activation function is applied to the output of each neuron in a neural network. Without it, a neural network — no matter how many layers it has — would just be a linear model.

Each neuron computes:

$$z = \vec{w} \cdot \vec{x} + b$$

$$a = g(z)$$

Where $g$ is the **activation function**. The output $a$ (called the **activation**) is passed to the next layer.

---

## Why Do We Need Activation Functions?

Without an activation function, stacking multiple layers collapses into a single linear transformation:

$$\text{Layer 1: } a^{[1]} = W^{[1]}x + b^{[1]}$$

$$\text{Layer 2: } a^{[2]} = W^{[2]}a^{[1]} + b^{[2]} = W^{[2]}(W^{[1]}x + b^{[1]}) + b^{[2]}$$

This simplifies to just $Ax + B$ — a single linear function. No matter how many layers, the network can only learn linear relationships.

Activation functions introduce **non-linearity**, allowing the network to learn complex patterns like curves, boundaries, and hierarchical features.

---

## Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Output range:** $(0, 1)$

```
σ(z)
 1 |          ___________
   |        /
0.5|------/-------------- z = 0
   |    /
 0 |___/
   +--------------------- z
```

### Properties

| Property | Value |
|----------|-------|
| Output range | (0, 1) |
| Derivative | $\sigma(z)(1 - \sigma(z))$ |
| Max derivative | 0.25 (at z = 0) |
| Saturates at | z → ±∞ |

### When to use

- **Output layer** of binary classification — output is a probability between 0 and 1
- Rarely used in hidden layers anymore

### Problems

- **Vanishing gradient:** derivative is always ≤ 0.25, so gradients shrink as they propagate back through many layers
- **Not zero-centred:** outputs are always positive, which can slow down gradient descent

---

## Tanh (Hyperbolic Tangent)

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

**Output range:** $(-1, 1)$

```
tanh(z)
 1 |          ___________
   |        /
 0 |------/-------------- z = 0
   |    /
-1 |___/
   +--------------------- z
```

### Properties

| Property | Value |
|----------|-------|
| Output range | (-1, 1) |
| Derivative | $1 - \tanh^2(z)$ |
| Max derivative | 1 (at z = 0) |
| Zero-centred | Yes |

### Relationship to sigmoid

$$\tanh(z) = 2\sigma(2z) - 1$$

Tanh is a scaled and shifted version of sigmoid.

### When to use

- **Hidden layers** — almost always better than sigmoid because it is zero-centred
- Output layer when you need outputs in $(-1, 1)$

### Problems

- Still suffers from **vanishing gradients** for very large or very small $z$
- Computationally slightly more expensive than ReLU

---

## ReLU (Rectified Linear Unit)

$$\text{ReLU}(z) = \max(0, z)$$

**Output range:** $[0, \infty)$

```
ReLU(z)
  |          /
  |        /
  |      /
  |    /
  0___/__________________ z
  0
```

### Properties

| Property | Value |
|----------|-------|
| Output range | [0, ∞) |
| Derivative | 1 if z > 0, else 0 |
| Computationally | Very fast |
| Zero-centred | No |

### Why ReLU is preferred

- **No vanishing gradient** for positive inputs — derivative is always 1, gradients flow freely
- **Computationally cheap** — just a max operation, no exponentials
- **Sparse activation** — roughly half the neurons output 0, making the network efficient
- Converges much faster in practice

### When to use

- **Default choice for hidden layers** in most neural networks

### Problems

- **Dying ReLU:** if a neuron's input is always negative, its gradient is always 0 — the neuron never updates and "dies"
- Not zero-centred

---

## Leaky ReLU

$$\text{Leaky ReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ 0.01z & \text{if } z \leq 0 \end{cases}$$

**Output range:** $(-\infty, \infty)$

```
LeakyReLU(z)
  |          /
  |        /
  |      /
  0    /_________________ z
   \  /
    \/  (slight negative slope)
```

### Why it exists

Solves the dying ReLU problem by allowing a small gradient (0.01) for negative inputs. The neuron can still learn even when $z < 0$.

### Parametric ReLU (PReLU)

The slope for negative inputs becomes a **learnable parameter** $\alpha$:

$$\text{PReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \leq 0 \end{cases}$$

---

## Softmax

Used in the **output layer for multi-class classification** (more than 2 classes).

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

For $K$ classes, softmax converts a vector of raw scores into a **probability distribution** that sums to 1.

### Example

Raw output scores: $z = [2.0, 1.0, 0.1]$

$$\text{softmax}(z) = \left[\frac{e^{2}}{e^{2}+e^{1}+e^{0.1}}, \frac{e^{1}}{e^{2}+e^{1}+e^{0.1}}, \frac{e^{0.1}}{e^{2}+e^{1}+e^{0.1}}\right] \approx [0.659, 0.242, 0.099]$$

All values are positive and sum to 1. The model predicts class 0 with 65.9% confidence.

### Relationship to sigmoid

For binary classification ($K = 2$), softmax reduces to sigmoid.

---

## Linear (No Activation)

$$g(z) = z$$

Used only in the **output layer for regression** tasks where you want to predict a continuous value with no range restriction.

---

## Choosing the Right Activation Function

| Layer | Task | Recommended |
|-------|------|-------------|
| Hidden layers | Any | **ReLU** (default) |
| Hidden layers | If dying ReLU is a problem | Leaky ReLU |
| Output layer | Binary classification | **Sigmoid** |
| Output layer | Multi-class classification | **Softmax** |
| Output layer | Regression | **Linear** (none) |
| Output layer | Output in (-1, 1) | **Tanh** |

---

## Comparison Summary

| Function | Range | Vanishing Gradient | Zero-centred | Speed |
|----------|-------|--------------------|--------------|-------|
| Sigmoid | (0, 1) | Yes | No | Slow |
| Tanh | (-1, 1) | Yes | Yes | Slow |
| ReLU | [0, ∞) | No (for z > 0) | No | Fast |
| Leaky ReLU | (-∞, ∞) | No | No | Fast |
| Softmax | (0, 1) per class | — | No | Medium |

---

## Key Terms

| Term | Meaning |
|------|---------|
| **Activation function** | Non-linear function applied to each neuron's output |
| **Activation** ($a$) | The output of a neuron after applying the activation function |
| **Non-linearity** | What allows neural networks to learn complex patterns |
| **Vanishing gradient** | Gradients become near-zero in deep networks; early layers stop learning |
| **Dying ReLU** | Neurons stuck at 0 output because their inputs are always negative |
| **Saturation** | When the activation function's output is at its extreme (e.g. ~0 or ~1 for sigmoid), gradient ≈ 0 |
| **Softmax** | Converts raw scores to a probability distribution for multi-class output |
