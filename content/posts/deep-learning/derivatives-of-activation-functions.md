---
title: "Derivatives of Activation Functions"
date: 2026-04-02
description: "A detailed breakdown of how to differentiate common activation functions — sigmoid, tanh, ReLU, and Leaky ReLU — and why their derivatives matter for backpropagation."
categories:
    - Deep-Learning
tags:
    - deep-learning
    - activation-functions
    - derivatives
    - backpropagation
draft: false
---

## Why Do We Need Derivatives of Activation Functions?

During **backpropagation**, the chain rule requires the derivative of every function in the network — including activation functions. For each neuron:

$$\frac{\partial J}{\partial z} = \frac{\partial J}{\partial a} \cdot \frac{\partial a}{\partial z} = \frac{\partial J}{\partial a} \cdot g'(z)$$

Where $g'(z)$ is the derivative of the activation function. Without it, gradients cannot flow backwards through the network.

The properties of $g'(z)$ directly determine how well a network trains:
- If $g'(z)$ is very small → **vanishing gradient** → early layers stop learning
- If $g'(z)$ is always large → gradients flow freely → network trains efficiently

---

## Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

### Derivation

$$\sigma'(z) = \frac{d}{dz} \left(1 + e^{-z}\right)^{-1}$$

Using the chain rule:

$$= -\left(1 + e^{-z}\right)^{-2} \cdot (-e^{-z})$$

$$= \frac{e^{-z}}{(1 + e^{-z})^2}$$

Rewrite by splitting the fraction:

$$= \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}}$$

$$= \frac{1}{1 + e^{-z}} \cdot \left(1 - \frac{1}{1 + e^{-z}}\right)$$

$$\boxed{\sigma'(z) = \sigma(z)\left(1 - \sigma(z)\right)}$$

### Key insight

The derivative is expressed in terms of the output $a = \sigma(z)$ itself:

$$g'(z) = a(1 - a)$$

This is computationally convenient — during backpropagation, we already have $a$ stored from the forward pass.

### Values at key points

| $z$ | $\sigma(z)$ | $\sigma'(z)$ |
|-----|------------|-------------|
| -10 | ≈ 0 | ≈ 0 |
| -2 | 0.119 | 0.105 |
| 0 | 0.5 | **0.25** (maximum) |
| 2 | 0.881 | 0.105 |
| 10 | ≈ 1 | ≈ 0 |

### The vanishing gradient problem

The maximum value of $\sigma'(z)$ is **0.25**. In a deep network with $L$ layers, the gradient of the first layer involves:

$$\frac{\partial J}{\partial W^{[1]}} \propto \sigma'(z^{[L]}) \cdot \sigma'(z^{[L-1]}) \cdots \sigma'(z^{[1]})$$

Each term is at most 0.25, so with 10 layers:

$$0.25^{10} \approx 0.000001$$

The gradient essentially vanishes — early layers receive almost no learning signal.

---

## Tanh

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

### Derivation

Using the quotient rule, let $u = e^z - e^{-z}$ and $v = e^z + e^{-z}$:

$$\tanh'(z) = \frac{u'v - uv'}{v^2}$$

$$u' = e^z + e^{-z}, \quad v' = e^z - e^{-z}$$

$$\tanh'(z) = \frac{(e^z + e^{-z})^2 - (e^z - e^{-z})^2}{(e^z + e^{-z})^2}$$

$$= 1 - \frac{(e^z - e^{-z})^2}{(e^z + e^{-z})^2}$$

$$\boxed{\tanh'(z) = 1 - \tanh^2(z)}$$

### Key insight

Like sigmoid, the derivative is expressed in terms of the output $a = \tanh(z)$:

$$g'(z) = 1 - a^2$$

### Values at key points

| $z$ | $\tanh(z)$ | $\tanh'(z)$ |
|-----|-----------|------------|
| -10 | ≈ -1 | ≈ 0 |
| -2 | -0.964 | 0.071 |
| 0 | 0 | **1.0** (maximum) |
| 2 | 0.964 | 0.071 |
| 10 | ≈ 1 | ≈ 0 |

### Comparison with sigmoid

- Max derivative of tanh = **1.0** vs sigmoid's **0.25**
- Tanh gradients are 4× larger → less vanishing gradient
- Tanh is zero-centred → better gradient flow in practice
- But still saturates for large $|z|$ → vanishing gradient still occurs in very deep networks

---

## ReLU

$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

### Derivative

ReLU is not differentiable at exactly $z = 0$, but in practice we define:

$$\boxed{\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}}$$

At $z = 0$, we typically set the derivative to 0 or 0.5 by convention — it makes no difference in practice.

### Why this solves the vanishing gradient problem

For any positive $z$, the derivative is exactly **1**. Multiplying by 1 does not shrink the gradient:

$$\frac{\partial J}{\partial W^{[1]}} \propto 1 \cdot 1 \cdots 1 \cdot \frac{\partial J}{\partial a^{[L]}}$$

Gradients flow back unchanged through all layers where $z > 0$.

### The dying ReLU problem

For $z \leq 0$, the derivative is **0** — the gradient is completely blocked. If a neuron's $z$ is always negative (e.g. due to a large negative bias), it will never receive a gradient update. The neuron is permanently "dead."

```
Gradient flow:
z > 0  →  gradient passes through (×1)  ✅
z ≤ 0  →  gradient blocked (×0)         ❌ neuron dies
```

---

## Leaky ReLU

$$\text{Leaky ReLU}(z) = \begin{cases} z & \text{if } z > 0 \\ 0.01z & \text{if } z \leq 0 \end{cases}$$

### Derivative

$$\boxed{\text{Leaky ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0.01 & \text{if } z \leq 0 \end{cases}}$$

### Why this helps

The gradient for $z \leq 0$ is now **0.01** instead of 0. Neurons with negative inputs still receive a small gradient — they can still learn and never permanently die.

---

## Summary

| Function | Derivative formula | Max derivative | Vanishing gradient? | Dead neurons? |
|----------|--------------------|---------------|---------------------|---------------|
| Sigmoid | $\sigma(z)(1-\sigma(z))$ | 0.25 | Yes (severe) | No |
| Tanh | $1 - \tanh^2(z)$ | 1.0 | Yes (moderate) | No |
| ReLU | 0 or 1 | 1 | No | Yes (dying ReLU) |
| Leaky ReLU | 0.01 or 1 | 1 | No | No |

---

## How Derivatives Are Used in Backpropagation

For a single layer, the backward pass computes:

$$dZ^{[l]} = dA^{[l]} \ast g'^{[l]}(Z^{[l]})$$

Where $\ast$ is element-wise multiplication and $dA^{[l]}$ is the gradient flowing in from the layer above.

The derivative $g'(Z^{[l]})$ **gates** how much gradient passes through:
- If $g'$ is large → gradient passes freely → layer learns quickly
- If $g'$ is small → gradient is throttled → layer learns slowly or not at all

This is why the choice of activation function has such a large impact on how deep networks train.

---

## Key Terms

| Term | Meaning |
|------|---------|
| $g'(z)$ | Derivative of the activation function with respect to $z$ |
| **Vanishing gradient** | $g'(z)$ is very small, so gradients shrink to near zero as they propagate back |
| **Dying ReLU** | Neuron with $z \leq 0$ always has $g'(z) = 0$, so it never updates |
| **Saturation** | When $z$ is very large or small, $g'(z) \approx 0$ (sigmoid and tanh) |
| **Backpropagation** | Algorithm that uses $g'(z)$ at every layer to compute gradients |
