---
title: "Derivatives and Their Role in Deep Learning"
date: 2026-03-27
description: "A detailed explanation of derivatives — what they are, how to compute them, and how they power gradient descent and backpropagation in deep learning."
categories:
    - Machine-Learning
tags:
    - derivatives
    - calculus
    - gradient-descent
    - backpropagation
draft: false
---

## What Is a Derivative?

A derivative measures **how much a function's output changes when you nudge its input by a tiny amount**.

Formally:

$$\frac{df}{dx} = \lim_{\epsilon \to 0} \frac{f(x + \epsilon) - f(x)}{\epsilon}$$

In plain English: if I increase $x$ by a very small amount $\epsilon$, how much does $f(x)$ change?

### Intuition: The Slope

The derivative at a point is the **slope of the tangent line** at that point.

```
f(x)
  |          /
  |        /  ← steep slope → large derivative
  |      /
  |    —      ← flat slope → small derivative
  |  \
  |    \      ← negative slope → negative derivative
  +-------------- x
```

- Positive derivative → function is increasing
- Negative derivative → function is decreasing
- Zero derivative → function is at a local minimum or maximum

---

## Basic Derivative Rules

### Power Rule

$$\frac{d}{dx} x^n = n x^{n-1}$$

| Function | Derivative |
|----------|-----------|
| $x^2$ | $2x$ |
| $x^3$ | $3x^2$ |
| $x$ | $1$ |
| $5$ (constant) | $0$ |

### Constant Multiple Rule

$$\frac{d}{dx} [c \cdot f(x)] = c \cdot f'(x)$$

Example: $\frac{d}{dx} 3x^2 = 3 \cdot 2x = 6x$

### Sum Rule

$$\frac{d}{dx} [f(x) + g(x)] = f'(x) + g'(x)$$

Example: $\frac{d}{dx} (x^2 + x) = 2x + 1$

### Chain Rule

This is the most important rule for deep learning.

$$\frac{d}{dx} f(g(x)) = f'(g(x)) \cdot g'(x)$$

Or written differently — if $y = f(u)$ and $u = g(x)$:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

Example: $\frac{d}{dx} (3x + 1)^2$

Let $u = 3x + 1$, so $f(u) = u^2$:

$$\frac{d}{dx} = \frac{d}{du}(u^2) \cdot \frac{d}{dx}(3x+1) = 2u \cdot 3 = 6(3x+1)$$

---

## Partial Derivatives

When a function has **multiple inputs**, a partial derivative measures how the output changes with respect to **one input at a time**, holding all others constant.

Notation: $\frac{\partial f}{\partial x}$ means "derivative of $f$ with respect to $x$, treating all other variables as constants."

### Example

$$f(x, y) = x^2 + 3xy + y^2$$

$$\frac{\partial f}{\partial x} = 2x + 3y \quad \text{(treat } y \text{ as constant)}$$

$$\frac{\partial f}{\partial y} = 3x + 2y \quad \text{(treat } x \text{ as constant)}$$

---

## The Gradient

The **gradient** is a vector of all partial derivatives of a function:

$$\nabla J = \begin{bmatrix} \frac{\partial J}{\partial w_1} \\ \frac{\partial J}{\partial w_2} \\ \vdots \\ \frac{\partial J}{\partial w_n} \\ \frac{\partial J}{\partial b} \end{bmatrix}$$

The gradient points in the direction of **steepest increase** of the function.

To minimise the function, we move in the **opposite direction** of the gradient — this is exactly what gradient descent does.

---

## Derivatives in Gradient Descent

Gradient descent uses derivatives to iteratively update parameters to minimise the cost function $J$.

### The Update Rule

$$w := w - \alpha \frac{\partial J}{\partial w}$$

- If $\frac{\partial J}{\partial w} > 0$ → $J$ increases as $w$ increases → decrease $w$ ✅
- If $\frac{\partial J}{\partial w} < 0$ → $J$ decreases as $w$ increases → increase $w$ ✅
- If $\frac{\partial J}{\partial w} = 0$ → at a minimum → $w$ does not change ✅

### Why the Learning Rate $\alpha$ Matters

The derivative tells us the **direction** to move. The learning rate $\alpha$ controls **how big a step** to take:

```
Cost J
  |    \
  |     \      ← large α: might overshoot the minimum
  |      \   ↗
  |       \_/  ← minimum
  |
  +-------------- w
```

- $\alpha$ too large → overshoot, may diverge
- $\alpha$ too small → converges but very slowly
- $\alpha$ just right → steady convergence

### Derivative at the Minimum

As gradient descent approaches the minimum, the slope $\frac{\partial J}{\partial w}$ gets smaller and smaller. This means the steps automatically get smaller — gradient descent naturally slows down near the minimum.

---

## Derivatives of Common Functions in Deep Learning

### Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

$$\frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))$$

Important property: the derivative is always between 0 and 0.25. This becomes a problem in very deep networks (vanishing gradients — covered below).

### ReLU (Rectified Linear Unit)

$$\text{ReLU}(z) = \max(0, z)$$

$$\frac{d}{dz}\text{ReLU}(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases}$$

The derivative is either 0 or 1 — simple and fast to compute. This is why ReLU is preferred over sigmoid in hidden layers of deep networks.

### Log Loss (Logistic Regression Cost)

$$J = -\frac{1}{m} \sum \left[ y \log(f) + (1-y) \log(1-f) \right]$$

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (f^{(i)} - y^{(i)}) x_j^{(i)}$$

This clean result comes from the chain rule applied through the sigmoid and log functions.

### Squared Error (Linear Regression Cost)

$$J = \frac{1}{2m} \sum (f^{(i)} - y^{(i)})^2$$

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (f^{(i)} - y^{(i)}) x_j^{(i)}$$

---

## Backpropagation: The Chain Rule at Scale

In a neural network, the cost $J$ depends on the output $\hat{y}$, which depends on the last layer's activations, which depend on the previous layer's activations, all the way back to the weights in the first layer.

To compute $\frac{\partial J}{\partial w}$ for any weight, we need to trace the chain of dependencies — this is **backpropagation**, which is just the **chain rule applied repeatedly**.

### A Simple Example

Consider a 2-layer network:

```
x → [w¹, b¹] → z¹ → a¹ → [w², b²] → z² → a² = ŷ → J
```

To find $\frac{\partial J}{\partial w^1}$, apply the chain rule:

$$\frac{\partial J}{\partial w^1} = \frac{\partial J}{\partial a^2} \cdot \frac{\partial a^2}{\partial z^2} \cdot \frac{\partial z^2}{\partial a^1} \cdot \frac{\partial a^1}{\partial z^1} \cdot \frac{\partial z^1}{\partial w^1}$$

Each term is a simple derivative. Backpropagation computes these from right to left (from output to input), reusing intermediate results to avoid redundant computation.

### Forward Pass vs Backward Pass

| Pass | Direction | Purpose |
|------|-----------|---------|
| Forward propagation | Input → Output | Compute prediction $\hat{y}$ and cost $J$ |
| Backward propagation | Output → Input | Compute $\frac{\partial J}{\partial w}$ for every weight |

After both passes, we have all the gradients needed to update every weight with gradient descent.

---

## The Vanishing Gradient Problem

In very deep networks using sigmoid activations, backpropagation multiplies many small derivatives together.

Since the sigmoid derivative is always $\leq 0.25$:

$$\frac{\partial J}{\partial w^1} = \underbrace{\frac{\partial J}{\partial a^n} \cdots}_{\text{many terms, each} \leq 0.25}$$

After 10 layers: $0.25^{10} \approx 0.000001$ — the gradient becomes nearly zero.

**Consequence:** Weights in early layers barely update — the network stops learning.

**Solution:** Use **ReLU** activation instead of sigmoid in hidden layers. ReLU's derivative is 1 (not < 1) for positive inputs, so gradients do not shrink as they propagate back.

---

## Summary: How Derivatives Power Deep Learning

```
1. Forward pass
   → compute ŷ using current weights
   → compute cost J

2. Backward pass (backpropagation)
   → apply chain rule layer by layer
   → compute ∂J/∂w for every weight

3. Gradient descent
   → w := w - α · ∂J/∂w
   → update every weight

4. Repeat until J converges
```

Every step of training a neural network — from the simplest logistic regression to the largest language model — relies on this loop. Derivatives are the engine that makes learning possible.

---

## Key Terms

| Term | Meaning |
|------|---------|
| **Derivative** | Rate of change of a function with respect to its input |
| **Partial derivative** | Derivative with respect to one variable, holding others fixed |
| **Gradient** | Vector of all partial derivatives; points in direction of steepest increase |
| **Chain rule** | Rule for differentiating composed functions; the foundation of backpropagation |
| **Gradient descent** | Iterative algorithm that follows the negative gradient to minimise cost |
| **Backpropagation** | Efficient application of the chain rule through all layers of a network |
| **Vanishing gradient** | Gradients shrink to near zero in deep networks using sigmoid; solved by ReLU |
| **Learning rate $\alpha$** | Controls step size in gradient descent |
| **ReLU** | Activation function with derivative 0 or 1; prevents vanishing gradients |
