---
title: "Logistic Regression"
date: 2026-03-24
description: "A deep dive into logistic regression — the sigmoid function, decision boundary, cost function, and gradient descent."
categories:
    - Machine-Learning
tags:
    - supervised-learning
    - classification
    - logistic-regression
draft: false
---

## What Is Logistic Regression?

Logistic regression is a **supervised learning algorithm for binary classification**. Despite the word "regression" in its name, it predicts **categories** (0 or 1), not continuous numbers.

Common use cases:
- Is this email spam? (1 = spam, 0 = not spam)
- Is this transaction fraudulent? (1 = fraud, 0 = legitimate)
- Is this tumour malignant? (1 = malignant, 0 = benign)

---

## From Linear to Logistic

In linear regression, the model outputs:

$$f(\vec{x}) = \vec{w} \cdot \vec{x} + b$$

This can produce any value from $-\infty$ to $+\infty$, which is not useful for classification. We want the output to be a **probability between 0 and 1**.

The fix is to wrap the linear output in a **sigmoid function**.

---

## The Sigmoid Function

$$g(z) = \frac{1}{1 + e^{-z}}$$

Key properties:

| Input z | Output g(z) |
|---------|-------------|
| Very large (e.g. +100) | ≈ 1 |
| 0 | exactly 0.5 |
| Very small (e.g. -100) | ≈ 0 |

The sigmoid function **always outputs between 0 and 1**, which we interpret as a probability.

### The Full Logistic Regression Model

$$z = \vec{w} \cdot \vec{x} + b$$

$$f_{\vec{w},b}(\vec{x}) = g(z) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$$

The output $f$ means:

> "The probability that y = 1, given input $\vec{x}$ and parameters $\vec{w}$, $b$"

Formally written as $P(y=1 \mid \vec{x}; \vec{w}, b)$.

Since probabilities must sum to 1:

$$P(y=0) + P(y=1) = 1$$

---

## Decision Boundary

The model outputs a probability, but we need a hard prediction of 0 or 1. We use a **threshold** — typically **0.5**:

$$\hat{y} = \begin{cases} 1 & \text{if } f \geq 0.5 \\ 0 & \text{if } f < 0.5 \end{cases}$$

Since $g(z) \geq 0.5$ whenever $z \geq 0$, this is equivalent to:

$$\hat{y} = 1 \quad \text{when} \quad \vec{w} \cdot \vec{x} + b \geq 0$$

The **decision boundary** is the set of points where $z = \vec{w} \cdot \vec{x} + b = 0$ — this is exactly where the model transitions from predicting 0 to predicting 1.

### Linear Decision Boundary

With two features $x_1$ and $x_2$, if the learned parameters are $w_1 = 1$, $w_2 = 1$, $b = -3$:

$$z = x_1 + x_2 - 3 = 0 \quad \Rightarrow \quad x_1 + x_2 = 3$$

This is a **straight line** separating the two classes.

### Non-Linear Decision Boundary

By engineering polynomial features like $x_1^2$ and $x_2^2$, we can create curved boundaries. For example, with parameters $w_1 = 1$, $w_2 = 1$, $b = -1$:

$$z = x_1^2 + x_2^2 - 1 = 0 \quad \Rightarrow \quad x_1^2 + x_2^2 = 1$$

This is a **circle** of radius 1. Points inside the circle are predicted as 1, points outside as 0.

---

## Cost Function

### Why Not Squared Error?

For linear regression we used:

$$J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} (f^{(i)} - y^{(i)})^2$$

If we plug the sigmoid function in for $f$, this cost function becomes **non-convex** — it has many local minima, so gradient descent may get stuck and never find the best parameters.

### Log Loss (Binary Cross-Entropy)

Logistic regression uses **log loss**, which is convex and guarantees gradient descent finds the global minimum:

$$J(\vec{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f^{(i)}) + (1 - y^{(i)}) \log(1 - f^{(i)}) \right]$$

### Understanding the Loss

The loss for a single training example breaks into two cases:

**When y = 1:**

$$L = -\log(f)$$

- If $f = 1$ (perfect prediction) → $L = 0$
- If $f = 0$ (completely wrong) → $L = \infty$

**When y = 0:**

$$L = -\log(1 - f)$$

- If $f = 0$ (perfect prediction) → $L = 0$
- If $f = 1$ (completely wrong) → $L = \infty$

This means the cost function **heavily penalises confident wrong predictions**, which is exactly the behaviour we want.

---

## Gradient Descent

To minimise the cost, we use gradient descent — repeatedly updating each parameter in the direction that reduces J:

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

The partial derivatives work out to:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left( f^{(i)} - y^{(i)} \right) x_j^{(i)}$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( f^{(i)} - y^{(i)} \right)$$

These look identical to the linear regression update rules — the only difference is that $f$ is now the **sigmoid output** instead of a raw linear value.

All parameters $w_1, w_2, \ldots, w_n$ and $b$ must be updated **simultaneously** on each step.

---

## Regularisation

When a model has too many features or polynomial terms, it can **overfit** — memorising the training data but performing poorly on new data. Regularisation penalises large weights to keep the model simple.

### Regularised Cost Function

$$J(\vec{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f^{(i)}) + (1 - y^{(i)}) \log(1 - f^{(i)}) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

- $\lambda$ is the **regularisation parameter** — controls how much to penalise large weights
- Large $\lambda$ → simpler model (may underfit)
- Small $\lambda$ → complex model (may overfit)
- $b$ is **not** regularised by convention

### Regularised Gradient Descent

The $w_j$ update becomes:

$$w_j := w_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} \left( f^{(i)} - y^{(i)} \right) x_j^{(i)} + \frac{\lambda}{m} w_j \right]$$

This can be rewritten as:

$$w_j := w_j \left(1 - \alpha \frac{\lambda}{m}\right) - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f^{(i)} - y^{(i)} \right) x_j^{(i)}$$

The term $\left(1 - \alpha \frac{\lambda}{m}\right)$ is slightly less than 1, so each step **shrinks $w_j$ a little** before applying the normal gradient update.

---

## Key Terms

| Term | Meaning |
|------|---------|
| **Sigmoid function** | Maps any real number to (0, 1); used to output probabilities |
| **Decision boundary** | The line/curve where $z = 0$; separates the two predicted classes |
| **Log loss** | The convex cost function used for logistic regression |
| **Threshold** | The probability cutoff (default 0.5) to decide between class 0 and 1 |
| **Regularisation** | A penalty on large weights to prevent overfitting |
| **$\lambda$** | Regularisation parameter; controls the strength of the penalty |
| **Overfitting** | Model fits training data too well but generalises poorly |
| **Underfitting** | Model is too simple and fails to capture the pattern in the data |
