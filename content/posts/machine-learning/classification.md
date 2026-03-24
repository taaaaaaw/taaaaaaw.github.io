---
title: "Classification"
date: 2026-03-24
description: "What is classification, how it differs from regression, and the key concepts behind logistic regression."
categories:
    - Machine-Learning
tags:
    - supervised-learning
    - classification
    - logistic-regression
draft: false
---

## What Is Classification?

Classification is a type of supervised learning where the output **y is a category**, not a continuous number.

| Problem | Output |
|---------|--------|
| Is this email spam? | Yes / No |
| Is this tumour malignant? | Yes / No |
| Which digit is this? | 0, 1, 2 … 9 |

The first two examples are **binary classification** — only two possible outputs, usually labelled **0** (negative) and **1** (positive).

---

## Why Not Use Linear Regression?

You might think: can't I just use linear regression and say "if output > 0.5, predict 1"?

The problem is that linear regression is sensitive to outliers. Adding a data point far to the right shifts the line and changes all predictions — even the ones that were already correct.

Linear regression also produces values outside the range [0, 1], which makes it hard to interpret as a probability.

---

## Logistic Regression

Logistic regression is the standard algorithm for binary classification. Despite the name, it is a **classification** algorithm, not a regression one.

### The Sigmoid Function

Instead of outputting a raw number, logistic regression passes the output through the **sigmoid function** (also called the logistic function):

$$g(z) = \frac{1}{1 + e^{-z}}$$

The sigmoid function always outputs a value between **0 and 1**, which we can interpret as a probability.

| z | g(z) |
|---|------|
| very large (e.g. +10) | ≈ 1 |
| 0 | 0.5 |
| very small (e.g. -10) | ≈ 0 |

### The Full Model

$$z = \vec{w} \cdot \vec{x} + b$$

$$f_{\vec{w},b}(\vec{x}) = g(z) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$$

The output $f$ is interpreted as:

> "The probability that y = 1 given input x"

For example, if $f = 0.7$, the model is saying there is a **70% chance** the label is 1.

---

## Decision Boundary

To make an actual prediction (0 or 1), we pick a **threshold** — usually **0.5**:

$$\hat{y} = \begin{cases} 1 & \text{if } f \geq 0.5 \\ 0 & \text{if } f < 0.5 \end{cases}$$

Since $g(z) \geq 0.5$ when $z \geq 0$, this is equivalent to:

$$\hat{y} = 1 \quad \text{when} \quad \vec{w} \cdot \vec{x} + b \geq 0$$

The **decision boundary** is the line (or curve) where $z = 0$, which is where the model transitions from predicting 0 to predicting 1.

### Linear Decision Boundary

With two features x₁ and x₂, the boundary $w_1x_1 + w_2x_2 + b = 0$ is a straight line dividing the two classes.

### Non-Linear Decision Boundary

By adding polynomial features (e.g. $x_1^2$, $x_2^2$), the decision boundary can become a curve — a circle, ellipse, or more complex shape — allowing the model to separate classes that are not linearly separable.

---

## Cost Function for Logistic Regression

The **squared error cost** used in linear regression does not work well here — it produces a non-convex curve with many local minima, making gradient descent unreliable.

Instead, logistic regression uses the **log loss** (also called binary cross-entropy):

$$J(\vec{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f^{(i)}) + (1 - y^{(i)}) \log(1 - f^{(i)}) \right]$$

### Why This Works

The loss for a single example is:

$$L = \begin{cases} -\log(f) & \text{if } y = 1 \\ -\log(1 - f) & \text{if } y = 0 \end{cases}$$

- When $y = 1$: if $f \to 1$, loss $\to 0$. If $f \to 0$, loss $\to \infty$.
- When $y = 0$: if $f \to 0$, loss $\to 0$. If $f \to 1$, loss $\to \infty$.

This heavily penalises confident wrong predictions, and the overall cost function is **convex** — gradient descent is guaranteed to find the global minimum.

---

## Gradient Descent for Logistic Regression

The update rule looks identical to linear regression:

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

Where the derivatives work out to:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (f^{(i)} - y^{(i)}) x_j^{(i)}$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (f^{(i)} - y^{(i)})$$

The key difference from linear regression: $f$ here is the **sigmoid** output, not a raw linear value.

---

## Key Terms

| Term | Meaning |
|------|---------|
| **Binary classification** | Output is one of two classes (0 or 1) |
| **Sigmoid / logistic function** | Maps any number to (0, 1) |
| **Decision boundary** | The line/curve where the model switches prediction |
| **Log loss / cross-entropy** | The cost function used for classification |
| **Threshold** | The cutoff (usually 0.5) to convert probability to a class label |
