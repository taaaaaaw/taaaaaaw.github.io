---
title: "Cost Function with Regularization"
date: 2026-03-25
description: "A detailed breakdown of regularization in both linear and logistic regression — what it does to the cost function, how it affects gradient descent, and how to choose lambda."
categories:
    - Machine-Learning
tags:
    - supervised-learning
    - regularization
    - linear-regression
    - logistic-regression
    - overfitting
draft: false
---

## Why Regularization?

When a model has too many features or polynomial terms, it can **overfit** — the weights grow very large to chase every training point, producing a complex curve that fails on new data.

Regularization fixes this by adding a **penalty for large weights** directly into the cost function. This forces gradient descent to keep the weights small, resulting in a simpler, smoother model.

---

## The Core Idea

Without regularization, gradient descent only cares about fitting the training data:

$$\min_{\vec{w}, b} \underbrace{\frac{1}{m} \sum_{i=1}^{m} \text{Loss}}_{\text{fit the data}}$$

With regularization, we add a second objective — keep the weights small:

$$\min_{\vec{w}, b} \underbrace{\frac{1}{m} \sum_{i=1}^{m} \text{Loss}}_{\text{fit the data}} + \underbrace{\frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2}_{\text{keep weights small}}$$

The parameter $\lambda$ controls the tradeoff between these two goals.

> **Note:** The bias term $b$ is **not** regularized by convention. Regularizing $b$ makes very little practical difference since it is just one parameter.

---

## The Regularization Parameter $\lambda$

| $\lambda$ value | Effect |
|----------------|--------|
| $\lambda = 0$ | No regularization — model may overfit |
| $\lambda$ very large | All weights pushed to ≈ 0 — model underfits (outputs $f \approx b$) |
| $\lambda$ just right | Balanced — model fits data without chasing noise |

Choosing $\lambda$ is a **hyperparameter tuning** problem. In practice, try a range of values (e.g. 0, 0.01, 0.1, 1, 10, 100) and pick the one with the lowest validation error.

---

## Regularized Linear Regression

### Cost Function

$$J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f^{(i)} - y^{(i)} \right)^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

Where $f^{(i)} = \vec{w} \cdot \vec{x}^{(i)} + b$.

The first term is the usual **squared error** — minimise prediction error.
The second term is the **regularization penalty** — minimise the sum of squared weights.

### Gradient Descent

Simultaneously update all parameters on each step:

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j} \quad \text{for } j = 1, \ldots, n$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

The partial derivatives:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left( f^{(i)} - y^{(i)} \right) x_j^{(i)} + \frac{\lambda}{m} w_j$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( f^{(i)} - y^{(i)} \right)$$

Substituting the $w_j$ update:

$$w_j := w_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} \left( f^{(i)} - y^{(i)} \right) x_j^{(i)} + \frac{\lambda}{m} w_j \right]$$

This can be rearranged to:

$$w_j := w_j \left(1 - \alpha \frac{\lambda}{m}\right) - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( f^{(i)} - y^{(i)} \right) x_j^{(i)}$$

### Weight Shrinkage Interpretation

The term $\left(1 - \alpha \frac{\lambda}{m}\right)$ is slightly less than 1 (e.g. 0.999).

Every update step **shrinks $w_j$ by a small factor** before applying the usual gradient correction. Over many steps, this prevents any single weight from growing too large.

### Example: Effect on a Polynomial Model

Suppose we fit a 4th-degree polynomial $f(x) = w_1 x + w_2 x^2 + w_3 x^3 + w_4 x^4 + b$:

- **No regularization ($\lambda = 0$):** weights can grow large, curve fits every point — overfitting
- **$\lambda = 1$:** weights are controlled, smooth curve — good fit
- **$\lambda = 10000$:** all weights ≈ 0, model reduces to $f(x) \approx b$ — underfitting (flat line)

---

## Regularized Logistic Regression

### Cost Function

$$J(\vec{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f^{(i)}) + (1 - y^{(i)}) \log(1 - f^{(i)}) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

Where $f^{(i)} = g(\vec{w} \cdot \vec{x}^{(i)} + b) = \dfrac{1}{1 + e^{-(\vec{w} \cdot \vec{x}^{(i)} + b)}}$.

The first term is the **log loss** — the standard logistic regression cost.
The second term is the same **regularization penalty** as in linear regression.

### Gradient Descent

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j} \quad \text{for } j = 1, \ldots, n$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

The partial derivatives:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left( f^{(i)} - y^{(i)} \right) x_j^{(i)} + \frac{\lambda}{m} w_j$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( f^{(i)} - y^{(i)} \right)$$

The update equations look **identical to regularized linear regression** — the only difference is that $f^{(i)}$ is the sigmoid output instead of a linear output.

### Effect on the Decision Boundary

Regularization does not just shrink weights — it directly shapes the decision boundary:

- **No regularization:** boundary can be highly irregular, perfectly separating all training points
- **Moderate $\lambda$:** boundary is smoother, may misclassify a few training points but generalises better
- **Very large $\lambda$:** boundary becomes very simple (near-linear), many training points misclassified — underfitting

---

## Comparing the Two

| | Regularized Linear Regression | Regularized Logistic Regression |
|---|---|---|
| $f^{(i)}$ | $\vec{w} \cdot \vec{x}^{(i)} + b$ | $g(\vec{w} \cdot \vec{x}^{(i)} + b)$ |
| Base cost | Squared error | Log loss |
| Regularization term | $\dfrac{\lambda}{2m} \sum w_j^2$ | $\dfrac{\lambda}{2m} \sum w_j^2$ |
| $\partial J / \partial w_j$ formula | Same form | Same form |
| Output | Continuous number | Probability (0 to 1) |
| Task | Regression | Classification |

The gradient descent update equations have the **same structure** for both — only $f^{(i)}$ differs.

---

## Intuition: What Regularization Is Really Doing

Think of each weight $w_j$ as controlling how much influence feature $x_j$ has on the prediction.

Without regularization, gradient descent will happily set $w_{100}$ to a huge value like 500 if that helps reduce training error even slightly. This leads to a model that is extremely sensitive to small changes in $x_{100}$.

With regularization, the cost function says: "yes, reducing training error is good — but making $w_{100} = 500$ is also very costly." The model is forced to find a balance — only assign large weights to features that genuinely matter.

This is why regularization tends to produce models where:
- Truly important features have moderate, meaningful weights
- Less important features have weights close to zero
- The overall model is stable and generalises well

---

## Choosing $\lambda$ in Practice

1. Split data into **training set**, **validation set**, and **test set**
2. Train the model with several values of $\lambda$ (e.g. 0, 0.01, 0.1, 1, 10, 100)
3. Evaluate each model on the **validation set**
4. Pick the $\lambda$ with the lowest validation error
5. Report final performance on the **test set** (used only once)

```
λ too small → low training error, high validation error → overfit
λ just right → low training error, low validation error → good
λ too large  → high training error, high validation error → underfit
```

---

## Key Terms

| Term | Meaning |
|------|---------|
| **Regularization** | Adding a weight penalty to the cost function to prevent overfitting |
| **$\lambda$** | Regularization parameter; controls the strength of the penalty |
| **L2 regularization** | Penalty proportional to $\sum w_j^2$; used here |
| **Weight shrinkage** | Each gradient step slightly reduces $w_j$ toward zero |
| **Hyperparameter** | A parameter set before training (like $\lambda$ or $\alpha$) — not learned by gradient descent |
| **Validation set** | Held-out data used to choose $\lambda$ |
| **Overfitting** | Model too complex; large weights; fits training noise |
| **Underfitting** | Model too simple; weights too small; misses the pattern |
