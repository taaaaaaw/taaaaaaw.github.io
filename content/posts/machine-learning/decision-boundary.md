---
title: "Decision Boundary"
date: 2026-03-24
description: "A detailed look at decision boundaries in logistic regression — what they are, how they are determined, and how to create non-linear boundaries with polynomial features."
categories:
    - Machine-Learning
tags:
    - supervised-learning
    - classification
    - logistic-regression
draft: false
---

## What Is a Decision Boundary?

In classification, the model needs to output a hard label — either **0 or 1**. The **decision boundary** is the line (or curve) that separates the region where the model predicts 1 from the region where it predicts 0.

Everything on one side → predict 1
Everything on the other side → predict 0

---

## How It Is Determined

Recall the logistic regression model:

$$f_{\vec{w},b}(\vec{x}) = g(z) = \frac{1}{1 + e^{-z}}, \quad z = \vec{w} \cdot \vec{x} + b$$

We apply a threshold of 0.5:

$$\hat{y} = \begin{cases} 1 & \text{if } f \geq 0.5 \\ 0 & \text{if } f < 0.5 \end{cases}$$

Since the sigmoid $g(z) \geq 0.5$ exactly when $z \geq 0$, the threshold becomes:

$$\hat{y} = 1 \quad \text{when} \quad z = \vec{w} \cdot \vec{x} + b \geq 0$$

The **decision boundary is the set of points where $z = 0$**, i.e.:

$$\vec{w} \cdot \vec{x} + b = 0$$

> Note: the decision boundary is a property of the **parameters** $\vec{w}$ and $b$, not of the training data itself. The data influences what values gradient descent learns for $\vec{w}$ and $b$, which then defines the boundary.

---

## Linear Decision Boundary

With two features $x_1$ and $x_2$, the model is:

$$z = w_1 x_1 + w_2 x_2 + b$$

The boundary $z = 0$ gives:

$$w_1 x_1 + w_2 x_2 + b = 0$$

This is the equation of a **straight line** in the $x_1$–$x_2$ plane.

### Example

Suppose gradient descent learns: $w_1 = 1,\ w_2 = 1,\ b = -3$

$$z = x_1 + x_2 - 3 = 0 \quad \Rightarrow \quad x_1 + x_2 = 3$$

The boundary is the line $x_1 + x_2 = 3$.

- Points where $x_1 + x_2 \geq 3$ → predict **1**
- Points where $x_1 + x_2 < 3$ → predict **0**

```
x₂
 |        * * *   (predict 1)
 |      /
 3    /  ← boundary: x₁ + x₂ = 3
 |  /
 | /  o o o      (predict 0)
 +-------------- x₁
       3
```

---

## Non-Linear Decision Boundary

A straight line cannot separate all datasets. If the two classes form a circular pattern, a line will always misclassify many points.

The solution: **add polynomial features** to create a more complex $z$.

### Circle Boundary

Add features $x_1^2$ and $x_2^2$:

$$z = w_1 x_1^2 + w_2 x_2^2 + b$$

Suppose gradient descent learns: $w_1 = 1,\ w_2 = 1,\ b = -1$

$$z = x_1^2 + x_2^2 - 1 = 0 \quad \Rightarrow \quad x_1^2 + x_2^2 = 1$$

This is a **circle of radius 1**.

- Points **inside** the circle ($x_1^2 + x_2^2 < 1$) → predict **0**
- Points **outside** the circle ($x_1^2 + x_2^2 \geq 1$) → predict **1**

```
x₂
 1  * * * * *
   *    o    *
   *  o   o  *   o = class 0 (inside)
   *    o    *   * = class 1 (outside)
 -1  * * * * *
    -1       1   x₁
```

### Ellipse Boundary

With different weights for $x_1^2$ and $x_2^2$:

$$z = w_1 x_1^2 + w_2 x_2^2 + b = 0$$

When $w_1 \neq w_2$, the boundary becomes an **ellipse**.

### Even More Complex Boundaries

With higher-order or cross terms:

$$z = w_1 x_1 + w_2 x_2 + w_3 x_1^2 + w_4 x_1 x_2 + w_5 x_2^2 + b$$

The boundary can become a complex curve that wraps around irregular clusters. The more polynomial terms you add, the more flexible (and potentially more complex) the boundary.

---

## The Boundary Lives in Feature Space

An important distinction:

| Plot | Axes | What is shown |
|------|------|---------------|
| Scatterplot | $x_1$, $x_2$ (features) | Training data points + decision boundary |
| Contour plot | $w_1$, $w_2$ (parameters) | Cost function $J$ over parameter space |

The decision boundary is drawn in **feature space** (the scatterplot), not in parameter space. When you visualise the boundary, you are asking: "for which input values $(x_1, x_2)$ does the model switch its prediction?"

---

## Effect of Parameters on the Boundary

Changing $\vec{w}$ and $b$ directly changes the position and orientation of the boundary.

| Change | Effect on boundary |
|--------|-------------------|
| Increase $b$ | Shifts the boundary (parallel shift) |
| Change $w_1 / w_2$ ratio | Rotates the boundary |
| Add polynomial features | Allows curved boundaries |
| Increase regularisation $\lambda$ | Simpler, smoother boundary (less overfit) |
| Decrease regularisation $\lambda$ | More complex boundary (risk of overfit) |

---

## Overfitting and the Decision Boundary

With too many polynomial features and no regularisation, the boundary can become overly complex — perfectly separating the training data but failing on new examples.

**Underfitting** — boundary too simple, misclassifies both classes:
```
  * * | o o o
  * * | o o o     straight line, but data needs a curve
```

**Good fit** — boundary matches the true pattern:
```
  * *   o o o
  * * ( o o o )   curved boundary fits well
```

**Overfitting** — boundary twists to capture every training point:
```
  *~*~o~*~o~o     jagged boundary, won't generalise
```

Regularisation (the $\lambda$ term in the cost function) penalises large weights, forcing the boundary to remain smooth.

---

## Key Terms

| Term | Meaning |
|------|---------|
| **Decision boundary** | The surface where $z = 0$; separates predicted class 0 from class 1 |
| **Linear boundary** | A straight line; results from using raw features only |
| **Non-linear boundary** | A curve; results from adding polynomial features |
| **Feature space** | The space defined by input features $x_1, x_2, \ldots$ |
| **Threshold** | The probability cutoff (default 0.5) that defines where the boundary sits |
| **Overfitting** | Boundary too complex; fits training data but fails on new data |
| **Regularisation** | Penalty on large weights to keep the boundary smooth |
