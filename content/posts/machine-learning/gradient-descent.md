---
title: "Gradient Descent for Linear Regression"
date: 2026-03-17
description: "What gradient descent is, how it works with linear regression, and the key terms you need to understand it."
categories:
    - Machine-Learning
tags:
    - supervised-learning
    - regression
    - gradient-descent
    - optimization
draft: false
---

## What Is Gradient Descent?

Gradient descent is the **optimization algorithm** that teaches the model to improve itself. After each prediction, it looks at how wrong the answer was and nudges the model's parameters in the direction that reduces the error.

Think of it like being blindfolded on a hilly landscape. Your goal is to reach the lowest point (minimum error). You can't see the whole map, but you can feel which direction slopes downhill — so you take a step in that direction, check again, and repeat until you stop going down.

---

## The Cost Function

Before gradient descent can run, you need something to minimize. In linear regression, that's the **cost function** (also called loss function):

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2$$

Where:
- $m$ = number of training examples
- $\hat{y}^{(i)} = wx^{(i)} + b$ = predicted value for example $i$
- $y^{(i)}$ = actual value for example $i$

The $\frac{1}{2}$ is just a convenience to make the derivative cleaner. The goal is to find $w$ and $b$ that make $J(w, b)$ as small as possible.

---

## The Gradient Descent Algorithm

Repeat until convergence:

$$w := w - \alpha \frac{\partial J(w,b)}{\partial w}$$

$$b := b - \alpha \frac{\partial J(w,b)}{\partial b}$$

The partial derivatives work out to:

$$\frac{\partial J(w,b)}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right) x^{(i)}$$

$$\frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)$$

**Important:** $w$ and $b$ must be updated **simultaneously** — calculate both new values first, then assign them at the same time.

---

## Key Terminology

### Parameters ($w$ and $b$)
The values the model learns during training.
- $w$ (weight) — the slope, controls how much the input affects the output
- $b$ (bias) — the intercept, shifts the line up or down

### Learning Rate ($\alpha$)
Controls how big each step is during gradient descent. A critical hyperparameter:
- **Too large** → overshoots the minimum, may never converge or even diverge
- **Too small** → converges, but very slowly
- **Just right** → smoothly reaches the minimum

### Gradient
The partial derivative of the cost function with respect to a parameter. It tells you the **direction and steepness** of the slope at the current position. Gradient descent moves in the **opposite** direction of the gradient (downhill).

### Convergence
When the parameters stop changing significantly between steps — the algorithm has found (or is very close to) the minimum. In practice, you stop when the cost function decreases by less than a tiny threshold.

### Iteration (Epoch)
One full pass through the update equations. Gradient descent runs for many iterations until convergence.

### Batch Gradient Descent
The standard form shown above — uses **all** $m$ training examples to compute the gradient at each step. Accurate but slow on large datasets.

---

## Why It Works for Linear Regression

Linear regression has a **convex** cost function — it has exactly one minimum and no local minima. This means gradient descent is guaranteed to find the global minimum as long as the learning rate isn't too large.

This is not always the case in more complex models like neural networks, where the cost surface can have many local minima.

---

## Intuition Behind the Update Rule

When $\hat{y}^{(i)} > y^{(i)}$ (prediction too high):
- The error term $(\hat{y} - y)$ is positive
- The gradient is positive
- $w$ decreases → the line gets less steep → predictions come down

When $\hat{y}^{(i)} < y^{(i)}$ (prediction too low):
- The error term is negative
- The gradient is negative
- $w$ increases → the line gets steeper → predictions go up

Each update nudges the line closer to the data.

---

## Alternative: The Normal Equation

Instead of iteratively adjusting parameters, the **Normal Equation** solves for $w$ and $b$ directly in one step using linear algebra:

$$\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

Where:
- $\mathbf{X}$ = the matrix of input features (with a column of 1s added for the bias term)
- $\mathbf{y}$ = the vector of target values

No learning rate, no iterations — just plug in the data and get the optimal parameters immediately.

### Drawbacks

- **Slow on large datasets** — computing $(\mathbf{X}^T \mathbf{X})^{-1}$ requires matrix inversion, which scales at roughly $O(n^3)$ where $n$ is the number of features. With thousands of features, this becomes very expensive.
- **Only works for linear regression** — the Normal Equation has no equivalent for other models like logistic regression or neural networks, so gradient descent remains the general-purpose tool.
- **Matrix may not be invertible** — if features are linearly dependent (e.g., duplicate columns) or you have more features than training examples, $\mathbf{X}^T \mathbf{X}$ becomes singular and cannot be inverted.

### When to Use Which

| | Gradient Descent | Normal Equation |
|---|---|---|
| Number of features | Works well with large $n$ | Slow when $n$ is large |
| Iterations needed | Yes | No |
| Learning rate | Must be tuned | Not needed |
| Applicability | Any model | Linear regression only |
