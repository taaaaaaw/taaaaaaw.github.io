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
