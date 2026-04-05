---
title: "Polynomial Regression"
date: 2026-03-20
description: "What polynomial regression is, why it exists, and how it extends linear regression to fit curved relationships."
categories:
    - Machine-Learning
tags:
    - supervised-learning
    - regression
draft: false
---

## Why Not Just Linear Regression?

Linear regression fits a straight line through the data. But real-world relationships are often curved — for example, the price of a house doesn't grow linearly with size forever, and crop yield vs. fertilizer amount follows a curve that eventually drops off.

When data has a non-linear pattern, forcing a straight line through it leads to **underfitting** — the model is too simple to capture what's actually going on.

Polynomial regression solves this by fitting a **curve** instead of a line.

---

## What Is Polynomial Regression?

Polynomial regression extends linear regression by adding **higher-order terms** of the input feature. Instead of:

$$\hat{y} = w_1 x + b$$

You use:

$$\hat{y} = w_1 x + w_2 x^2 + w_3 x^3 + \dots + b$$

Each term is a power of $x$. The model can now bend and curve to fit more complex patterns.

Despite the name, this is still a **linear** model at heart — it's linear in the parameters $w_1, w_2, w_3$. Only the features are non-linear. This means gradient descent and the normal equation still apply.

---

## A Concrete Example

Suppose you're predicting house prices based on size $x$ (in sq ft). A straight line might underfit, so you try a cubic polynomial:

$$\hat{y} = w_1 x + w_2 x^2 + w_3 x^3 + b$$

You treat $x$, $x^2$, and $x^3$ as three separate features and feed them into a standard linear regression model. The model learns the best $w_1$, $w_2$, $w_3$, and $b$ to minimize the cost.

---

## Alternative Feature Choices

You're not limited to powers of $x$. You can also engineer features like:

$$\hat{y} = w_1 x + w_2 \sqrt{x} + b$$

Square root flattens out quickly and might better capture diminishing returns. The key insight is: **you choose the shape of the features**, and the model learns the best weights for them.

This is why **feature engineering** — deciding which transformations to apply — is an important skill in machine learning.

---

## The Degree of a Polynomial

The **degree** is the highest power used:

| Degree | Name | Equation |
|---|---|---|
| 1 | Linear | $w_1 x + b$ |
| 2 | Quadratic | $w_1 x + w_2 x^2 + b$ |
| 3 | Cubic | $w_1 x + w_2 x^2 + w_3 x^3 + b$ |
| $n$ | Degree-$n$ polynomial | $w_1 x + \dots + w_n x^n + b$ |

Higher degree = more flexible curve = can fit more complex patterns.

---

## The Overfitting Problem

The more flexible the model, the greater the risk of **overfitting** — where the model fits the training data too well, including the noise, and performs poorly on new data.

For example, a degree-10 polynomial might wiggle through every single training point perfectly but make terrible predictions on unseen data.

This is the fundamental tradeoff in polynomial regression:

- **Too low degree** → underfitting (misses the pattern)
- **Too high degree** → overfitting (memorizes the noise)
- **Just right** → generalizes well to new data

---

## Feature Scaling Is Essential

When you add polynomial terms, the scale differences become extreme. For example, if $x$ ranges from 1 to 1000:

- $x$ ranges up to $10^3$
- $x^2$ ranges up to $10^6$
- $x^3$ ranges up to $10^9$

Without feature scaling, gradient descent will struggle badly. **Always apply feature scaling** when using polynomial features.

---

## How to Choose the Right Degree

There is no formula — you evaluate model performance on a **validation set** (data not used during training):

1. Train models with degree 1, 2, 3, ...
2. Measure error on the validation set for each
3. Pick the degree that gives the lowest validation error

This process is part of **model selection**, which will be covered in a later post.
