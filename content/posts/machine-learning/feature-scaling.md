---
title: "Feature Scaling"
date: 2026-03-20
description: "Why feature scaling matters, and the common techniques used to scale your data before training."
categories:
    - Machine-Learning
tags:
    - supervised-learning
    - preprocessing
    - gradient-descent
draft: false
---

## What Is Feature Scaling?

Feature scaling is the process of **normalizing the range of input features** so they are on a similar scale. Without it, features with larger values can dominate the learning process and cause gradient descent to behave poorly.

---

## Why Does It Matter?

Consider predicting house prices with two features:

- **Size**: ranges from 300 to 5000 sq ft
- **Number of bedrooms**: ranges from 1 to 5

Without scaling, the cost function becomes elongated and skewed — gradient descent has to take many small steps in one direction and large steps in another, making it slow to converge.

With scaling, both features are on a similar range, the cost function becomes more symmetric, and gradient descent converges much faster.

---

## Common Techniques

### Divide by Maximum

The simplest scaling method — just divide each value by the maximum value of that feature:

$$x' = \frac{x}{x_{max}}$$

For example, if house sizes range from 300 to 2000 sq ft:

$$x_{1,scaled} = \frac{x_1}{2000} \quad \Rightarrow \quad 0.15 \leq x_{1,scaled} \leq 1$$

The result won't start from 0 (unlike Min-Max), but the values are compressed into a small, manageable range. It's fast and intuitive, but like Min-Max, it's sensitive to outliers since the maximum value drives the scaling.

---

### Min-Max Normalization

Rescales each feature to a fixed range, typically $[0, 1]$:

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

Simple and intuitive, but sensitive to outliers — one extreme value can compress everything else into a tiny range.

---

### Mean Normalization

Centers the data around zero:

$$x' = \frac{x - \mu}{x_{max} - x_{min}}$$

Where $\mu$ is the mean of the feature. The result roughly falls in the range $[-1, 1]$.

---

### Z-score Standardization

Transforms the data to have **mean = 0** and **standard deviation = 1**:

$$x' = \frac{x - \mu}{\sigma}$$

Where:
- $\mu$ = mean of the feature
- $\sigma$ = standard deviation of the feature

**How to compute $\sigma$:**

$$\sigma = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu)^2}$$

Step by step:
1. Compute the mean $\mu$
2. Subtract the mean from each value to get the deviation
3. Square each deviation (to remove negatives)
4. Average the squared deviations — this is the **variance** $\sigma^2$
5. Take the square root to get the standard deviation $\sigma$

Intuitively, $\sigma$ measures how **spread out** the data is around the mean. A large $\sigma$ means the data is widely scattered; a small $\sigma$ means it's tightly clustered.

This is the most widely used technique because it handles outliers better than min-max normalization and works well with most machine learning algorithms.

---

## Comparing the Three Methods

| | Min-Max | Mean Normalization | Z-score |
|---|---|---|---|
| Output range | $[0, 1]$ | $\approx [-1, 1]$ | No fixed range |
| Mean after scaling | Not 0 | 0 | 0 |
| Sensitive to outliers | Yes | Yes | Less so |
| Most common use | Neural networks | General | General |

---

## Key Rules of Thumb

- Aim for features to be roughly in the range $[-1, 1]$ or $[0, 1]$
- Features already in a small, similar range (e.g., 0 to 3) don't necessarily need scaling
- Features with very large ranges (e.g., 0 to 100,000) almost always benefit from scaling
- Always scale using statistics computed from the **training set only** — then apply the same transformation to the validation and test sets

---

## When Is Feature Scaling Needed?

**Needed:**
- Gradient descent-based algorithms (linear regression, logistic regression, neural networks)
- Distance-based algorithms (K-nearest neighbors, SVM, K-means)

**Not needed:**
- Tree-based algorithms (decision trees, random forests, XGBoost) — they split on thresholds and are unaffected by scale
