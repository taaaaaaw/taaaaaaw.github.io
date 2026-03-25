---
title: "Overfitting and Underfitting"
date: 2026-03-25
description: "A detailed look at underfitting and overfitting — what causes them, how to detect them, and how to fix them using regularisation and other techniques."
categories:
    - Machine-Learning
tags:
    - supervised-learning
    - overfitting
    - underfitting
    - regularisation
    - bias-variance
draft: false
---

## The Goal of Generalisation

When we train a model, the real goal is not to perform well on the **training data** — it is to perform well on **new, unseen data**. This ability is called **generalisation**.

A model that generalises well has learned the true underlying pattern, not just memorised the training examples.

---

## Underfitting (High Bias)

Underfitting occurs when the model is **too simple** to capture the pattern in the data. It performs poorly on both training data and new data.

### What It Looks Like

Suppose you are predicting house prices from size. The true relationship is slightly curved, but you fit a straight line:

```
Price
  |          * *
  |       * /
  |     */            ← straight line misses the curve
  |   */
  | */  *
  +-------------- Size
```

The line cannot capture the curve no matter how long you train it. The model has **high bias** — it brings its own strong assumption (linearity) that does not match the data.

### Signs of Underfitting

- Training error is **high**
- Validation error is also **high**
- Training and validation errors are close to each other (both bad)

### Causes

- Model is too simple (e.g. linear model for non-linear data)
- Too few features
- Too much regularisation (over-penalising the weights)
- Not enough training (stopped gradient descent too early)

---

## Overfitting (High Variance)

Overfitting occurs when the model is **too complex** and learns the noise and specific quirks of the training data rather than the true pattern. It performs very well on training data but poorly on new data.

### What It Looks Like

Using a very high-degree polynomial to fit the same house price data:

```
Price
  |    *   *
  |   / \ / \        ← curve passes through every point
  |  /   *   \
  | /          \  *
  |*             **
  +-------------- Size
```

The model memorised every training point. Ask it to predict a new house size and it will likely give a wildly wrong answer.

### Signs of Overfitting

- Training error is **very low** (close to 0)
- Validation error is **much higher** than training error
- Large gap between training and validation performance

### Causes

- Model is too complex (too many parameters or high-degree polynomial)
- Too many features relative to training examples
- Too little regularisation
- Too little training data

---

## The Bias–Variance Tradeoff

These two problems sit at opposite ends of a spectrum:

| | Underfitting | Good Fit | Overfitting |
|---|---|---|---|
| Bias | High | Low | Low |
| Variance | Low | Low | High |
| Training error | High | Low | Very low |
| Validation error | High | Low | High |
| Model complexity | Too simple | Just right | Too complex |

**Bias** — error from wrong assumptions. The model consistently misses in the same direction.

**Variance** — error from sensitivity to small fluctuations in training data. The model changes drastically when trained on a slightly different dataset.

The goal is to find the sweet spot with **low bias and low variance**.

```
Error
  |
  |  \          /
  |   \        /   ← validation error
  |    \      /
  |     \    /
  |      \  /      ← training error
  |       \/
  |
  +--------------------------- Model Complexity
  ↑ underfitting    ↑ overfitting
```

---

## Visualising All Three Cases (Regression)

### Underfitting — Linear fit on non-linear data

$$f(x) = w_1 x + b$$

The model cannot bend to follow the data.

### Good Fit — Quadratic

$$f(x) = w_1 x + w_2 x^2 + b$$

Captures the true shape without chasing noise.

### Overfitting — Very high-degree polynomial

$$f(x) = w_1 x + w_2 x^2 + \ldots + w_9 x^9 + b$$

Passes through every training point but oscillates wildly between them.

---

## Visualising All Three Cases (Classification)

### Underfitting

A straight decision boundary when the true boundary is curved. Many training points are misclassified.

### Good Fit

A smooth curve that separates the two classes correctly, with a natural boundary.

### Overfitting

A jagged, twisting boundary that perfectly separates all training points — including the noisy ones — but will fail on new data.

```
Underfit          Good fit          Overfit
  o | *             o  (*)            o ~*~ o
  o | *            o ( * )           ~o~   ~*~
  o | *             o  (*)           o  ~*~  o
  straight line    smooth circle    jagged boundary
```

---

## How to Fix Underfitting

1. **Add more features** — include more relevant input variables
2. **Add polynomial features** — e.g. $x^2$, $x_1 x_2$ to capture non-linear relationships
3. **Reduce regularisation** — decrease $\lambda$ to allow larger weights
4. **Use a more complex model** — more layers in a neural network, higher-degree polynomial

---

## How to Fix Overfitting

### 1. Get More Training Data

The most effective fix. More data makes it harder for the model to memorise — it must learn the actual pattern.

```
Small dataset → model memorises
Large dataset → model must generalise
```

### 2. Feature Selection

Remove features that are not relevant or are redundant. Fewer inputs means less opportunity to overfit.

- Manually select the most useful features
- Use algorithms (e.g. forward/backward selection) to pick automatically

### 3. Regularisation

Keep all features but **shrink the weights** of less important ones toward zero. This is the most common and practical approach.

Add a penalty term to the cost function:

$$J(\vec{w}, b) = \underbrace{\frac{1}{m} \sum_{i=1}^{m} L(f^{(i)}, y^{(i)})}_{\text{fit the data}} + \underbrace{\frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2}_{\text{keep weights small}}$$

- $\lambda = 0$: no regularisation, risk of overfitting
- $\lambda$ very large: all weights pushed toward 0, model becomes flat → underfitting
- $\lambda$ just right: model is flexible but smooth

### 4. Reduce Model Complexity

Use fewer polynomial features or a simpler model architecture.

---

## Regularisation In Depth

### What It Does

Regularisation does not eliminate features — it **reduces their influence** by penalising large weights. A weight of 0 means the feature has no effect; a large weight means the feature strongly influences the output.

### L2 Regularisation (Ridge)

$$\text{penalty} = \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

Squares the weights — large weights are penalised disproportionately. This **shrinks all weights** but rarely sets them exactly to zero. Most commonly used.

### L1 Regularisation (Lasso)

$$\text{penalty} = \frac{\lambda}{m} \sum_{j=1}^{n} |w_j|$$

Uses absolute values — tends to drive some weights **exactly to zero**, performing automatic feature selection.

### Choosing $\lambda$

| $\lambda$ | Effect |
|-----------|--------|
| Too small | Overfitting — weights grow large, model too complex |
| Too large | Underfitting — weights crushed to zero, model too simple |
| Just right | Weights are meaningful but controlled |

In practice, $\lambda$ is tuned using a **validation set** — train with several values of $\lambda$, pick the one with the lowest validation error.

---

## Diagnosing with Learning Curves

A **learning curve** plots training error and validation error as the training set size increases.

### Underfitting Pattern

```
Error
  |
  |-------- validation error (flat, high)
  |
  |-------- training error (flat, also high)
  |
  +--------------------------- Training set size
```

Both errors are high and flat — adding more data does not help much. The model is simply not expressive enough.

### Overfitting Pattern

```
Error
  |
  |-------- validation error (high)
  |          \
  |           \_______ gap
  |                   \_______ training error (low)
  |
  +--------------------------- Training set size
```

Large gap between training and validation error. Adding more data would help — the gap closes as training set grows.

---

## Key Terms

| Term | Meaning |
|------|---------|
| **Generalisation** | Ability to perform well on new, unseen data |
| **Underfitting** | Model too simple; high error on both training and validation |
| **Overfitting** | Model too complex; low training error but high validation error |
| **Bias** | Error from wrong model assumptions; causes underfitting |
| **Variance** | Error from sensitivity to training data; causes overfitting |
| **Regularisation** | Penalty on large weights to prevent overfitting |
| **$\lambda$** | Regularisation strength; controls bias–variance tradeoff |
| **L2 (Ridge)** | Regularisation using squared weights; shrinks all weights |
| **L1 (Lasso)** | Regularisation using absolute weights; drives some to zero |
| **Learning curve** | Plot of training vs validation error over training set size |
| **Validation set** | Held-out data used to tune hyperparameters and detect overfitting |
