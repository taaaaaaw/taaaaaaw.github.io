---
title: "Cost Function for Logistic Regression"
date: 2026-03-25
description: "A detailed breakdown of why logistic regression needs its own cost function, how log loss works, and why it guarantees gradient descent finds the global minimum."
categories:
    - Machine-Learning
tags:
    - supervised-learning
    - classification
    - logistic-regression
    - cost-function
draft: false
---

## Recap: What Is a Cost Function?

A cost function $J(\vec{w}, b)$ measures **how wrong the model's predictions are** across all training examples. The goal of training is to find the values of $\vec{w}$ and $b$ that minimise $J$.

For logistic regression, the model output is:

$$f_{\vec{w},b}(\vec{x}) = \frac{1}{1 + e^{-(\vec{w} \cdot \vec{x} + b)}}$$

This outputs a probability between 0 and 1.

---

## Why Not Use Squared Error?

In linear regression we used the **squared error cost**:

$$J(\vec{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f^{(i)} - y^{(i)} \right)^2$$

Can we use this for logistic regression? Technically yes, but there is a serious problem.

### The Non-Convexity Problem

When $f$ is a linear function, the squared error cost is a **convex** bowl — one global minimum, and gradient descent always finds it.

When $f$ is the sigmoid function, the squared error cost becomes **non-convex** — it has many local minima. Gradient descent can get trapped in one of these and never reach the true minimum.

```
Convex (linear regression)     Non-convex (sigmoid + squared error)
         J                              J
         |                              |
         |    ___                       | /\/\  /\
         |   /   \                      |/    \/  \___
         |  /     \                     |
         | /       \___                 |
         +-------------- w              +-------------- w
           ↑                              ↑ ↑ ↑
        one minimum               many local minima
```

We need a cost function that is **convex** when used with the sigmoid output.

---

## The Loss Function (Single Example)

Before defining the full cost, we define the **loss** $L$ for a single training example $(x^{(i)}, y^{(i)})$:

$$L(f^{(i)}, y^{(i)}) = \begin{cases} -\log(f^{(i)}) & \text{if } y^{(i)} = 1 \\ -\log(1 - f^{(i)}) & \text{if } y^{(i)} = 0 \end{cases}$$

### When y = 1: Loss = $-\log(f)$

| $f$ (prediction) | $-\log(f)$ (loss) | Interpretation |
|------------------|-------------------|----------------|
| 1.0 | 0 | Perfect prediction, no penalty |
| 0.9 | 0.105 | Very small loss |
| 0.5 | 0.693 | Moderate loss |
| 0.1 | 2.303 | Large loss |
| → 0 | → ∞ | Predicted 0 when truth is 1, infinite penalty |

As $f \to 1$: loss $\to 0$ ✅
As $f \to 0$: loss $\to \infty$ ✅

```
Loss
  |
∞ |*
  | *
  |  *
  |   *
  |    **
  |      ***
  |          ******
0 +-------------------
  0         0.5       1   → f (predicted probability)
```

### When y = 0: Loss = $-\log(1 - f)$

| $f$ (prediction) | $-\log(1-f)$ (loss) | Interpretation |
|------------------|---------------------|----------------|
| 0.0 | 0 | Perfect prediction, no penalty |
| 0.1 | 0.105 | Very small loss |
| 0.5 | 0.693 | Moderate loss |
| 0.9 | 2.303 | Large loss |
| → 1 | → ∞ | Predicted 1 when truth is 0, infinite penalty |

As $f \to 0$: loss $\to 0$ ✅
As $f \to 1$: loss $\to \infty$ ✅

---

## The Simplified Loss Formula

The two cases can be merged into a single expression:

$$L(f^{(i)}, y^{(i)}) = -y^{(i)} \log(f^{(i)}) - (1 - y^{(i)}) \log(1 - f^{(i)})$$

**Why this works:**

- When $y^{(i)} = 1$: the second term vanishes → $L = -\log(f^{(i)})$ ✅
- When $y^{(i)} = 0$: the first term vanishes → $L = -\log(1 - f^{(i)})$ ✅

This single formula covers both cases cleanly.

---

## The Full Cost Function

The cost $J$ is the **average loss** over all $m$ training examples:

$$J(\vec{w}, b) = \frac{1}{m} \sum_{i=1}^{m} L(f^{(i)}, y^{(i)})$$

Expanded:

$$\boxed{J(\vec{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f^{(i)}) + (1 - y^{(i)}) \log(1 - f^{(i)}) \right]}$$

This is called **log loss** or **binary cross-entropy**.

### Properties

- $J \geq 0$ always
- $J = 0$ only when the model perfectly predicts every training example
- The function is **convex** — one global minimum, gradient descent is guaranteed to find it

---

## Where Does This Formula Come From?

The log loss is derived from **maximum likelihood estimation (MLE)**.

The idea: find the parameters $\vec{w}$ and $b$ that make the observed training labels **most probable** under the model.

For binary labels, the probability of a single example is:

$$P(y \mid x) = f^y \cdot (1-f)^{1-y}$$

- If $y = 1$: $P = f$
- If $y = 0$: $P = 1 - f$

To maximise the joint probability over all training examples (assuming independence):

$$\max \prod_{i=1}^{m} f^{y^{(i)}} \cdot (1-f^{(i)})^{1-y^{(i)}}$$

Taking the log (to turn the product into a sum) and negating (to turn maximisation into minimisation):

$$\min -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f^{(i)}) + (1-y^{(i)}) \log(1-f^{(i)}) \right]$$

This is exactly the log loss cost function.

---

## Gradient Descent with Log Loss

Once we have the cost function, we minimise it with gradient descent:

$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

The partial derivatives of the log loss work out to:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left( f^{(i)} - y^{(i)} \right) x_j^{(i)}$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( f^{(i)} - y^{(i)} \right)$$

These look identical to the linear regression derivatives — but remember $f^{(i)}$ is the **sigmoid output**, not a raw linear value.

---

## Regularised Cost Function

To prevent overfitting, we add an $L_2$ regularisation term that penalises large weights:

$$J(\vec{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f^{(i)}) + (1 - y^{(i)}) \log(1 - f^{(i)}) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

- $\lambda > 0$ is the regularisation parameter
- Large $\lambda$ → heavily penalises large weights → simpler model → risk of underfitting
- Small $\lambda$ → barely penalises → complex model → risk of overfitting
- $b$ is excluded from regularisation by convention

The regularised gradient for $w_j$:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} \left( f^{(i)} - y^{(i)} \right) x_j^{(i)} + \frac{\lambda}{m} w_j$$

---

## Summary: Squared Error vs Log Loss

| | Squared Error | Log Loss |
|---|---|---|
| Used for | Linear regression | Logistic regression |
| Shape with sigmoid | Non-convex ❌ | Convex ✅ |
| Local minima | Many | None (one global minimum) |
| Gradient descent | May get stuck | Always converges |
| Origin | Intuitive | Maximum likelihood estimation |

---

## Key Terms

| Term | Meaning |
|------|---------|
| **Loss** $L$ | Error for a single training example |
| **Cost** $J$ | Average loss over all training examples |
| **Log loss / binary cross-entropy** | The convex cost function used for logistic regression |
| **Convex** | Bowl-shaped — only one minimum, gradient descent always finds it |
| **Non-convex** | Multiple local minima — gradient descent may get stuck |
| **Maximum likelihood estimation** | The statistical principle behind log loss |
| **Regularisation** | Extra penalty term to keep weights small and prevent overfitting |
| **$\lambda$** | Regularisation strength parameter |
