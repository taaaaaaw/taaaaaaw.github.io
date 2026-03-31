---
title: "Vectorization in Deep Learning"
date: 2026-03-31
description: "A detailed explanation of vectorization — what it is, why it matters, and how it applies to logistic regression and gradient descent."
categories:
    - Deep-Learning
tags:
    - deep-learning
    - vectorization
    - logistic-regression
    - numpy
draft: false
---

## What Is Vectorization?

Vectorization is the practice of **replacing explicit for-loops with matrix and vector operations**. Instead of processing one element at a time, you process the entire dataset in one operation.

Modern CPUs and GPUs are optimised to perform these operations in parallel — vectorized code can be **100x or more faster** than equivalent loop-based code.

---

## Non-Vectorized vs Vectorized

### Example: Computing $z = \vec{w} \cdot \vec{x} + b$

**Without vectorization (slow):**

```python
z = 0
for i in range(len(w)):
    z += w[i] * x[i]
z += b
```

This loops through each element one by one.

**With vectorization (fast):**

```python
import numpy as np
z = np.dot(w, x) + b
```

`np.dot` computes the entire dot product in one call using optimised BLAS routines that run in parallel under the hood.

### Timing Comparison

```python
import numpy as np
import time

n = 1_000_000
w = np.random.randn(n)
x = np.random.randn(n)

# Non-vectorized
start = time.time()
z = 0
for i in range(n):
    z += w[i] * x[i]
print(f"Loop: {(time.time() - start) * 1000:.2f} ms")

# Vectorized
start = time.time()
z = np.dot(w, x)
print(f"Vectorized: {(time.time() - start) * 1000:.2f} ms")

# Typical output:
# Loop:       400.00 ms
# Vectorized:   1.50 ms
```

---

## More Vectorization Examples

### Applying a Function to Every Element

**Without vectorization:**

```python
u = np.zeros(n)
for i in range(n):
    u[i] = math.exp(v[i])
```

**Vectorized:**

```python
u = np.exp(v)   # applies exp to every element at once
```

NumPy has vectorized versions of most mathematical functions:

| Operation | Loop version | Vectorized |
|-----------|-------------|------------|
| Exponential | `math.exp(v[i])` | `np.exp(v)` |
| Log | `math.log(v[i])` | `np.log(v)` |
| Absolute value | `abs(v[i])` | `np.abs(v)` |
| Square | `v[i] ** 2` | `v ** 2` |
| Max with 0 (ReLU) | `max(0, v[i])` | `np.maximum(0, v)` |

### Matrix-Vector Multiplication

Suppose you have a matrix $A$ (shape $m \times n$) and a vector $x$ (shape $n$). Computing $u = Ax$:

**Without vectorization:**

```python
u = np.zeros(m)
for i in range(m):
    for j in range(n):
        u[i] += A[i][j] * x[j]
```

**Vectorized:**

```python
u = np.dot(A, x)    # or A @ x
```

---

## Vectorizing Logistic Regression

In logistic regression we have $m$ training examples, each with $n$ features.

### Setup

Stack all training examples into matrices:

$$X = \begin{bmatrix} | & | & & | \\ x^{(1)} & x^{(2)} & \cdots & x^{(m)} \\ | & | & & | \end{bmatrix} \quad \text{shape: } (n, m)$$

$$Y = \begin{bmatrix} y^{(1)} & y^{(2)} & \cdots & y^{(m)} \end{bmatrix} \quad \text{shape: } (1, m)$$

$$W = \begin{bmatrix} w_1 \\ w_2 \\ \vdots \\ w_n \end{bmatrix} \quad \text{shape: } (n, 1)$$

### Forward Pass — Non-Vectorized

```python
# For each of m training examples:
for i in range(m):
    z[i] = np.dot(w, x[:, i]) + b
    a[i] = sigmoid(z[i])
```

### Forward Pass — Vectorized

```python
Z = np.dot(W.T, X) + b    # shape: (1, m)
A = sigmoid(Z)             # shape: (1, m)
```

One line replaces the entire loop. `np.dot(W.T, X)` computes $w^T x^{(i)}$ for **all $m$ examples simultaneously**.

The `+ b` uses **broadcasting** — NumPy automatically expands the scalar $b$ to match the shape $(1, m)$.

---

## Vectorizing Logistic Regression's Gradient Output

After the forward pass, we compute the cost and then the gradients needed for gradient descent.

### Cost

$$J = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(a^{(i)}) + (1 - y^{(i)}) \log(1 - a^{(i)}) \right]$$

```python
J = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A))
```

### Gradients — Non-Vectorized

```python
dw = np.zeros((n, 1))
db = 0

for i in range(m):
    dz = a[i] - y[i]
    for j in range(n):
        dw[j] += x[j][i] * dz
    db += dz

dw /= m
db /= m
```

### Gradients — Vectorized

```python
dZ = A - Y                          # shape: (1, m)
dW = (1 / m) * np.dot(X, dZ.T)     # shape: (n, 1)
db = (1 / m) * np.sum(dZ)          # scalar
```

Three lines replace the double for-loop entirely.

### Why This Works

$$\frac{\partial J}{\partial W} = \frac{1}{m} X \cdot dZ^T$$

This is a matrix multiplication that computes the gradient **for all weights simultaneously**, across all training examples.

---

## Full Vectorized Logistic Regression

Putting it all together — one complete iteration of gradient descent:

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_step(X, Y, W, b, alpha):
    m = X.shape[1]

    # Forward pass
    Z = np.dot(W.T, X) + b       # (1, m)
    A = sigmoid(Z)                # (1, m)

    # Cost
    J = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    # Backward pass (gradients)
    dZ = A - Y                    # (1, m)
    dW = (1 / m) * np.dot(X, dZ.T)  # (n, 1)
    db = (1 / m) * np.sum(dZ)    # scalar

    # Gradient descent update
    W = W - alpha * dW
    b = b - alpha * db

    return W, b, J
```

**No for-loops at all.** This scales to any number of training examples $m$ and any number of features $n$ without changing the code.

---

## Broadcasting

Broadcasting is what allows NumPy to add a scalar to a matrix or a vector to a matrix without explicit loops.

### Rules

NumPy compares shapes element-wise from the right. Two dimensions are compatible if:
- They are equal, or
- One of them is 1

Try it yourself — run the code below directly in your browser:

{{< pyrunner >}}
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape (2, 3)

# ── Example 1: Scalar broadcast ──────────────────────────────
# 100 is added to every element
print("=" * 40)
print("Example 1: Scalar Broadcast")
print("=" * 40)
print(f"A (shape {A.shape}):\n{A}")
print(f"\nA + 100:\n{A + 100}")

# ── Example 2: Row vector broadcast ──────────────────────────
# b (shape 3,) is added to each row of A
b = np.array([10, 20, 30])  # shape (3,)
print("\n" + "=" * 40)
print("Example 2: Row Vector Broadcast")
print("=" * 40)
print(f"b (shape {b.shape}): {b}")
print(f"\nA + b:\n{A + b}")

# ── Example 3: Column vector broadcast ───────────────────────
# c (shape 2,1) is added to each column of A
c = np.array([[10],
              [20]])         # shape (2, 1)
print("\n" + "=" * 40)
print("Example 3: Column Vector Broadcast")
print("=" * 40)
print(f"c (shape {c.shape}):\n{c}")
print(f"\nA + c:\n{A + c}")

# ── Example 4: Logistic regression ───────────────────────────
# bias scalar broadcasts across all m training examples
W    = np.array([[0.5], [1.0], [-0.3]])   # shape (3, 1)
X    = np.array([[1, 2, 3, 4],            # shape (3, 4)
                 [0, 1, 0, 1],
                 [2, 0, 1, 3]])
bias = 2.0
Z    = np.dot(W.T, X) + bias              # bias → broadcasts to (1, 4)
print("\n" + "=" * 40)
print("Example 4: Logistic Regression Bias")
print("=" * 40)
print(f"W.T shape: {W.T.shape},  X shape: {X.shape}")
print(f"Z = W.T @ X + bias  →  shape {Z.shape}")
print(f"\nZ:\n{Z.round(2)}")
{{< /pyrunner >}}

In logistic regression, `np.dot(W.T, X) + b` works because $b$ is a scalar — NumPy broadcasts it across all $m$ columns.

---

## Why Vectorization Matters for Deep Learning

In a neural network:
- You might have $m = 1{,}000{,}000$ training examples
- Each layer might have $n = 1{,}000$ neurons
- The network might have $L = 100$ layers

A single forward pass requires billions of multiplications. Without vectorization, this would take hours per epoch. With vectorization on a GPU — seconds.

Deep learning at scale is only possible because:
1. **Vectorized operations** replace loops
2. **GPUs** execute these operations in parallel across thousands of cores
3. **Frameworks** (PyTorch, TensorFlow) handle this automatically

---

## Key Terms

| Term | Meaning |
|------|---------|
| **Vectorization** | Replacing for-loops with matrix/vector operations |
| **Broadcasting** | NumPy automatically expanding shapes to make operations compatible |
| **`np.dot`** | NumPy dot product / matrix multiplication |
| **$X$ matrix** | All training examples stacked as columns; shape $(n, m)$ |
| **$dZ$** | Gradient of cost with respect to $Z$; equals $A - Y$ for logistic regression |
| **$dW$** | Gradient of cost with respect to $W$; computed as $\frac{1}{m} X \cdot dZ^T$ |
| **SIMD** | Single Instruction Multiple Data — CPU/GPU hardware that enables parallel operations |
