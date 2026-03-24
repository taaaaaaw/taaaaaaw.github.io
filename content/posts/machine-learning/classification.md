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

## What's Next

The standard algorithm for binary classification is **logistic regression** — it uses the sigmoid function to output probabilities and a decision boundary to separate classes.

For a full breakdown of how logistic regression works, see the [Logistic Regression](/posts/machine-learning/logistic-regression/) post.
