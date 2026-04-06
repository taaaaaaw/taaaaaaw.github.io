---
title: "Parameters vs Hyperparameters"
date: 2026-04-05
description: "The difference between parameters and hyperparameters in deep learning, and why the distinction matters."
categories:
    - Deep-Learning
tags:
    - neural-networks
    - training
draft: false
---

## Parameters

Parameters are the values that the model **learns automatically** during training. You never set them manually — gradient descent updates them for you after every iteration.

In a neural network, the parameters are:

- $W^{[l]}$ — the weight matrix for layer $l$
- $b^{[l]}$ — the bias vector for layer $l$

Every layer has its own $W$ and $b$. They start at zero (or small random values) and get updated through forward and backward propagation until the model converges.

---

## Hyperparameters

Hyperparameters are values that **you set before training begins**. The model does not learn them — they control how the learning process works.

Common hyperparameters in deep learning:

| Hyperparameter | Description |
|---|---|
| Learning rate $\alpha$ | How big each gradient descent step is |
| Number of iterations | How many times to run forward + backward |
| Number of layers $L$ | How deep the network is |
| Number of units $n^{[l]}$ | How wide each layer is |
| Activation function $g^{[l]}$ | Which activation to use per layer |
| Mini-batch size | How many examples per gradient update |
| Momentum, Adam parameters | Advanced optimizer settings |
| Regularization ($\lambda$) | Controls overfitting |

---

## Why the Distinction Matters

Parameters and hyperparameters affect the model in completely different ways:

- **Parameters** determine what the model has learned — they encode the knowledge extracted from the training data
- **Hyperparameters** determine how the model learns — they control the training process itself

Changing a hyperparameter (like the learning rate or number of layers) changes how $W$ and $b$ evolve during training, which in turn changes the final model you get.

---

## How to Choose Hyperparameters

Unlike parameters, there is no formula that gives you the optimal hyperparameters. The standard approach is **empirical** — you try different values and see what works:

1. Start with a reasonable default (e.g. $\alpha = 0.01$, 2-3 layers)
2. Train the model and evaluate performance
3. Adjust hyperparameters based on results
4. Repeat

This process is called **hyperparameter tuning**, and it's one of the most important practical skills in deep learning.

A useful rule of thumb:
- If the model is **underfitting** → try more layers, more units, train longer
- If the model is **overfitting** → try regularization, fewer layers, less units

---

## Summary

| | Parameters | Hyperparameters |
|---|---|---|
| Examples | $W^{[l]}$, $b^{[l]}$ | $\alpha$, $L$, $n^{[l]}$, $g^{[l]}$ |
| Who sets them | Learned automatically by gradient descent | You set them manually |
| When they change | Updated every iteration during training | Fixed before training starts |
| How to find the best value | Gradient descent handles it | Trial and error (hyperparameter tuning) |
