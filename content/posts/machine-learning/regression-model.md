---
title: "Linear Regression"
date: 2026-03-11
description: "A beginner-friendly breakdown of linear regression — what it is, how it works, and the key terminology you need to know."
categories:
    - Machine-Learning
tags:
    - supervised-learning
    - regression
draft: false
---

## What Is Linear Regression?

Linear regression is one of the simplest and most fundamental algorithms in machine learning. The goal is to find a **straight line** that best fits the relationship between an input variable and an output variable.

In plain terms: given some data points, linear regression draws the best-fitting line through them so you can **predict new values**.

---

## A Simple Example

Imagine you want to predict a person's weight based on their height. You collect data from 100 people and plot it:

- X-axis (input): height
- Y-axis (output): weight

Linear regression finds the line that best describes the relationship. Once you have that line, you can plug in any height and get a predicted weight.

Another common example: predicting house prices based on square footage. The bigger the house, the higher the price — linear regression captures that relationship.

---

## How It Works

The model learns a linear equation of the form:

```
y = mx + b
```

Where:
- `y` = predicted output (e.g., house price)
- `x` = input feature (e.g., square footage)
- `m` = slope (how much y changes per unit of x)
- `b` = intercept (the value of y when x = 0)

During training, the model adjusts `m` and `b` to minimize the difference between its predictions and the actual values.

---

## Key Terminology

### Feature (Input Variable)
The input data used to make a prediction. In the house price example, square footage is the feature. You can have multiple features (e.g., size + number of bedrooms), which is called **multiple linear regression**.

### Label (Target Variable)
The output you're trying to predict — in this case, the house price.

### Training Data
The dataset the model learns from. It contains pairs of (feature, label) that the model uses to find the best-fit line.

### Loss Function
A measure of how wrong the model's predictions are. The most common one for regression is **Mean Squared Error (MSE)**:

```
MSE = (1/n) * Σ(predicted - actual)²
```

The smaller the loss, the better the model.

### Gradient Descent
The optimization algorithm used to minimize the loss. It iteratively adjusts `m` and `b` in the direction that reduces the error.

### Overfitting
When the model fits the training data too well — including the noise — and performs poorly on new data. A model that memorizes rather than generalizes.

### Underfitting
The opposite: the model is too simple and fails to capture the underlying pattern. A straight line trying to fit a curved relationship, for example.

### R² Score (Coefficient of Determination)
A metric that tells you how well the model explains the variance in the data. Ranges from 0 to 1 — closer to 1 means a better fit.

---

## Limitations

Linear regression assumes the relationship between input and output is linear. In real-world data, this is often not the case. When the relationship is more complex, you'll need more powerful models like polynomial regression, decision trees, or neural networks.
