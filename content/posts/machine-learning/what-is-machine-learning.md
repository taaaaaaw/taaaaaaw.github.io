---
title: "What Is Machine Learning?"
date: 2026-03-11
description: ""
categories:
    - Machine-Learning
tags:
    - beginner
draft: false
---

## What is machine learning?

> Machine learning is the field of study that gives computers the ability to learn without being explicitly programmed.

## Main Type of Machine Learning

### Supervised Learning

Supervised learning is the most common type of machine learning. The model is trained on a **labeled dataset**, meaning every input comes with a corresponding correct output. The goal is for the model to learn the mapping from input to output so it can predict the correct answer on new, unseen data.

**How it works:**
1. You provide the model with training data — pairs of (input, correct output)
2. The model makes predictions and compares them to the correct answers
3. It adjusts itself to minimize the error
4. After training, it can predict outputs for new inputs

**Common examples:**
- **Classification** — predicting a category (e.g., is this email spam or not spam?)
- **Regression** — predicting a continuous value (e.g., what will the house price be?)

**Real-world use cases:**
- Image recognition (labeling photos as "cat" or "dog")
- Spam filtering
- Medical diagnosis

---

### Unsupervised Learning

Unsupervised learning works with **unlabeled data** — there are no correct answers given. Instead, the model tries to find hidden patterns or structure in the data on its own.

**How it works:**
1. You provide the model with raw data, no labels
2. The model explores the data and finds natural groupings or structures
3. It surfaces patterns that weren't explicitly defined

**Common examples:**
- **Clustering** — grouping similar data points together (e.g., grouping customers by purchase behavior)
- **Dimensionality reduction** — compressing data while keeping its structure (e.g., PCA)

**Real-world use cases:**
- Customer segmentation
- Anomaly detection (finding unusual patterns in network traffic)
- Topic modeling in documents

---

### Key Difference

| | Supervised | Unsupervised |
|---|---|---|
| Data | Labeled | Unlabeled |
| Goal | Predict known output | Discover hidden structure |
| Example | Spam detection | Customer segmentation |



