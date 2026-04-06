---
title: "Introduction to Transformer"
date: 2026-04-05
description: "What the Transformer is, why it replaced RNNs, and how its core components work together."
categories:
    - Transformer
tags:
    - transformer
    - attention
    - deep-learning
    - NLP
draft: false
---

## What Is the Transformer?

The Transformer is a deep learning architecture introduced in the 2017 paper **"Attention Is All You Need"** by Google Brain. It completely changed the field of NLP and became the foundation of almost every modern AI model — BERT, GPT, Claude, LLaMA, and more.

Before the Transformer, sequence tasks (translation, text generation) were dominated by **RNNs (Recurrent Neural Networks)**. The Transformer replaced them almost entirely.

---

## Why Not RNNs?

RNNs process tokens **one at a time, left to right**. This created two major problems:

1. **Sequential processing** — you can't parallelize. Token 5 must wait for token 4, which must wait for token 3. Training is slow.
2. **Long-range dependency problem** — information from early tokens gets "forgotten" by the time the model reaches later tokens. Understanding "The cat that sat on the mat **was** hungry" requires connecting "cat" to "was" across many words — RNNs struggle with this.

The Transformer solves both problems with one idea: **Attention**.

---

## The Core Idea: Attention

Instead of processing tokens sequentially, the Transformer looks at **all tokens at once** and learns which tokens should pay attention to which other tokens.

For the sentence: *"The animal didn't cross the street because **it** was too tired"*

What does "it" refer to — the animal or the street? Attention allows the model to learn that "it" should attend strongly to "animal", giving the correct interpretation.

This is the key insight: **relationships between words don't depend on distance**. Attention handles long-range dependencies directly, in a single step.

---

## High-Level Architecture

The original Transformer has two parts:

```
Input Text (English)
      ↓
  [Encoder]
  - Multi-Head Self-Attention
  - Feed Forward Network
  - (repeated N times)
      ↓
   Context
      ↓
  [Decoder]
  - Masked Multi-Head Self-Attention
  - Cross-Attention (attends to Encoder output)
  - Feed Forward Network
  - (repeated N times)
      ↓
Output Text (German)
```

Modern LLMs (GPT, Claude) use only the **Decoder** part. Models like BERT use only the **Encoder** part.

---

## Key Components

### 1. Input Embedding

Words are converted into vectors (numbers) that the model can process. Each token becomes a high-dimensional vector that captures its meaning.

### 2. Positional Encoding

Attention has no built-in sense of order — it looks at all tokens simultaneously. Positional encoding adds information about **where each token is** in the sequence, so the model knows token 1 comes before token 2.

### 3. Self-Attention

The core mechanism. For each token, self-attention computes:
- **Query (Q)** — what this token is looking for
- **Key (K)** — what this token offers to others
- **Value (V)** — the actual information this token carries

The attention score between two tokens is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $QK^T$ — dot product measures how much two tokens should attend to each other
- $\sqrt{d_k}$ — scaling factor to prevent the dot products from getting too large
- softmax — converts scores into probabilities (all attention weights sum to 1)
- multiply by $V$ — weighted sum of values based on attention scores

### 4. Multi-Head Attention

Instead of running attention once, the Transformer runs it **multiple times in parallel** with different learned projections. Each "head" can attend to different aspects of the relationships between tokens — one head might focus on syntax, another on semantics.

The outputs of all heads are concatenated and projected back:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

### 5. Feed Forward Network

After attention, each token passes through a small, independent feed-forward network (same weights for every position):

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

This adds non-linearity and gives the model capacity to transform representations beyond just attention.

### 6. Residual Connections + Layer Normalization

After each sub-layer (attention or FFN), the input is **added back** to the output:

$$\text{output} = \text{LayerNorm}(x + \text{sublayer}(x))$$

This prevents gradients from vanishing in deep networks and makes training much more stable.

---

## Encoder vs Decoder vs Both

| Architecture | Examples | Use Case |
|---|---|---|
| Encoder only | BERT, embedding models | Understanding, classification, search |
| Decoder only | GPT, Claude, LLaMA | Text generation, chat, coding |
| Encoder + Decoder | Original Transformer, T5 | Translation, summarization |

---

## Why the Transformer Won

| | RNN | Transformer |
|---|---|---|
| Processing | Sequential (slow) | Parallel (fast) |
| Long-range dependencies | Struggles | Handles directly via attention |
| Scalability | Hard to scale | Scales extremely well |
| Training speed | Slow | Much faster |

The Transformer's ability to parallelize training meant it could be trained on vastly more data, which turned out to be the key to building powerful language models.

---

## What Came After

The Transformer architecture spawned an entire family of models:

- **2018** — BERT (Google): Encoder-only, bidirectional, revolutionized NLP benchmarks
- **2018** — GPT (OpenAI): Decoder-only, generative, scaled up to GPT-2, GPT-3, GPT-4
- **2020** — T5 (Google): Encoder-Decoder, frames everything as text-to-text
- **2022+** — LLaMA (Meta), Claude (Anthropic), Gemini (Google): Large-scale Decoder-only models

Everything starts from the same 2017 paper. Understanding the Transformer is the foundation for understanding all of modern AI.
