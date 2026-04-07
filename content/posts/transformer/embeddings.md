---
title: "Embeddings in Language Models"
date: 2026-04-07
description: "A detailed explanation of embeddings — how words are converted into vectors, what makes a good embedding, and how they are used inside language models."
categories:
    - Transformer
tags:
    - transformer
    - embeddings
    - language-model
    - nlp
draft: false
---

## The Problem: Computers Cannot Read Words

Neural networks work with numbers, not text. To feed a sentence into a model, every word (or token) must be converted into a number — or better, a **vector of numbers**.

The naive approach is **one-hot encoding**:

```
Vocabulary: ["cat", "dog", "fish", "bird"]

"cat"  → [1, 0, 0, 0]
"dog"  → [0, 1, 0, 0]
"fish" → [0, 0, 1, 0]
"bird" → [0, 0, 0, 1]
```

This works but has major problems:
- For a vocabulary of 50,000 words, each vector has 50,000 dimensions — mostly zeros
- Every word is equally distant from every other word — "cat" and "dog" are no more similar than "cat" and "airplane"
- No semantic information is captured at all

**Embeddings** solve this.

---

## What Is an Embedding?

An embedding maps each token to a **dense, low-dimensional vector** (typically 128 to 4096 dimensions depending on the model).

```
One-hot:   "cat" → [1, 0, 0, 0, 0, ..., 0]   (50,000 dims, mostly zeros)
Embedding: "cat" → [0.2, -0.5, 0.8, 0.1, ...]  (512 dims, all meaningful)
```

The key property: **semantically similar words have similar vectors**.

```
Similarity in embedding space:

  "cat" ──────── "dog"        (close — both animals, pets)
     \
      \──── "kitten"          (very close — cat-related)

  "airplane" ──────────────── (far from all of the above)
```

---

## How Embeddings Are Learned

Embeddings are not hand-crafted — they are **learned during training** from large amounts of text.

The embedding for each token is a row in a matrix called the **embedding matrix** $E$:

$$E \in \mathbb{R}^{V \times d}$$

Where:
- $V$ = vocabulary size (e.g. 50,000)
- $d$ = embedding dimension (e.g. 512)

To look up the embedding for token $i$, multiply the one-hot vector $\mathbf{e}_i$ by $E$:

$$\text{embedding} = \mathbf{e}_i \cdot E$$

In practice this is just a **table lookup** — grab row $i$ from $E$.

The values in $E$ start random and are updated by gradient descent during training, just like any other weight matrix.

---

## What Do Embedding Dimensions Represent?

No single dimension has a human-interpretable meaning assigned to it. The model figures out its own internal representation. However, after training, structure emerges.

### The Famous Example: Word2Vec Arithmetic

$$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$

The embedding space has implicitly encoded a "gender" direction and a "royalty" direction. Arithmetic in vector space corresponds to semantic relationships.

### Other relationships that emerge

| Analogy | Vector arithmetic |
|---------|------------------|
| Paris is to France as Rome is to Italy | $\vec{\text{Paris}} - \vec{\text{France}} \approx \vec{\text{Rome}} - \vec{\text{Italy}}$ |
| walked : walk = ran : run | Tense direction captured |
| big : bigger = small : smaller | Comparative direction captured |

---

## Embedding Similarity

The similarity between two embeddings is measured using **cosine similarity**:

$$\text{similarity}(a, b) = \frac{a \cdot b}{\|a\| \|b\|}$$

- Result of 1 → identical direction (very similar meaning)
- Result of 0 → orthogonal (unrelated)
- Result of -1 → opposite directions (antonyms)

This is why semantic search works — instead of matching keywords, you find vectors that point in similar directions.

---

## Tokenization: What Gets Embedded?

Modern language models do not embed whole words — they embed **tokens**, which are chunks of text produced by a tokenizer (e.g. Byte Pair Encoding, BPE).

```
"unhappiness" → ["un", "happi", "ness"]   (3 tokens)
"cat"         → ["cat"]                    (1 token)
"ChatGPT"     → ["Chat", "G", "PT"]       (3 tokens)
```

Each token gets its own embedding vector. This handles:
- Unknown words (break them into known sub-pieces)
- Morphology (prefixes, suffixes share embeddings across words)
- Efficiency (smaller vocabulary than full words)

A typical vocabulary size is **30,000–100,000 tokens**.

---

## Embeddings Inside a Transformer

In a transformer model, embeddings are the very first step:

```
Input text
    ↓
Tokenizer        "The cat sat" → [464, 3797, 3332]
    ↓
Embedding lookup  [464]  → [0.2, -0.5, 0.8, ...]    (d-dimensional vector)
                  [3797] → [-0.1, 0.3, 0.6, ...]
                  [3332] → [0.5, 0.1, -0.2, ...]
    ↓
Positional encoding added
    ↓
Transformer layers (attention, feedforward)
    ↓
Output
```

The embedding layer converts a sequence of token IDs into a matrix of shape $(T, d)$ — where $T$ is the sequence length and $d$ is the embedding dimension.

---

## Positional Embeddings

Standard embeddings have no sense of order — "cat sat the" and "the cat sat" would produce the same set of vectors, just in different positions.

To encode position, a **positional embedding** is added to each token embedding:

$$\text{input to transformer} = \text{token embedding} + \text{positional embedding}$$

### Sinusoidal (original Transformer)

The original Transformer used fixed sinusoidal functions:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Different frequencies encode different positions, allowing the model to distinguish token order.

### Learned positional embeddings

Many modern models (BERT, GPT) simply **learn** positional embeddings as additional parameters — each position $0, 1, 2, \ldots$ gets its own trainable vector.

### RoPE (Rotary Position Embedding)

Used in LLaMA, Mistral, and most modern LLMs. Instead of adding positional information, RoPE **rotates** the query and key vectors in attention by an angle proportional to position. This handles longer sequences more gracefully.

---

## Output Embeddings (Unembedding)

At the output of a transformer, the model needs to predict the next token. This is done by multiplying the final hidden state by the **unembedding matrix** (often the transpose of the input embedding matrix $E^T$):

$$\text{logits} = h \cdot E^T \in \mathbb{R}^V$$

Each logit corresponds to one vocabulary token. Softmax converts these to probabilities:

$$P(\text{next token} = i) = \text{softmax}(\text{logits})_i$$

The model samples or argmax-selects from this distribution to generate the next token.

---

## Embedding Dimension and Model Size

The embedding dimension $d$ is one of the most important hyperparameters. Larger $d$ → more representational capacity, but more parameters and compute.

| Model | Embedding dim $d$ | Vocabulary $V$ | Embedding parameters |
|-------|------------------|---------------|---------------------|
| GPT-2 small | 768 | 50,257 | ~38M |
| GPT-2 large | 1,280 | 50,257 | ~64M |
| GPT-3 | 12,288 | 50,257 | ~617M |
| LLaMA 3 8B | 4,096 | 128,256 | ~525M |

The embedding matrix alone can account for a significant fraction of a model's total parameters.

---

## Key Terms

| Term | Meaning |
|------|---------|
| **Embedding** | A dense vector representation of a token |
| **Embedding matrix** $E$ | Lookup table of shape $(V, d)$ mapping token IDs to vectors |
| **Token** | The unit of text that gets embedded (word, sub-word, or character) |
| **Vocabulary** $V$ | The set of all possible tokens the model knows |
| **Embedding dimension** $d$ | The size of each embedding vector |
| **Cosine similarity** | Measure of angle between two vectors; used to compare semantic similarity |
| **Positional embedding** | Additional vector added to encode the position of a token in a sequence |
| **Tokenizer** | Converts raw text into a sequence of token IDs |
| **Unembedding** | The final linear layer that converts hidden states back to vocabulary probabilities |
| **RoPE** | Rotary Position Embedding — encodes position by rotating attention vectors |
