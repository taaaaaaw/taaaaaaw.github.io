---
title: "What Is Deep Learning?"
date: 2026-03-26
description: "A beginner-friendly introduction to deep learning — what it is, how it relates to machine learning and AI, and why it has become so powerful."
categories:
    - Deep-Learning
tags:
    - deep-learning
    - neural-network
    - introduction
draft: false
---

## The Big Picture

Before diving into deep learning, it helps to see where it sits within the broader landscape:

```
Artificial Intelligence
└── Machine Learning
    └── Deep Learning
```

- **Artificial Intelligence (AI):** Any technique that enables machines to mimic human behaviour
- **Machine Learning (ML):** A subset of AI where machines learn from data rather than being explicitly programmed
- **Deep Learning (DL):** A subset of ML that uses **neural networks with many layers** to learn complex patterns automatically

---

## What Is Deep Learning?

Deep learning is a class of machine learning algorithms that use **artificial neural networks** inspired by the structure of the human brain. The word "deep" refers to the **many layers** of these networks — each layer learns increasingly abstract representations of the data.

Instead of manually engineering features (like deciding "house size" and "number of rooms" matter for predicting price), deep learning models **learn the features automatically** from raw data.

---

## The Neuron: Building Block of a Neural Network

A single artificial neuron takes some inputs, multiplies each by a weight, sums them up, adds a bias, and passes the result through an **activation function**:

$$a = g(w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b) = g(\vec{w} \cdot \vec{x} + b)$$

Where:
- $x_1, x_2, \ldots, x_n$ — input features
- $w_1, w_2, \ldots, w_n$ — weights (learned during training)
- $b$ — bias (learned during training)
- $g$ — activation function (e.g. ReLU, sigmoid)
- $a$ — the neuron's output (called **activation**)

This is essentially logistic regression — one neuron is one logistic regression unit.

---

## Neural Networks: Stacking Neurons into Layers

A neural network is formed by connecting many neurons in **layers**:

```
Input Layer    Hidden Layers    Output Layer

  x₁ ──┐
        ├──► [neuron] ──┐
  x₂ ──┤                ├──► [neuron] ──┐
        ├──► [neuron] ──┤                ├──► [neuron] ──► ŷ
  x₃ ──┤                ├──► [neuron] ──┘
        ├──► [neuron] ──┘
  x₄ ──┘
```

### Input Layer
Receives the raw features — no computation happens here. Each node represents one feature.

### Hidden Layers
The layers between input and output. Each neuron takes the outputs of the previous layer as its inputs and learns to detect a particular pattern or feature. The more hidden layers, the "deeper" the network.

### Output Layer
Produces the final prediction:
- **Regression:** one neuron, outputs a continuous number
- **Binary classification:** one neuron with sigmoid, outputs a probability
- **Multi-class classification:** multiple neurons with softmax, outputs a probability for each class

---

## What Does Each Layer Learn?

This is one of the most important ideas in deep learning. Each layer learns increasingly **abstract representations**:

### Image Recognition Example

| Layer | What It Detects |
|-------|----------------|
| Layer 1 | Edges, corners, colour gradients |
| Layer 2 | Shapes, textures (circles, lines) |
| Layer 3 | Parts of objects (eyes, wheels, doors) |
| Layer 4 | Full objects (faces, cars, dogs) |

### Speech Recognition Example

| Layer | What It Detects |
|-------|----------------|
| Layer 1 | Raw audio waveforms, frequencies |
| Layer 2 | Phonemes (basic sound units) |
| Layer 3 | Words |
| Layer 4 | Phrases and sentences |

The network figures all of this out **on its own** during training — no human tells it what to look for at each layer.

---

## Why "Deep"?

Shallow networks (1–2 hidden layers) can theoretically approximate any function, but they may need an **exponentially large** number of neurons to do so.

Deep networks (many hidden layers) can represent the same function with far fewer parameters by building up complexity layer by layer. This is the key advantage of depth.

```
Shallow network:   input → [massive hidden layer] → output
Deep network:      input → [small] → [small] → [small] → [small] → output
```

Both can solve the same problem, but the deep network is far more efficient.

---

## How Deep Learning Learns: Training

Training a neural network works the same way as training linear or logistic regression — **gradient descent** minimising a cost function — but extended to many layers.

### Forward Propagation
Pass the input through every layer from left to right, computing each neuron's activation. The final layer produces a prediction $\hat{y}$.

### Loss Calculation
Compare $\hat{y}$ to the true label $y$ using a cost function (e.g. log loss for classification, squared error for regression).

### Backpropagation
Compute how much each weight contributed to the error by propagating gradients **backwards** through the network using the chain rule. This is how neural networks efficiently compute $\frac{\partial J}{\partial w}$ for every weight.

### Weight Update
Apply gradient descent to update every weight:

$$w := w - \alpha \frac{\partial J}{\partial w}$$

Repeat until the cost converges.

---

## Why Has Deep Learning Become So Powerful?

Three factors came together in the 2010s:

### 1. Big Data
Deep networks need vast amounts of labelled data to learn from. The explosion of the internet, smartphones, and sensors made this data available.

### 2. Compute (GPUs)
Training deep networks requires billions of multiplications. GPUs (originally built for video games) can perform these operations in parallel, reducing training time from months to hours.

### 3. Algorithms
Key improvements like better activation functions (ReLU), dropout regularisation, batch normalisation, and the Adam optimiser made training deep networks stable and practical.

```
        Performance
            |                        Deep Learning
            |                      /
            |                    /
            |         ML (other)/
            |        /----------
            |       /
            +--------------------------- Amount of Data
                          ↑
              Deep learning keeps improving with more data;
              traditional ML plateaus
```

---

## Deep Learning vs Traditional Machine Learning

| | Traditional ML | Deep Learning |
|---|---|---|
| Feature engineering | Manual (human decides features) | Automatic (learned from data) |
| Data requirement | Works with small datasets | Needs large datasets |
| Compute | CPU is usually enough | Requires GPU |
| Interpretability | Often interpretable | Often a "black box" |
| Performance on structured data | Strong | Comparable |
| Performance on images/audio/text | Weak | State of the art |

---

## Common Applications

| Domain | Application |
|--------|-------------|
| **Computer Vision** | Image classification, object detection, facial recognition |
| **Natural Language Processing** | Translation, chatbots, sentiment analysis, summarisation |
| **Speech** | Speech recognition, text-to-speech |
| **Healthcare** | Medical image diagnosis, drug discovery |
| **Finance** | Fraud detection, algorithmic trading |
| **Autonomous Vehicles** | Lane detection, obstacle avoidance |
| **Generative AI** | Image generation (diffusion models), LLMs (GPT, Claude) |

---

## Key Terms

| Term | Meaning |
|------|---------|
| **Neuron** | Basic computational unit; computes a weighted sum + activation |
| **Layer** | A group of neurons at the same depth in the network |
| **Hidden layer** | Any layer between input and output |
| **Activation function** | Non-linear function applied to each neuron's output (e.g. ReLU, sigmoid) |
| **Deep network** | Neural network with many hidden layers |
| **Forward propagation** | Computing predictions by passing data through the network left to right |
| **Backpropagation** | Computing gradients by passing errors right to left through the network |
| **Gradient descent** | Optimisation algorithm that updates weights to minimise cost |
| **Parameters** | Weights and biases — everything learned during training |
| **Hyperparameters** | Settings chosen before training (number of layers, learning rate, etc.) |
