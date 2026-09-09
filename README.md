# Deep Learning: From Foundations to Transformers 🚀

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive deep learning repository documenting hands-on implementations, custom neural architecture designs, sequence modeling, and transformer fine-tuning.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Course Curriculum & Lessons](#-course-curriculum--lessons)
- [Key Architectural Highlights](#-key-architectural-highlights)
  - [1. Micrograd: Autograd Engine from Scratch](#1-micrograd-autograd-engine-from-scratch)
  - [2. WaveNet Text Generation](#2-wavenet-text-generation)
  - [3. Fine-Tuning BERT for Emotion Classification](#3-fine-tuning-bert-for-emotion-classification)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [License](#-license)

---

## 🧠 Overview

This repository contains a complete journey through modern Deep Learning, spanning:
1. **Mathematical Principles & Autograd Engine** (Tensors, DAG computational graphs, manual backpropagation).
2. **Neural Network Frameworks** (TensorFlow/Keras vs. PyTorch custom modules & data pipelines).
3. **Computer Vision** (CNNs, spatial convolutions, MaxPooling, Torchvision pipelines).
4. **Regularization & Optimization** (Inverted Dropout from scratch, Batch Normalization, Kaiming Init, PCA embedding diagnostics).
5. **Language Modeling** (Statistical N-grams, Bengio MLP character embeddings, WaveNet dilated causal convolutions).
6. **Sequence Modeling** (Character 1D CNNs, RNNs, LSTMs, GRUs).
7. **Transformers & Transfer Learning** (Hugging Face ecosystem, BERT fine-tuning, dynamic collation, HF Hub integration).

---

## 📚 Course Curriculum & Lessons

| Lesson | Core Topic / Focus Area | Key Concepts & Architectures |
| :---: | :--- | :--- |
| **Lesson 1** | PyTorch Tensors & Vector Math | Multi-dimensional tensors, matrix multiplication (`@`), autograd basics |
| **Lesson 2** | Micrograd Engine | Custom `Value` class, DAG backpropagation, chain rule, `ReLU`, `Sigmoid`, `Tanh` |
| **Lesson 3** | Keras Sequential API | Feedforward networks, `Dense`, `Flatten`, Softmax, Fashion-MNIST |
| **Lesson 4** | Advanced Keras APIs | Functional API (multi-branch graphs) & Model Subclassing (`call()`) |
| **Lesson 5** | PyTorch Pure Autograd | Manual parameter optimization (`w.data -= lr * w.grad`), MSE loss |
| **Lesson 6** | PyTorch Custom Modules | `nn.Module` subclassing, `TensorDataset`, mini-batch `DataLoader` |
| **Lesson 7** | Computer Vision (CNNs) | `nn.Conv2d`, `nn.MaxPool2d`, image transforms, Oxford Flowers |
| **Lesson 8** | Custom Dropout | Inverted dropout scaling (`1 / (1-p)`), training vs evaluation modes |
| **Lesson 9** | Statistical N-Grams | Bigram & Trigram frequency matrices, row-wise probability normalization |
| **Lesson 10** | Bengio MLP Architecture | Character embedding lookup ($C$), context window concatenation, Tanh hidden layer |
| **Lesson 11** | Batch Normalization & PCA | `nn.BatchNorm1d`, Kaiming He initialization, PCA 2D embedding visualization |
| **Lesson 12** | Deep MLPs & Residuals | Multi-layer networks, residual connections, embedding inspection |
| **Lessons 13–14** | TinyStories & Autoregressive LM | Context windows, positional encodings, next-token prediction |
| **Lesson 15** | WaveNet Architecture | 1D Causal Dilated Convolutions, Gated Residual Blocks, Tiktoken BPE |
| **Lessons 16–17** | Text 1D CNNs | `nn.Conv1d` text classification, dynamic collation with custom `collate_fn` |
| **Lessons 18–19** | Recurrent Models (RNN/LSTM/GRU) | Hidden state persistence, Cell state ($c_t$), Forget/Input/Output gates |
| **Lessons 20 & 23** | Hugging Face Foundations | `pipeline` API, `AutoTokenizer`, `AutoModelForSequenceClassification` |
| **Lessons 21 & 25** | Fine-Tuning BERT | Pre-trained `bert-base-uncased`, `DataCollatorWithPadding`, AdamW + Linear Warmup, HF Hub |

---

## 🛠️ Key Architectural Highlights

### 1. Micrograd: Autograd Engine from Scratch
Building an automatic differentiation engine that tracks topological node dependencies in a computational graph:

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
```

### 2. WaveNet Text Generation
Implementing dilated causal convolutions with gated activation units ($\tanh \odot \sigma$) and skip connections:

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            channels, channels * 2, kernel_size=2,
            dilation=dilation, padding=dilation
        )
        self.conv_res = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv_skip = nn.Conv1d(channels, channels, kernel_size=1)
        
    def forward(self, x):
        out = self.dilated_conv(x)
        out = out[:, :, :-self.dilated_conv.dilation[0]] # Maintain causality
        filter_out, gate_out = out.chunk(2, dim=1)
        gated = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        return x + self.conv_res(gated), self.conv_skip(gated)
```

### 3. Fine-Tuning BERT for Emotion Classification
Fine-tuning `bert-base-uncased` on the SMILE Twitter Emotion Dataset:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

# Dynamic mini-batch padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

---

## 📁 Repository Structure

```
.
├── Homework/                       # Complete Homework Lessons (Lessons 1 – 25)
│   ├── lesson-1.py                 # PyTorch Tensors
│   ├── lesson-2.py                 # Custom Micrograd Engine
│   ├── ...
│   ├── lesson-25.py                # BERT Fine-Tuning Script
│   └── lesson-25.ipynb             # BERT Fine-Tuning Notebook
├── Language Models/                # Language Model Experiments
├── practice/                       # Notebooks & Practice Scripts
├── Deep_Learning_Course_Review.docx # Comprehensive Word Study Guide
├── Deep_Learning_Course_Review.md   # Comprehensive Markdown Study Guide
├── requirements.txt                # Python Dependencies
├── LICENSE                         # MIT License
└── README.md                       # Project Documentation
```

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* PyTorch 2.0+

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hamzayevtemur-lab/deep_learning.git
   cd deep_learning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run any lesson script (e.g., BERT Fine-Tuning):
   ```bash
   python Homework/lesson-25.py
   ```

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
