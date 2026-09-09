# Deep Learning Course Review & Study Guide

**Comprehensive Synthesis of Lessons 1 – 25: Foundations to Fine-Tuning Transformers**

---

## 1. Deep Learning Course Roadmap Overview

This study guide synthesizes all 25 lessons from your Deep Learning course. It covers the progression from low-level mathematical principles (tensors, backpropagation, autograd engines) to computer vision, custom regularization, statistical & neural language modeling, autoregressive sequence models (WaveNet, RNNs, LSTMs), and state-of-the-art Transformer fine-tuning with Hugging Face BERT.

| Module | Lessons Included | Core Topics & Frameworks Learned |
| :--- | :--- | :--- |
| **1. Math & Autograd** | Lessons 1 – 2 | Tensors, Vector Operations, Computational Graphs, Micrograd Engine, Custom Autograd |
| **2. Frameworks & APIs** | Lessons 3 – 6 | TensorFlow/Keras (Sequential, Functional, Subclassing), PyTorch Pure Autograd, Custom `nn.Module`, DataLoaders |
| **3. Computer Vision** | Lesson 7 | Convolutional Neural Networks (CNNs), Torchvision Transforms, Conv2d/MaxPool2d, Oxford Flowers Classification |
| **4. Regularization & Diagnostics** | Lessons 8 & 11 | Custom Dropout (Bernoulli Masking), Batch Normalization, Kaiming Weight Initialization, PCA Embedding Analysis |
| **5. Language Modeling (N-Grams)** | Lessons 9, 10, 12 | Statistical Bigrams/Trigrams, Bengio MLP Architecture, Character Embeddings, Context Windows |
| **6. Advanced Sequence Models** | Lessons 13 – 19 | TinyStories Dataset, Positional Encodings, WaveNet (1D Causal Dilated Convolutions, Residual/Gated Blocks, Tiktoken BPE), Text 1D CNNs, RNNs, LSTMs, GRUs |
| **7. Transformers & Transfer Learning** | Lessons 20, 21, 23, 25 | Hugging Face Pipelines, Tokenizers, AutoModel, Fine-Tuning BERT for Emotion Classification, DataCollator, AdamW + Warmup Scheduler, HF Hub Upload |

---

## 2. Module 1: Foundations & Automatic Differentiation

### Lesson 1: PyTorch Tensors & Vector Math
- **PyTorch Tensors:** Multi-dimensional arrays optimized for GPU acceleration and automatic differentiation.
- **Core Tensor Operations:** Element-wise arithmetic (`+`, `-`, `*`), matrix multiplication (`torch.matmul` or `@`), reshaping (`.view()`, `.reshape()`), and axis reduction (`.sum(dim=...)`, `.mean()`).
- **Autograd Basics:** Setting `requires_grad=True` enables PyTorch to record computational graphs for backpropagation.

### Lesson 2: Micrograd – Custom Autograd Engine from Scratch
- **The Scalar Value Object:** Building a custom `Value` class that wraps scalar data and tracks DAG (Directed Acyclic Graph) node relationships (`_prev`, `_op`).
- **Backpropagation Chain Rule:** Computing derivatives manually: $\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial x}$. Each operator defines its local gradient lambda (`_backward`).
- **Topological Sorting:** Nodes are sorted topologically so gradients flow backwards from loss to inputs without uninitialized references.
- **Custom Activations:** Implementing `ReLU`, `Sigmoid`, and `Tanh` forward pass and derivative equations from scratch.

```python
# Micrograd Chain Rule implementation snippet
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

---

## 3. Module 2: Neural Network Frameworks & Architectures

### Lesson 3: Keras Foundations & Fashion-MNIST Classification
- **Sequential API:** Stacking linear layers (`tf.keras.layers.Dense`) for end-to-end feedforward networks.
- **Flatten Layer:** Converting 2D image grids (28x28) into 1D feature vectors (784 features).
- **Loss & Softmax:** Using `SparseCategoricalCrossentropy` with multi-class Softmax outputs.

### Lesson 4: Advanced Keras API Paradigms
- **Functional API:** Defining non-linear graph architectures, multi-input/multi-output models using `Input()` and callable layer objects.
- **Model Subclassing API:** Inheriting from `tf.keras.Model` and overriding `call(inputs)` for maximum custom logic and imperative flexibility.

### Lesson 5: Pure PyTorch Autograd & Linear Regression
- **Manual Parameters:** Explicitly instantiating `w` and `b` tensors with `requires_grad=True`.
- **Manual Optimization Loop:** Computing MSE loss, executing `loss.backward()`, updating weights with `w.data -= lr * w.grad`, and resetting `w.grad.zero_()`.
- **Feature Scaling:** Using `StandardScaler` from scikit-learn for standardized zero-mean, unit-variance input normalization.

### Lesson 6: PyTorch Modular Networks & Data Pipeline
- **Custom Linear Layer:** Writing custom subclass of `nn.Module` managing internal `weight` and `bias` parameters.
- **Dataset & DataLoader:** Wrapping arrays into `TensorDataset` and utilizing `DataLoader` for mini-batch splitting, shuffling, and multi-threaded loading.

```python
# PyTorch Custom Module Structure
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
    def forward(self, x):
        return x @ self.weight.T + self.bias
```

---

## 4. Module 3: Computer Vision & Convolutional Neural Networks

### Lesson 7: CNNs & Torchvision Image Pipelines
- **Convolutional Layer (`nn.Conv2d`):** Applies spatial filters to preserve 2D feature maps, capturing edges, textures, and visual patterns.
- **Max Pooling (`nn.MaxPool2d`):** Downsamples feature map dimensions, reducing spatial size and introducing translation invariance.
- **Torchvision Transforms:** Resizing images, converting PIL objects to Tensors, and applying ImageNet normalization (`transforms.Normalize`).
- **Custom Image Dataset:** Creating `FlowerDataset` to parse MATLAB metadata files (`.mat`) and load flower images on demand.

```python
# Conv2D Feature Extractor Pipeline
class FlowerCNN(nn.Module):
    def __init__(self, num_classes=102):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Linear(64 * 56 * 56, num_classes)
```

---

## 5. Module 4: Deep Learning Regularization & Diagnostics

### Lesson 8: Custom Dropout Implementation
- **Overfitting Prevention:** Randomly zeroing out hidden activation units during training prevents feature co-adaptation.
- **Inverted Dropout Scaling:** Scaling active units by $\frac{1}{1 - p}$ during training ensures identical activation magnitudes at test time without modifying evaluation code.
- **Training vs Evaluation Modes:** Using `model.train()` vs `model.eval()` to toggle dropout masks.

### Lesson 11: Batch Normalization & Embedding Analysis
- **Batch Normalization (`nn.BatchNorm1d`):** Normalizes mini-batch activations to zero mean and unit variance, reducing internal covariate shift and accelerating convergence.
- **Kaiming / He Weight Initialization:** Initializing weights with variance scaled by $\sqrt{2 / \text{fan\_in}}$ prevents exploding/vanishing gradients in deep ReLU networks.
- **PCA Embedding Diagnostics:** Using Principal Component Analysis (PCA) to project high-dimensional character/word embeddings onto 2D space to visualize semantic clusters.

---

## 6. Module 5: Statistical & Neural N-Gram Language Modeling

### Lesson 9: Statistical Bigram & Trigram Language Models
- **Frequency Count Matrix:** Counting consecutive character pairs (Bigram: $P(w_t \mid w_{t-1})$) and triplets (Trigram: $P(w_t \mid w_{t-2}, w_{t-1})$).
- **Probability Normalization:** Converting counts to probabilities via row-wise summation and division.
- **Autoregressive Sampling:** Generating novel names character-by-character using `torch.multinomial()` categorical distribution sampling.

### Lesson 10: Bengio MLP Architecture for Language Models
- **Character Embedding Table ($C$):** Mapping discrete character integers to dense continuous vector space ($[\text{vocab\_size}, \text{embed\_dim}]$).
- **Context Window Concatenation:** Flattens `block_size` embeddings into a single feature vector as input to hidden layer.
- **Hidden Tanh Layer & Logits:** Projects concatenated embeddings through hidden linear layer with Tanh activation to produce output vocabulary logits.

### Lesson 12: Deep MLPs & Residual Connections
- **Stacking Multi-Layer Architectures:** Extending MLPs with multiple hidden layers, batch normalization, and skip/residual connections for language modeling.

---

## 7. Module 6: Advanced Sequence Modeling & Autoregressive Architectures

### Lesson 13 & 14: TinyStories & Autoregressive Next-Token Prediction
- **TinyStories Dataset:** Using Hugging Face `datasets` library to load synthetic stories for language modeling benchmarks.
- **Positional Encoding:** Injecting sequence order information into input embeddings.
- **Causal Masking:** Ensuring tokens only attend to past and current tokens, maintaining strict autoregressive generation rules.

### Lesson 15: WaveNet Architecture for Text Generation
- **1D Causal Dilated Convolutions:** Expands receptive field exponentially with dilation factors ($1, 2, 4, 8, 16$) without increasing parameter count.
- **Gated Activation Units:** Combines Tanh and Sigmoid gates: $z = \tanh(W_f * x) \odot \sigma(W_g * x)$.
- **Residual & Skip Connections:** Passes residual signals to deeper layers while summing skip outputs across all blocks for final projection.
- **Byte-Pair Encoding (BPE):** Using OpenAI's `tiktoken` library with GPT-2 vocabulary ($50,257$ tokens) for tokenization.

### Lessons 16 – 17: 1D CNNs for Text Classification
- **Character-Level Text CNN:** Applying 1D Convolutions (`nn.Conv1d`) over sequence embeddings for multi-class nationality classification from names.
- **Dynamic Collation (`collate_fn`):** Pads sequences dynamically to maximum length per batch rather than global fixed length.

### Lessons 18 & 19: Recurrent Sequence Models (RNN, LSTM, GRU)
- **Vanilla RNN (`nn.RNN`):** Maintains persistent hidden state $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$ over time steps.
- **LSTM (`nn.LSTM`):** Solves vanishing gradients via Cell State ($c_t$) governed by Forget, Input, and Output gates.
- **GRU (`nn.GRU`):** Streamlines LSTM by combining cell and hidden states using Update and Reset gates.

---

## 8. Module 7: Modern Transformers & Transfer Learning (Hugging Face)

### Lessons 20 & 23: Introduction to Hugging Face Ecosystem
- **Pipeline API:** Instant inference interface for sentiment analysis, text generation, and zero-shot classification.
- **Manual Loading:** Using `AutoTokenizer` and `AutoModelForSequenceClassification` for custom pipeline control.

### Lessons 21 & 25: Fine-Tuning BERT for Emotion Classification
- **Pre-trained BERT (`bert-base-uncased`):** Bidirectional Encoder Representations from Transformers pre-trained on massive text corpora.
- **Dynamic Padding with `DataCollatorWithPadding`:** Pads sequence input IDs and attention masks efficiently per batch.
- **Optimizer & Warmup Scheduler:** AdamW optimizer coupled with linear warmup schedule (`get_linear_schedule_with_warmup`).
- **Gradient Clipping:** Enforcing `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` to stabilize transformer gradients.
- **Model Checkpointing & Hub Upload:** Saving optimal weights based on validation weighted F1-score and pushing directly to Hugging Face Hub using `push_to_hub()`.

```python
# BERT Fine-Tuning Step
outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
loss = outputs.loss
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
scheduler.step()
```

---

## 9. Essential Formulas & Quick Reference

| Concept / Metric | Mathematical Formula / Notation |
| :--- | :--- |
| **Cross-Entropy Loss** | $L = -\sum y_i \log(\hat{y}_i)$ |
| **Softmax Activation** | $\sigma(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ |
| **Inverted Dropout Scaling** | $x_{train} = \frac{x \cdot m}{1 - p}, \quad m \sim \text{Bernoulli}(1-p)$ |
| **Bengio Embed Matrix** | $X_{concat} = [C[x_1], C[x_2], \dots, C[x_n]]$ |
| **1D Dilated Conv Index** | $y[t] = \sum_k w[k] \cdot x[t - d \cdot k]$ |
| **LSTM Cell State** | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ |
| **Transformer Attention** | $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ |
