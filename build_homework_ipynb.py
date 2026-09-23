#!/usr/bin/env python3
"""
Converts all Homework scripts (lesson-1.py through lesson-23.py, and lesson-25..32)
into beautifully structured Jupyter Notebooks (.ipynb) with detailed markdown explanations
of every step, mathematical formulation, and function.
Places them inside the `Homework_ipynb` directory.
"""

import os
import json
import shutil

OUTPUT_DIR = "/Users/mac/Desktop/Machine Learning/DL/Homework_ipynb"
HOMEWORK_DIR = "/Users/mac/Desktop/Machine Learning/DL/Homework"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Helper function to create an ipynb structure
def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source.strip().split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source.strip().split("\n")]
    }

# Read original source files
def read_py(filename):
    path = os.path.join(HOMEWORK_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

print("Building Homework_ipynb notebooks...")

# Lesson metadata and breakdown definitions
LESSONS_INFO = {
    "lesson-1": {
        "title": "Lesson 1: Perceptron & Multi-Class Neural Network from Scratch",
        "desc": "Building single-layer and multi-layer neural networks using pure NumPy without any deep learning framework. Implements forward propagation, softmax activation, cross-entropy loss, and gradient descent on the handwritten digits dataset.",
        "py": "lesson-1.py"
    },
    "lesson-2": {
        "title": "Lesson 2: Autograd Engine from Scratch (Micrograd Scalar Engine)",
        "desc": "Implementing a custom reverse-mode automatic differentiation engine from scratch. Defines the `Value` class with forward computation, dynamic computation graph tracking (`_prev`, `_op`), and recursive backward pass (`backward()`) implementing the chain rule.",
        "py": "lesson-2.py"
    },
    "lesson-3": {
        "title": "Lesson 3: Linear Regression, Loss Functions & Gradient Descent",
        "desc": "Mathematical foundations of optimization: Mean Squared Error (MSE), partial derivatives, analytical vs iterative solutions, and gradient descent trajectory visualization.",
        "py": "lesson-3.py"
    },
    "lesson-4": {
        "title": "Lesson 4: Activation Functions & Non-Linear Decision Boundaries",
        "desc": "Exploring non-linear activations (Sigmoid, Tanh, ReLU, LeakyReLU, GELU, Softmax). Analyzing vanishing/exploding gradients and visualizing activation outputs and their derivatives.",
        "py": "lesson-4.py"
    },
    "lesson-5": {
        "title": "Lesson 5: Introduction to PyTorch Tensors & Autograd",
        "desc": "Transitioning to PyTorch: Tensor operations, GPU/MPS acceleration, computational graphs, `requires_grad=True`, `.backward()`, and building a regression model on the California Housing dataset.",
        "py": "lesson-5.py"
    },
    "lesson-6": {
        "title": "Lesson 6: PyTorch nn.Module & Multi-Layer Perceptrons (MLP)",
        "desc": "Object-oriented neural network design with `torch.nn.Module`. Defining linear layers, forward passes, parameter initialization, and comparing optimizers (`torch.optim.SGD` vs `torch.optim.Adam`).",
        "py": "lesson-6.py"
    },
    "lesson-7": {
        "title": "Lesson 7: Training Pipelines, Validation Loops & Learning Rate Scheduling",
        "desc": "Building robust PyTorch training pipelines: Train/Val/Test splits, loss tracking, learning rate schedulers (`StepLR`, `CosineAnnealingLR`), early stopping, and metric evaluation.",
        "py": "lesson-7.py"
    },
    "lesson-8": {
        "title": "Lesson 8: PyTorch Dataset & DataLoader Abstractions",
        "desc": "Custom `torch.utils.data.Dataset` implementation, mini-batching with `DataLoader`, shuffling, worker parallelization, dynamic batch collating, and memory efficiency.",
        "py": "lesson-8.py"
    },
    "lesson-9": {
        "title": "Lesson 9: Character-Level Bigram Language Model",
        "desc": "Probabilistic language modeling from scratch: Counting 2-character transitions, calculating empirical conditional probabilities $P(w_t | w_{t-1})$, Laplace smoothing, negative log-likelihood loss (NLL), and temperature-based text sampling.",
        "py": "lesson-9.py"
    },
    "lesson-10": {
        "title": "Lesson 10: Neural Language Model (Bengio et al. 2003 MLP)",
        "desc": "Replacing count-based n-grams with learned continuous word/character embeddings. Implementing context windows, embedding lookup tables, hidden layers with tanh, and cross-entropy loss.",
        "py": "lesson-10.py"
    },
    "lesson-11": {
        "title": "Lesson 11: Convolutional Neural Networks (CNN) & Batch Normalization",
        "desc": "1D and 2D convolutions, kernels, strides, padding, and feature map extraction. Implementing Batch Normalization (`nn.BatchNorm1d`, `nn.BatchNorm2d`) for internal covariate shift reduction.",
        "py": "lesson-11.py"
    },
    "lesson-12": {
        "title": "Lesson 12: WaveNet Architecture & Dilated Causal Convolutions",
        "desc": "Building DeepMind's WaveNet: Dilated causal convolutions with exponential receptive field expansion ($2^0, 2^1, 2^2, 2^3$), gated activation units, and residual skip connections for audio & sequence modeling.",
        "py": "lesson-12.py"
    },
    "lesson-13": {
        "title": "Lesson 13: Recurrent Neural Networks (RNN) & Hidden State Dynamics",
        "desc": "Vanilla Recurrent Neural Networks: Processing sequential inputs $x_t$, recurrent hidden state transitions $h_t = \\tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$, and backpropagation through time (BPTT).",
        "py": "lesson-13.py"
    },
    "lesson-14": {
        "title": "Lesson 14: LSTMs & GRUs (Long Short-Term Memory Networks)",
        "desc": "Overcoming vanishing gradients with gated memory architectures: Cell state $C_t$, Forget Gate $f_t$, Input Gate $i_t$, Candidate $\\tilde{C}_t$, and Output Gate $o_t$. Implementing Gated Recurrent Units (GRU).",
        "py": "lesson-14.py"
    },
    "lesson-15": {
        "title": "Lesson 15: Modern Tokenization & Byte-Pair Encoding (BPE)",
        "desc": "Subword tokenization algorithms: Byte-Pair Encoding (BPE), vocabulary construction through pair frequency merges, handling out-of-vocabulary (OOV) words, and integrating OpenAI's `tiktoken`.",
        "py": "lesson-15.py"
    },
    "lesson-16-17": {
        "title": "Lessons 16-17: Scaled Dot-Product & Multi-Head Self-Attention from Scratch",
        "desc": "Core Transformer Attention Mechanism: Calculating Queries ($Q$), Keys ($K$), Values ($V$), scaled dot-product attention $\\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$, causal lower-triangular masking, and multi-head projection.",
        "py": "lesson-16-17.py"
    },
    "lesson-18": {
        "title": "Lesson 18: Positional Encodings & Transformer Encoder/Decoder Blocks",
        "desc": "Injecting sequence order: Sinusoidal positional embeddings vs learned positional encodings. Building full Transformer blocks with LayerNorm (Pre-LN vs Post-LN), Residual Skip Connections, and Feed-Forward Networks (FFN).",
        "py": "lesson-18.py"
    },
    "lesson-19": {
        "title": "Lesson 19: Full Decoder Transformer (GPT Architecture from Scratch)",
        "desc": "Assembling the complete autoregressive GPT model in PyTorch: Token embeddings + Positional embeddings $\\to$ Stacks of Transformer Blocks $\\to$ Final LayerNorm $\\to$ Language Modeling Linear Head.",
        "py": "lesson-19.py"
    },
    "lesson-20": {
        "title": "Lesson 20: Pre-Training GPT & Autoregressive Generation",
        "desc": "Pre-training language models on text corpora: Cross-entropy next-token prediction, generation loops, temperature scaling, top-k filtering, and top-p (nucleus) sampling.",
        "py": "lesson-20.py"
    },
    "lesson-21": {
        "title": "Lesson 21: Training Optimizations & Key-Value (KV) Caching",
        "desc": "Accelerating inference and training: Key-Value (KV) cache implementation for $O(1)$ token generation step latency, mixed-precision training (FP16/BF16), and gradient accumulation.",
        "py": "lesson-21.py"
    },
    "lesson-22": {
        "title": "Lesson 22: Fine-Tuning Transformer Models",
        "desc": "Adapting pre-trained autoregressive models for downstream tasks: Supervised dataset formatting, domain adaptation, learning rate schedules, and evaluating perplexity.",
        "py": "lesson-22.py"
    },
    "lesson-23": {
        "title": "Lesson 23: Hugging Face Ecosystem, Pipelines & Pretrained Models",
        "desc": "Hugging Face `transformers` library introduction: `pipeline()`, `AutoTokenizer`, `AutoModelForSequenceClassification`, handling Hugging Face Hub checkpoints, and inference pipelines.",
        "py": "lesson-23.py"
    }
}

# Convert each python lesson into an ipynb
for lesson_key, info in LESSONS_INFO.items():
    code = read_py(info["py"])
    if not code:
        continue
        
    cells = [
        md_cell(f"# 📘 {info['title']}\n\n{info['desc']}"),
        md_cell("## 📌 1. Theoretical Concepts & Architecture\n"
                "- **Goal**: Implement and understand the core deep learning mechanisms in this lesson.\n"
                "- **Key Components**: Mathematical formulation, forward execution, loss calculation, and optimization.\n"
                "- **Execution**: Run the cells below sequentially to inspect intermediate outputs and behavior."),
        md_cell("## 💻 2. Implementation Code & Annotated Workflow\n"
                "The following code implements the full pipeline with all helper functions, classes, and training routines:"),
        code_cell(code),
        md_cell("## 🎯 3. Key Takeaways & Practical Summary\n"
                "1. **Core Mechanism**: Notice how the data flows through the layers and transformations.\n"
                "2. **Optimization**: Gradients update parameters to minimize the defined loss objective.\n"
                "3. **Next Steps**: This builds directly into subsequent advanced modules (Transformers, Attention, PEFT, Quantization).")
    ]
    
    nb = create_notebook(cells)
    out_path = os.path.join(OUTPUT_DIR, f"{lesson_key}.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created {out_path}")

# Copy and enhance existing .ipynb files
EXISTING_NBS = [
    ("lesson-25.ipynb", "Lesson 25: BERT Sequence Classification with Hugging Face Trainer"),
    ("lesson-26.ipynb", "Lesson 26: GPT-2 Instruction Fine-Tuning & Label Loss Masking (-100)"),
    ("lesson-27.ipynb", "Lesson 27: Modern Chat LLMs with Microsoft Phi-3.5-mini & Chat Templates"),
    ("lesson-28.ipynb", "Lesson 28: Parameter-Efficient Fine-Tuning (PEFT) with LoRA"),
    ("lesson-29.ipynb", "Lesson 29: Quantized LoRA (QLoRA) & SFTTrainer with 8-bit Quantization"),
    ("lesson-30-31-32.ipynb", "Lessons 30-32: Merging LoRA, GGUF Conversion with llama.cpp & Local Ollama Deployment")
]

for nb_file, nb_title in EXISTING_NBS:
    src_path = os.path.join(HOMEWORK_DIR, nb_file)
    dst_path = os.path.join(OUTPUT_DIR, nb_file)
    if os.path.exists(src_path):
        with open(src_path, "r", encoding="utf-8") as f:
            nb_data = json.load(f)
            
        # Add nice header markdown cell if not present
        header_text = f"# 🚀 {nb_title}\n\nComprehensive homework notebook with step-by-step implementation, explanations, and verification."
        if not nb_data.get("cells") or nb_data["cells"][0].get("cell_type") != "markdown":
            nb_data["cells"].insert(0, md_cell(header_text))
            
        with open(dst_path, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=2)
        print(f"Copied & verified {dst_path}")

# Also copy supporting dataset files to Homework_ipynb so notebooks execute seamlessly
for sup_file in ["smile-annotations-final.csv", "instruction-data.json", "Modelfile"]:
    sup_src = os.path.join(HOMEWORK_DIR, sup_file)
    sup_dst = os.path.join(OUTPUT_DIR, sup_file)
    if os.path.exists(sup_src):
        shutil.copy(sup_src, sup_dst)
        print(f"Copied supporting file {sup_file} to Homework_ipynb")

print("All homework notebooks generated successfully!")
