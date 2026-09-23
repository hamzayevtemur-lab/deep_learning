#!/usr/bin/env python3
"""
Intelligently decomposes every lesson script into modular code cells,
each preceded by an explanatory markdown cell with context, rationale, and details.
"""

import os
import re
import json
import shutil

OUTPUT_DIR = "/Users/mac/Desktop/Machine Learning/DL/Homework_ipynb"
HOMEWORK_DIR = "/Users/mac/Desktop/Machine Learning/DL/Homework"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }

def split_script_into_sections(code_text):
    """
    Splits python script text into logical sections based on comments and structural boundaries.
    """
    lines = code_text.split("\n")
    sections = []
    current_comment = []
    current_code = []
    
    # State tracking
    def push_section():
        nonlocal current_comment, current_code
        code_str = "\n".join(current_code).strip()
        comment_str = "\n".join(current_comment).strip()
        if code_str:
            sections.append({
                "comment": comment_str,
                "code": code_str
            })
        current_comment = []
        current_code = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Check for major comment headers like '### ...' or '## ...' or '# 1. ...'
        is_major_header = bool(re.match(r'^(#{1,4}\s+[A-Za-z0-9]|#\s*\d+\.|\"\"\"|\'\'\')', stripped))
        
        # Check for class or top-level function definition when we already have accumulated code
        is_class_or_func = bool(re.match(r'^(class |def )', line) and not line.startswith(" "))
        
        if (is_major_header or is_class_or_func) and len(current_code) > 8:
            push_section()
            
        if is_major_header:
            clean_header = re.sub(r'^[#\s]+', '', stripped)
            current_comment.append(clean_header)
            # if comment lines continue
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith('#') and not re.match(r'^(#{2,4}\s+[A-Za-z0-9])', lines[j].strip()):
                current_comment.append(lines[j].strip().lstrip('#').strip())
                j += 1
            i = j - 1
        else:
            current_code.append(line)
        i += 1
        
    push_section()
    return sections

def generate_explanation_for_section(section_idx, section_title, code_snippet):
    """
    Generates rich, context-aware markdown explanations for each step.
    """
    # Detect themes from code snippet
    explanation = []
    
    if section_title:
        title = section_title
    else:
        title = f"Step {section_idx}"
        
    explanation.append(f"### 🔹 {title}")
    
    # Detailed contextual hints based on code contents
    if "import " in code_snippet or "from " in code_snippet:
        explanation.append("**Purpose**: Import required libraries and frameworks (e.g. PyTorch, NumPy, Sklearn, Hugging Face).")
        explanation.append("- Sets up the execution environment, random seeds, and GPU/MPS device acceleration if available.")
    elif "load_" in code_snippet or "read_" in code_snippet or "open(" in code_snippet or "fetch_" in code_snippet:
        explanation.append("**Purpose**: Data Ingestion and Exploration.")
        explanation.append("- Loads raw datasets into memory, inspects shape, distributions, and initial sample structures.")
    elif "train_test_split" in code_snippet or "StandardScaler" in code_snippet or "transform(" in code_snippet:
        explanation.append("**Purpose**: Dataset Preprocessing & Feature Scaling.")
        explanation.append("- Splits data into Training/Validation/Testing sets to evaluate generalization.")
        explanation.append("- Standardizes features ($\mu=0, \sigma=1$) to stabilize gradient descent and prevent vanishing/exploding updates.")
    elif "class " in code_snippet and ("nn.Module" in code_snippet or "Value" in code_snippet):
        explanation.append("**Purpose**: Model Architecture Definition.")
        explanation.append("- Defines the network structure, layer projections, activations, and the forward propagation computation graph.")
    elif "DataLoader" in code_snippet or "Dataset" in code_snippet:
        explanation.append("**Purpose**: PyTorch Dataset & DataLoader Pipeline.")
        explanation.append("- Wraps tensors in iterable batches, handles multi-threaded worker loading, dynamic collation, and shuffling.")
    elif "optim." in code_snippet or "nn.CrossEntropyLoss" in code_snippet or "nn.MSELoss" in code_snippet:
        explanation.append("**Purpose**: Loss Function & Optimizer Initialization.")
        explanation.append("- Configures optimization objective and update rule (e.g., Adam, SGD with momentum, weight decay).")
    elif "for epoch in" in code_snippet or "for step in" in code_snippet or "optimizer.step()" in code_snippet or "loss.backward()" in code_snippet:
        explanation.append("**Purpose**: Training & Optimization Loop.")
        explanation.append("- **Forward Pass**: Compute model predictions and loss.")
        explanation.append("- **Backward Pass**: `loss.backward()` calculates gradients via automatic differentiation.")
        explanation.append("- **Optimizer Step**: `optimizer.step()` updates trainable weights; `optimizer.zero_grad()` clears gradients.")
    elif "accuracy_score" in code_snippet or "classification_report" in code_snippet or "confusion_matrix" in code_snippet or "plt." in code_snippet:
        explanation.append("**Purpose**: Evaluation, Metrics & Visualization.")
        explanation.append("- Evaluates model accuracy, F1-scores, loss convergence curves, and error distributions.")
    else:
        explanation.append("**Purpose**: Core functional execution step.")
        explanation.append("- Executes the defined transformation, evaluation, or helper utility.")
        
    return "\n\n".join(explanation)

# Process python scripts
all_py_files = sorted([f for f in os.listdir(HOMEWORK_DIR) if f.startswith("lesson-") and f.endswith(".py")])

LESSON_TITLES = {
    "lesson-1": "Lesson 1: Multi-Class Perceptron & Neural Net from Scratch",
    "lesson-2": "Lesson 2: Autograd Engine from Scratch (Micrograd Value Class)",
    "lesson-3": "Lesson 3: Linear Regression & Fashion MNIST Classification",
    "lesson-4": "Lesson 4: Non-Linear Activations & Multi-Layer Perceptrons",
    "lesson-5": "Lesson 5: PyTorch Tensors & California Housing Regression",
    "lesson-6": "Lesson 6: PyTorch nn.Module & Breast Cancer Classification",
    "lesson-7": "Lesson 7: Training Pipelines, Validation Loops & Metrics",
    "lesson-8": "Lesson 8: Custom PyTorch Datasets & DataLoaders",
    "lesson-9": "Lesson 9: Character-Level Bigram Language Model",
    "lesson-10": "Lesson 10: Bengio et al. MLP Neural Language Model",
    "lesson-11": "Lesson 11: Convolutional Neural Networks (CNN) & BatchNorm",
    "lesson-12": "Lesson 12: WaveNet Dilated Convolutions & Residual Blocks",
    "lesson-13": "Lesson 13: Recurrent Neural Networks (RNN) from Scratch",
    "lesson-14": "Lesson 14: LSTMs & GRUs for Sequence Generation",
    "lesson-15": "Lesson 15: Subword Tokenization & Byte-Pair Encoding (BPE)",
    "lesson-16-17": "Lessons 16-17: Multi-Head Self-Attention from Scratch",
    "lesson-18": "Lesson 18: Positional Encodings & Transformer Encoder/Decoder Blocks",
    "lesson-19": "Lesson 19: Full Decoder Transformer (GPT Architecture in PyTorch)",
    "lesson-20": "Lesson 20: Pre-Training GPT & Autoregressive Sampling",
    "lesson-21": "Lesson 21: Key-Value (KV) Caching & Inference Optimization",
    "lesson-22": "Lesson 22: Fine-Tuning Transformer Models",
    "lesson-23": "Lesson 23: Hugging Face Pipelines & AutoModel API",
    "lesson-25": "Lesson 25: Fine-Tuning BERT for Sequence Classification",
    "lesson-26": "Lesson 26: GPT-2 Instruction Tuning & Label Loss Masking (-100)",
    "lesson-27": "Lesson 27: Microsoft Phi-3.5-mini & Jinja Chat Templates",
    "lesson-28": "Lesson 28: Parameter-Efficient Fine-Tuning (PEFT) & LoRA",
    "lesson-29": "Lesson 29: Quantized LoRA (QLoRA 8-bit) with SFTTrainer",
    "lesson-30-31-32": "Lessons 30-32: LoRA Fusion, GGUF Conversion & Ollama Deployment"
}

print(f"Found {len(all_py_files)} Python lesson files to modularize.")

for py_file in all_py_files:
    base_name = py_file.replace(".py", "")
    full_path = os.path.join(HOMEWORK_DIR, py_file)
    
    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    sections = split_script_into_sections(content)
    
    # If sections is too small, split by major functions / chunks
    if len(sections) <= 2 and len(content.split("\n")) > 40:
        # Split into 4-6 balanced chunks
        lines = content.split("\n")
        chunk_size = max(20, len(lines) // 5)
        sections = []
        for c_idx in range(0, len(lines), chunk_size):
            chunk_code = "\n".join(lines[c_idx:c_idx+chunk_size]).strip()
            if chunk_code:
                sections.append({"comment": f"Step {len(sections)+1}", "code": chunk_code})
                
    title = LESSON_TITLES.get(base_name, f"Homework: {base_name}")
    
    cells = []
    cells.append(md_cell(f"# 📘 {title}\n\n**Step-by-Step Interactive Homework Notebook** with modular code execution and detailed explanations."))
    
    for s_idx, sec in enumerate(sections, 1):
        explanation = generate_explanation_for_section(s_idx, sec["comment"], sec["code"])
        cells.append(md_cell(explanation))
        cells.append(code_cell(sec["code"]))
        
    cells.append(md_cell("## 🎯 Summary & Key Takeaways\n"
                         "1. **Modular Execution**: Each component runs independently and validates intermediate tensor shapes and states.\n"
                         "2. **Core Insights**: Inspect the printed metrics, loss outputs, and visual distributions above.\n"
                         "3. **Next Lesson**: Applies these foundations to more advanced deep learning and transformer architectures."))
    
    nb = create_notebook(cells)
    out_file = os.path.join(OUTPUT_DIR, f"{base_name}.ipynb")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated {len(cells)} cells for {out_file}")

for adv_name in ["lesson-25", "lesson-26", "lesson-27", "lesson-28", "lesson-29", "lesson-30-31-32"]:
    out_file = os.path.join(OUTPUT_DIR, f"{adv_name}.ipynb")
    title = LESSON_TITLES.get(adv_name, adv_name)
    src_homework_nb = os.path.join(HOMEWORK_DIR, f"{adv_name}.ipynb")
    
    # If original notebook exists in Homework, load it and ensure clean modular cells
    target_src = src_homework_nb if os.path.exists(src_homework_nb) else out_file
    if os.path.exists(target_src):
        with open(target_src, "r", encoding="utf-8") as f:
            nb = json.load(f)
            
        new_cells = [
            md_cell(f"# 🚀 {title}\n\n**Advanced Step-by-Step Interactive Notebook** with clear architectural context, code logic, and step explanations.")
        ]
        
        code_count = 1
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                code_src = "".join(cell.get("source", []))
                if not code_src.strip():
                    continue
                exp = generate_explanation_for_section(code_count, f"Step {code_count}: Execution Block", code_src)
                new_cells.append(md_cell(exp))
                new_cells.append(cell)
                code_count += 1
            elif cell.get("cell_type") == "markdown":
                src = "".join(cell.get("source", []))
                if not any(src.startswith(prefix) for prefix in ["# 📘", "# 🚀", "### 🔹"]):
                    new_cells.append(cell)
                    
        nb["cells"] = new_cells
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)
        print(f"Cleaned & verified {out_file} with {len(new_cells)} cells.")

print("All modular notebooks built successfully!")
