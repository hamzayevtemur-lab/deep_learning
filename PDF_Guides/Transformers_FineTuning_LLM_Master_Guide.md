# 🚀 Transformers, Fine-Tuning, LoRA, QLoRA & LLM Deployment: The Complete Master Guide

> **Author**: Deep Learning Course Masterclass  
> **Topic**: From Hugging Face Transformers to Modern LLM Fine-Tuning & Local Ollama Deployment  
> **Target Models**: BERT, GPT-2, Microsoft Phi-3.5-mini-instruct, Llama architectures  
> **Key Libraries**: `transformers`, `datasets`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `llama.cpp`, `ollama`

---

## 📑 Table of Contents
1. [Core Architectural Foundations: Encoders vs. Decoders](#1-core-architectural-foundations)
2. [The Hugging Face Ecosystem & Core APIs](#2-the-hugging-face-ecosystem--core-apis)
3. [Encoder Fine-Tuning: BERT for Sequence Classification](#3-encoder-fine-tuning-bert)
4. [Decoder Instruction Fine-Tuning: GPT-2 & Label Loss Masking](#4-decoder-instruction-fine-tuning-gpt-2)
5. [Modern Chat LLMs: Phi-3.5-mini & Chat Templates](#5-modern-chat-llms-phi-35-mini)
6. [Parameter-Efficient Fine-Tuning (PEFT) & LoRA Mathematics](#6-parameter-efficient-fine-tuning-peft--lora)
7. [Quantization & QLoRA with SFTTrainer](#7-quantization--qlora-with-sfttrainer)
8. [Full Deployment Lifecycle: Merging, GGUF & Ollama](#8-full-deployment-lifecycle-merging-gguf--ollama)
9. [Master API Cheat Sheet & Function Glossary](#9-master-api-cheat-sheet--glossary)

---

# 1. Core Architectural Foundations

```
                            TRANSFORMER TAXONOMY
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                                                       ▼
   ENCODER-ONLY                                            DECODER-ONLY
 (e.g. BERT, RoBERTa)                                  (e.g. GPT-2, Phi-3.5, Llama)
──────────────────────                                ──────────────────────────────
• Bidirectional Attention                              • Causal (Unidirectional) Masked Attention
• Sees past + future context                           • Sees only past tokens (autoregressive)
• Objective: Masked LM (MLM)                           • Objective: Next Token Prediction (NTP)
• Best for: Classification, NER, Embeddings            • Best for: Text Generation, Chat, Coding
```

### Key Intuition:
* **Encoders (BERT)** calculate self-attention across the *entire sequence simultaneously* ($A = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$). Every word attends to every other word.
* **Decoders (GPT/Phi)** enforce a lower-triangular attention mask ($-\infty$ on upper triangle) so token $t$ cannot look ahead to token $t+1$.

---

# 2. The Hugging Face Ecosystem & Core APIs

```
┌────────────────────────────────────────────────────────────────────────┐
│                        HUGGING FACE ECOSYSTEM                          │
│                                                                        │
│   datasets ──► Tokenizer ──► Pretrained Model ──► Trainer ──► HF Hub   │
│   (Arrow/JSON) (BPE/WordPiece) (PyTorch/Safetensors) (Optimization)    │
└────────────────────────────────────────────────────────────────────────┘
```

### Essential Tokenizer Functions:
* `AutoTokenizer.from_pretrained(model_name)`: Automatically instantiates the correct tokenizer architecture (e.g., `GPT2TokenizerFast`, `LlamaTokenizer`, `BertTokenizer`).
* `tokenizer(text, truncation=True, max_length=512, padding="max_length")`: Encodes raw text into PyTorch tensors:
  * `input_ids`: Integer token indices in the vocabulary.
  * `attention_mask`: `1` for real tokens, `0` for padding tokens.
* `tokenizer.decode(token_ids, skip_special_tokens=True)`: Reconstructs human-readable text from token IDs.
* `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`: Formats standard `[{"role": "user", ...}, {"role": "assistant", ...}]` lists into model-specific chat template strings.

---

# 3. Encoder Fine-Tuning: BERT for Sequence Classification

```
   Raw Text  ──►  Tokenizer  ──►  [CLS] Token Embedding  ──►  Linear Classifier  ──►  Softmax Probs
 "I love ML!"   [101, 1045, ...]       768-dim vector             num_classes=5         [0.01, 0.95, ...]
```

### Full Annotated Pipeline:
```python
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 5

# 1. Load Tokenizer and Classification Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# 2. Dynamic Batch Padding
# Pads each mini-batch to the longest sequence in that batch (saves memory vs static 512 padding)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 3. Evaluation Metric Function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./bert-classifier",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,            # Small LR for pre-trained weights
    weight_decay=0.01,            # L2 regularization
    warmup_ratio=0.1,             # Linear warmup over first 10% steps
    lr_scheduler_type="linear",
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none"
)
```

---

# 4. Decoder Instruction Fine-Tuning: GPT-2 & Label Loss Masking

In autoregressive instruction tuning, the model must **only be penalized for predicting the assistant's answer**, not for predicting the user prompt!

```
Sequence:  <|user|> What is 2+2? <|assistant|> 4 <|end|>
Tokens:    [ 50257, 2061, 318, 716, 50258,     19, 50256 ]
Labels:    [  -100,  -100, -100, -100,  -100,    19, 50256 ]
             └──────── MASKED (-100) ────────┘ └── LOSS COMPUTED ──┘
```

### Why `-100`?
PyTorch's `nn.CrossEntropyLoss(ignore_index=-100)` ignores all targets set to `-100`. Gradients are not computed for prompt tokens.

### Special Tokens & Embedding Resizing:
```python
# 1. Define custom special tokens
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"

# 2. Add to tokenizer
tokenizer.add_special_tokens({
    "pad_token": tokenizer.eos_token,
    "additional_special_tokens": [USER_TOKEN, ASSISTANT_TOKEN]
})

# 3. MUST resize model embeddings matrix to accommodate new token IDs!
model.resize_token_embeddings(len(tokenizer))
```

### Target Label Masking Function:
```python
def prepare_input(example, max_length=512):
    text = example["text"]
    encoding = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
    
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    labels = [-100] * len(input_ids)
    
    assistant_id = tokenizer.convert_tokens_to_ids(ASSISTANT_TOKEN)
    try:
        assistant_pos = input_ids.index(assistant_id)
        # Compute loss only after the assistant delimiter
        for i in range(assistant_pos + 1, len(input_ids)):
            if attention_mask[i] == 1:
                labels[i] = input_ids[i]
    except ValueError:
        pass
        
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
```

---

# 5. Modern Chat LLMs: Phi-3.5-mini & Chat Templates

Modern models use Jinja-based **Chat Templates** stored directly in `tokenizer_config.json`:

```python
messages = [
    {"role": "system", "content": "You are an expert agronomist."},
    {"role": "user", "content": "How do I treat tomato late blight?"}
]

# Formats dialogue according to model's official syntax:
formatted_text = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)
# Output: "<|system|>\nYou are an expert...\n<|user|>\nHow do I...\n<|assistant|>\n"
```

### Dynamic Batch Collation with `DataCollatorForSeq2Seq`:
Instead of static padding, `DataCollatorForSeq2Seq` dynamically pads both `input_ids` and automatically fills padded label positions with `-100`.

---

# 6. Parameter-Efficient Fine-Tuning (PEFT) & LoRA Mathematics

### The Core Math of LoRA (Low-Rank Adaptation):
Full fine-tuning updates the entire weight matrix $W \in \mathbb{R}^{d \times k}$:
$$W_{new} = W_0 + \Delta W$$

LoRA freezes $W_0$ and decomposes $\Delta W$ into two low-rank matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ where $r \ll \min(d, k)$:
$$h = W_0 x + \frac{\alpha}{r} (B \cdot A) x$$

```
   Input x (dim d)
      │
      ├───► [ Frozen Base Weights W_0 (d x k) ] ───────────┐
      │                                                     ▼
      └───► [ Matrix A (d x r) ] ──► [ Matrix B (r x k) ] ──(+)──► Output h
               (Gaussian Init)           (Zero Init)
```

### Parameter Savings:
* For hidden size $d=4096$, rank $r=8$:
  * Full weights: $4096 \times 4096 = \mathbf{16,777,216}$ parameters.
  * LoRA weights: $(4096 \times 8) + (8 \times 4096) = \mathbf{65,536}$ parameters (**99.6% reduction!**).

### PEFT Code Implementation:
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,                       # Rank dimension
    lora_alpha=16,             # Scaling factor (scaling = alpha / r = 2.0)
    lora_dropout=0.05,         # Dropout probability for LoRA layers
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["qkv_proj", "o_proj"]  # Attention layers to adapt
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 3,028,998 || all params: 3,824,108,544 || trainable%: 0.0792%
```

---

# 7. Quantization & QLoRA with SFTTrainer

### INT8 & INT4 Quantization Mechanics:
Quantization maps 32-bit floating point numbers to discrete low-bit integer bins:
$$X_{\text{quant}} = \text{round}\left(\frac{X}{\text{scale}}\right) + \text{zero\_point}$$

| Precision | Bits per Parameter | Memory for 7B Model | Accuracy Retention |
| :--- | :---: | :---: | :---: |
| **FP32** (Standard) | 32 bits | ~28 GB | 100% |
| **FP16 / BF16** | 16 bits | ~14 GB | 99.9% |
| **INT8 (8-bit)** | 8 bits | ~7 GB | 99.5% |
| **NF4 (4-bit QLoRA)** | 4 bits | ~3.8 GB | 98.5% |

### 8-Bit Quantized Loading with `BitsAndBytesConfig`:
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    "gpt2-xl",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### Supervised Fine-Tuning with `SFTTrainer` (`trl`):
`SFTTrainer` automates sequence packing, dataset mapping, and LoRA injection:
```python
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir="./qlora-alpaca",
    dataset_text_field="formatted_text",
    max_seq_length=512,
    packing=True,              # Packs multiple short examples into 512-token chunks
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
    processing_class=tokenizer
)

trainer.train()
```

---

# 8. Full Deployment Lifecycle: Merging, GGUF & Ollama

```
┌───────────────────────────┐     ┌───────────────────────────┐     ┌──────────────────────────┐
│ 1. Merge LoRA with Base   │ ──► │ 2. Convert to GGUF Binary │ ──► │ 3. Local Ollama Runtime │
│    model.merge_and_unload │     │    llama.cpp conversion   │     │    Modelfile + CLI / API │
└───────────────────────────┘     └───────────────────────────┘     └──────────────────────────┘
```

### Step 1: Fuse Adapters (`merge_and_unload`)
```python
from peft import PeftModel

# Fuse adapter weights directly into base weights: W_merged = W_0 + (alpha/r)*B*A
merged_model = model.merge_and_unload()

# Save complete standalone model
merged_model.save_pretrained("./phi-3.5-mini-merged", safe_serialization=True)
tokenizer.save_pretrained("./phi-3.5-mini-merged")
```

### Step 2: Convert to GGUF using `llama.cpp`
```bash
python3 llama.cpp/convert_hf_to_gguf.py ./phi-3.5-mini-merged \
  --outfile ./phi-3.5-mini-f16.gguf \
  --outtype f16
```

### Step 3: Create Ollama `Modelfile`
```dockerfile
FROM ./phi-3.5-mini-f16.gguf

TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
"""

SYSTEM """You are a helpful, precise, and instruction-aligned AI assistant."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
```

### Step 4: Register & Run in Ollama
```bash
# Build model in Ollama
ollama create phi-3.5-custom -f Modelfile

# Interactive terminal chat
ollama run phi-3.5-custom
```

---

# 9. Master API Cheat Sheet & Function Glossary

| Function / Class | Module | Exact Purpose & Key Arguments |
| :--- | :--- | :--- |
| `AutoTokenizer.from_pretrained()` | `transformers` | Downloads & caches vocabulary, tokenizer configuration, and special tokens. |
| `apply_chat_template()` | `transformers` | Converts structured role-based dialogue into exact tokenized prompt strings. |
| `resize_token_embeddings()` | `transformers` | Expands model embedding weights when new tokens are added (`<|user|>`, etc.). |
| `get_memory_footprint()` | `transformers` | Returns active GPU/RAM memory occupied by model parameters in bytes. |
| `DataCollatorWithPadding` | `transformers` | Dynamically pads mini-batches to max batch length for encoder classification. |
| `DataCollatorForSeq2Seq` | `transformers` | Dynamically pads inputs and fills target label pad positions with `-100`. |
| `LoraConfig` | `peft` | Configures rank ($r$), alpha ($\alpha$), dropout, and target layer names (`qkv_proj`, `c_attn`). |
| `get_peft_model()` | `peft` | Wraps base PyTorch model with LoRA trainable decomposition matrices. |
| `prepare_model_for_kbit_training()`| `peft` | Casts non-trainable layers to FP32 for stable low-bit quantization gradients. |
| `merge_and_unload()` | `peft` | Fuses trained LoRA weights permanently into base model weights. |
| `BitsAndBytesConfig` | `transformers` | Enables 8-bit (`load_in_8bit=True`) or 4-bit (`load_in_4bit=True`, `NF4`) precision. |
| `SFTTrainer` | `trl` | High-level trainer supporting PEFT configs, dataset text fields, and sample packing. |
| `convert_hf_to_gguf.py` | `llama.cpp` | Converts Hugging Face SafeTensors checkpoints to `.gguf` binary format. |
| `ollama create` | CLI | Reads a `Modelfile` and compiles a local Ollama model container. |
| `ollama run` | CLI | Boots local model with GPU/Metal acceleration for real-time interactive streaming. |
