#!/usr/bin/env python3
"""
Generates the Ultimate Masterclass PDF: Deep_Learning_LLM_Complete_Masterclass.pdf
Covers ALL 32 Lessons from scratch Perceptrons, Micrograd, PyTorch, CNNs, WaveNet, RNN/LSTM,
Self-Attention, GPT from Scratch, BERT, GPT-2, Phi-3.5, LoRA, QLoRA, SFTTrainer, GGUF, Ollama.
Formatted like an interactive master notebook with code cells, step-by-step deep explanations,
mathematical proofs, and a complete API encyclopedia.
"""

import sys
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable, Preformatted, PageBreak
)
from reportlab.pdfgen import canvas

# ─────────────────────────────────────────────────────────────────────────────
# NUMBERED CANVAS FOR RUNNING HEADERS & FOOTERS
# ─────────────────────────────────────────────────────────────────────────────
class MasterNumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(MasterNumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super(MasterNumberedCanvas, self).showPage()
        super(MasterNumberedCanvas, self).save()

    def draw_header_footer(self, page_count):
        self.saveState()
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#334155"))
            self.drawString(45, 752, "DEEP LEARNING TO LLMS: COMPLETE MASTER ENCYCLOPEDIA")
            self.drawRightString(567, 752, "COURSE MASTERCLASS • LESSONS 1–32")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(45, 744, 567, 744)

        # Footer (all pages)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(45, 32, "Foundations • Transformers • LoRA • QLoRA • GGUF • Ollama Deployment")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(567, 32, page_str)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(45, 42, 567, 42)
        
        self.restoreState()


def build_master_pdf(filename="Deep_Learning_LLM_Complete_Masterclass.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=50,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()
    
    # Palette
    C_PRIMARY   = colors.HexColor("#0F172A") # Slate 900
    C_ACCENT    = colors.HexColor("#1E40AF") # Blue 800
    C_TEAL      = colors.HexColor("#0F766E") # Teal 700
    C_INDIGO    = colors.HexColor("#4338CA") # Indigo 700
    C_DARK      = colors.HexColor("#1E293B") # Slate 800
    C_TEXT      = colors.HexColor("#334155") # Slate 700
    C_BG_CODE   = colors.HexColor("#F8FAFC") # Slate 50
    C_BORDER    = colors.HexColor("#E2E8F0") # Slate 200
    C_CALLOUT_BG= colors.HexColor("#F0F9FF") # Sky 50
    C_CALLOUT_BD= colors.HexColor("#0284C7") # Sky 600
    C_NOTE_BG   = colors.HexColor("#FDF4FF") # Purple 50
    C_NOTE_BD   = colors.HexColor("#9333EA") # Purple 600

    # Typography
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=C_PRIMARY,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13.5,
        textColor=C_ACCENT,
        spaceAfter=10
    )

    ch_style = ParagraphStyle(
        'ChapterHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13.5,
        leading=17,
        textColor=C_PRIMARY,
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=C_ACCENT,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'Heading3_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.8,
        leading=11.5,
        textColor=C_DARK,
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=C_TEXT,
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.8,
        textColor=C_TEXT,
        leftIndent=10,
        spaceAfter=2.5
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=6.8,
        leading=8.8,
        textColor=colors.HexColor("#0F172A")
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#0369A1")
    )

    tbl_header_style = ParagraphStyle(
        'TblHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white
    )

    tbl_cell_style = ParagraphStyle(
        'TblCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.2,
        textColor=C_TEXT
    )

    tbl_cell_code_style = ParagraphStyle(
        'TblCellCode',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=6.8,
        leading=8.8,
        textColor=colors.HexColor("#1E40AF")
    )

    story = []

    def code_box(code_text):
        p = Preformatted(code_text.strip(), code_style)
        t = Table([[p]], colWidths=[522])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), C_BG_CODE),
            ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        return t

    def callout_box(text, title="Key Architectural Concept"):
        full_text = f"<b>{title}:</b> {text}"
        p = Paragraph(full_text, callout_style)
        t = Table([[p]], colWidths=[522])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), C_CALLOUT_BG),
            ('LINELEFT', (0,0), (0,-1), 3, C_CALLOUT_BD),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#BAE6FD")),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 7),
            ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ]))
        return t

    # ─────────────────────────────────────────────────────────────────────────
    # TITLE & CURRICULUM OVERVIEW
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("📖 Deep Learning to Modern LLMs: The Complete Masterclass", title_style))
    story.append(Paragraph("<b>Comprehensive Course Guide & Notebook:</b> Every Mathematical Foundation, Model Architecture, Code Implementation, and Production Step Explained", subtitle_style))
    
    meta_html = "<b>Scope:</b> Lessons 1–32 (NumPy Foundations &rarr; Micrograd Autograd &rarr; PyTorch &rarr; WaveNet &rarr; Transformers &rarr; BERT &rarr; GPT-2 &rarr; Phi-3.5 &rarr; LoRA &rarr; QLoRA &rarr; GGUF &rarr; Ollama)"
    meta_p = Paragraph(meta_html, ParagraphStyle('MetaText', parent=styles['Normal'], fontName='Helvetica', fontSize=7.2, leading=9.5, textColor=colors.HexColor("#475569")))
    meta_table = Table([[meta_p]], colWidths=[522])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=6))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 1: FOUNDATIONS FROM SCRATCH (LESSONS 1–4)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 1: Neural Networks & Autograd from Scratch (Lessons 1–4)", ch_style))
    story.append(Paragraph(
        "Before modern frameworks, deep learning is governed by linear algebra and recursive calculus. "
        "Every neural network transforms input $x \\in \\mathbb{R}^{d_{in}}$ into predictions $\\hat{y} \\in \\mathbb{R}^{d_{out}}$ via affine mappings followed by non-linear activations:",
        body_style
    ))
    
    story.append(Paragraph("<b>1. Mathematical Formulation of Multi-Class Perceptron:</b>", h2_style))
    story.append(Paragraph(
        "&bull; <b>Affine Transformation:</b> $z = W x + b$ where $W \\in \\mathbb{R}^{C \\times D}$, $b \\in \\mathbb{R}^C$.<br/>"
        "&bull; <b>Softmax Probability Distribution:</b> $P(Y=k | x) = \\frac{e^{z_k}}{\\sum_{j=1}^C e^{z_j}}$ (normalizes logits into valid probabilities $\\sum p_k = 1$).<br/>"
        "&bull; <b>Cross-Entropy Loss Objective:</b> $L = -\\sum_{k=1}^C y_k \\log(\\hat{y}_k)$ where $y$ is the one-hot target vector.<br/>"
        "&bull; <b>Gradient of Loss w.r.t Logits:</b> $\\frac{\\partial L}{\\partial z_k} = \\hat{y}_k - y_k$ (the prediction error directly drives the weight update!).",
        bullet_style
    ))
    
    p1_code = (
        "# Lesson 1: NumPy Softmax & Cross-Entropy from Scratch\n"
        "def softmax(logits):\n"
        "    exp_shifted = np.exp(logits - np.max(logits, axis=1, keepdims=True)) # Numerical stability\n"
        "    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)\n\n"
        "def compute_loss_and_grad(W, b, X, y_onehot):\n"
        "    N = X.shape[0]\n"
        "    probs = softmax(X @ W + b)\n"
        "    loss = -np.sum(y_onehot * np.log(probs + 1e-12)) / N\n"
        "    dlogits = (probs - y_onehot) / N\n"
        "    dW = X.T @ dlogits\n"
        "    db = np.sum(dlogits, axis=0, keepdims=True)\n"
        "    return loss, dW, db"
    )
    story.append(code_box(p1_code))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("<b>2. Micrograd: Reverse-Mode Automatic Differentiation Engine:</b>", h2_style))
    story.append(Paragraph(
        "Every operation builds a directed acyclic graph (DAG). The chain rule $\\frac{\\partial L}{\\partial x} = \\frac{\\partial L}{\\partial z} \\cdot \\frac{\\partial z}{\\partial x}$ is computed in topological order:",
        body_style
    ))
    
    micro_code = (
        "# Lesson 2: Scalar Autograd Engine (Value Class)\n"
        "class Value:\n"
        "    def __init__(self, data, _children=(), _op=''):\n"
        "        self.data, self.grad = float(data), 0.0\n"
        "        self._backward = lambda: None\n"
        "        self._prev, self._op = set(_children), _op\n\n"
        "    def __add__(self, other):\n"
        "        other = other if isinstance(other, Value) else Value(other)\n"
        "        out = Value(self.data + other.data, (self, other), '+')\n"
        "        def _backward():\n"
        "            self.grad += 1.0 * out.grad; other.grad += 1.0 * out.grad\n"
        "        out._backward = _backward\n"
        "        return out\n\n"
        "    def backward(self): # Topological sort backward execution\n"
        "        topo, visited = [], set()\n"
        "        def build(v):\n"
        "            if v not in visited:\n"
        "                visited.add(v)\n"
        "                for child in v._prev: build(child)\n"
        "                topo.append(v)\n"
        "        build(self); self.grad = 1.0\n"
        "        for v in reversed(topo): v._backward()"
    )
    story.append(code_box(micro_code))
    story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 2: PYTORCH ARCHITECTURE & TRAINING (LESSONS 5–8)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 2: PyTorch Core, nn.Module & Training Pipelines (Lessons 5–8)", ch_style))
    story.append(Paragraph(
        "PyTorch abstracts tensor operations onto GPUs/MPS and organizes neural networks into modular classes subclassing `torch.nn.Module`:",
        body_style
    ))
    
    torch_code = (
        "# Lessons 5-8: PyTorch Multi-Layer Perceptron Pipeline\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "from torch.utils.data import Dataset, DataLoader\n\n"
        "class CustomMLP(nn.Module):\n"
        "    def __init__(self, in_features, hidden_dim, out_classes, dropout_p=0.2):\n"
        "        super().__init__()\n"
        "        self.net = nn.Sequential(\n"
        "            nn.Linear(in_features, hidden_dim),\n"
        "            nn.BatchNorm1d(hidden_dim),       # Reduces internal covariate shift\n"
        "            nn.ReLU(),\n"
        "            nn.Dropout(dropout_p),            # Regularization by randomly zeroing units\n"
        "            nn.Linear(hidden_dim, out_classes)\n"
        "        )\n"
        "    def forward(self, x):\n"
        "        return self.net(x)\n\n"
        "# Standard Training Loop\n"
        "def train_epoch(model, dataloader, criterion, optimizer, device):\n"
        "    model.train()\n"
        "    total_loss = 0.0\n"
        "    for x_batch, y_batch in dataloader:\n"
        "        x_batch, y_batch = x_batch.to(device), y_batch.to(device)\n"
        "        optimizer.zero_grad()                 # Clear accumulated gradients\n"
        "        outputs = model(x_batch)\n"
        "        loss = criterion(outputs, y_batch)\n"
        "        loss.backward()                       # Autograd backpropagation\n"
        "        optimizer.step()                      # Update weights via Adam/SGD\n"
        "        total_loss += loss.item() * x_batch.size(0)\n"
        "    return total_loss / len(dataloader.dataset)"
    )
    story.append(code_box(torch_code))
    story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 3: SEQUENCE MODELING & CONVOLUTIONS (LESSONS 9–14)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 3: Sequence Modeling: N-Grams, WaveNet & RNNs (Lessons 9–14)", ch_style))
    story.append(Paragraph(
        "Modeling text requires capturing sequential dependencies. We progress from count-based statistical n-grams to learned continuous representations, dilated causal convolutions, and gated recurrent units:",
        body_style
    ))
    
    story.append(Paragraph(
        "&bull; <b>Character Bigrams (Lesson 9):</b> Computes transition matrix $C[i, j]$ counting how often char $j$ follows char $i$. Sampling uses multinomial sampling on smoothed row probabilities.<br/>"
        "&bull; <b>Bengio et al. MLP (Lesson 10):</b> Replaces sparse one-hot encodings with a learned continuous embedding matrix $C \\in \\mathbb{R}^{|V| \\times d}$. Concatenates $k$ context word embeddings into a dense hidden layer.<br/>"
        "&bull; <b>WaveNet (Lesson 12):</b> Uses <b>Dilated Causal Convolutions</b> where the dilation rate doubles exponentially ($d=1, 2, 4, 8, \\dots$), expanding the receptive field exponentially without losing temporal resolution.",
        bullet_style
    ))
    
    seq_code = (
        "# Lesson 12 & 14: Dilated Conv & LSTM Cell Mechanics\n"
        "# 1. WaveNet Dilated Causal Convolution Block:\n"
        "class WaveNetBlock(nn.Module):\n"
        "    def __init__(self, channels, dilation):\n"
        "        super().__init__()\n"
        "        self.conv = nn.Conv1d(channels, channels * 2, kernel_size=2, dilation=dilation, padding=dilation)\n"
        "        self.res_out = nn.Conv1d(channels, channels, kernel_size=1)\n"
        "    def forward(self, x):\n"
        "        out = self.conv(x)[:, :, :-self.conv.padding[0]] # Causal slice\n"
        "        f, g = torch.chunk(out, 2, dim=1)\n"
        "        gate = torch.tanh(f) * torch.sigmoid(g)           # Gated Activation Unit\n"
        "        return x + self.res_out(gate)                     # Residual connection\n\n"
        "# 2. LSTM Gated Cell Equations:\n"
        "# f_t = sigma(W_f x_t + U_f h_{t-1} + b_f)  [Forget Gate: What to discard from memory]\n"
        "# i_t = sigma(W_i x_t + U_i h_{t-1} + b_i)  [Input Gate: What new information to store]\n"
        "# C_t = f_t * C_{t-1} + i_t * tanh(W_c x_t) [Cell State: Uninterrupted gradient highway]\n"
        "# o_t = sigma(W_o x_t + U_o h_{t-1} + b_o)  [Output Gate: Filtered hidden state]"
    )
    story.append(code_box(seq_code))
    story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 4: TRANSFORMER ARCHITECTURE (LESSONS 15–22)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 4: Scaled Dot-Product Attention & GPT from Scratch (Lessons 15–22)", ch_style))
    story.append(Paragraph(
        "Transformers replace recurrence entirely with the <b>Scaled Dot-Product Self-Attention Mechanism</b>, allowing every token to attend to all other tokens in parallel:",
        body_style
    ))
    
    attn_eq = (
        "                    ATTENTION MECHANISM & CAUSAL MASK\n\n"
        "      Attention(Q, K, V) = softmax( (Q · K^T) / sqrt(d_k) + M ) · V\n\n"
        "   Where M is the Causal Attention Mask:\n"
        "          [ 0,  -inf, -inf, -inf ]\n"
        "      M = [ 0,   0,   -inf, -inf ]  <-- Forces position t to attend ONLY\n"
        "          [ 0,   0,    0,   -inf ]      to past positions <= t\n"
        "          [ 0,   0,    0,    0   ]"
    )
    story.append(code_box(attn_eq))
    story.append(Spacer(1, 4))
    
    gpt_code = (
        "# Lessons 16-19: Multi-Head Self-Attention and Full GPT Block\n"
        "class CausalSelfAttention(nn.Module):\n"
        "    def __init__(self, d_model, n_heads, block_size):\n"
        "        super().__init__()\n"
        "        self.n_heads, self.head_dim = n_heads, d_model // n_heads\n"
        "        self.qkv_proj = nn.Linear(d_model, 3 * d_model)\n"
        "        self.out_proj = nn.Linear(d_model, d_model)\n"
        "        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size))\n"
        "                                           .view(1, 1, block_size, block_size))\n\n"
        "    def forward(self, x):\n"
        "        B, T, C = x.shape\n"
        "        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)\n"
        "        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # (B, H, T, D)\n"
        "        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)\n"
        "        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)\n\n"
        "        # Scaled Dot Product with Causal Mask\n"
        "        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)\n"
        "        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))\n"
        "        weights = F.softmax(scores, dim=-1)\n"
        "        out = weights @ v\n"
        "        out = out.transpose(1, 2).contiguous().view(B, T, C)\n"
        "        return self.out_proj(out)\n\n"
        "class TransformerBlock(nn.Module):\n"
        "    def __init__(self, d_model, n_heads, block_size, mlp_ratio=4):\n"
        "        super().__init__()\n"
        "        self.ln1 = nn.LayerNorm(d_model) # Pre-LN architecture\n"
        "        self.attn = CausalSelfAttention(d_model, n_heads, block_size)\n"
        "        self.ln2 = nn.LayerNorm(d_model)\n"
        "        self.mlp = nn.Sequential(\n"
        "            nn.Linear(d_model, d_model * mlp_ratio),\n"
        "            nn.GELU(),\n"
        "            nn.Linear(d_model * mlp_ratio, d_model)\n"
        "        )\n"
        "    def forward(self, x):\n"
        "        x = x + self.attn(self.ln1(x))    # Residual connection 1\n"
        "        x = x + self.mlp(self.ln2(x))     # Residual connection 2\n"
        "        return x"
    )
    story.append(code_box(gpt_code))
    story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 5: HUGGING FACE & BERT ENCODER (LESSONS 23 & 25)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 5: Hugging Face Ecosystem & BERT Fine-Tuning (Lessons 23 & 25)", ch_style))
    story.append(Paragraph(
        "BERT (Bidirectional Encoder Representations from Transformers) learns deep bidirectional representations. "
        "For classification, the first token `[CLS]` aggregates sentence-level context and is projected into class probabilities:",
        body_style
    ))
    
    bert_pipe = (
        "# Lesson 25: BERT Classification Pipeline with Dynamic Batch Padding\n"
        "from transformers import (AutoTokenizer, AutoModelForSequenceClassification,\n"
        "                          DataCollatorWithPadding, Trainer, TrainingArguments)\n\n"
        "tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')\n"
        "model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)\n\n"
        "# Dynamic batch padding collator: Pads ONLY to longest item in batch (saves 40-60% VRAM!)\n"
        "collator = DataCollatorWithPadding(tokenizer=tokenizer)\n\n"
        "training_args = TrainingArguments(\n"
        "    output_dir='./bert-classifier',\n"
        "    num_train_epochs=3,\n"
        "    per_device_train_batch_size=16,\n"
        "    learning_rate=2e-5,          # Low learning rate preserves pretrained weights\n"
        "    weight_decay=0.01,          # L2 weight regularization\n"
        "    warmup_ratio=0.1,           # Linear learning rate warmup over initial 10% steps\n"
        "    eval_strategy='epoch',\n"
        "    save_strategy='epoch',\n"
        "    fp16=torch.cuda.is_available(),\n"
        "    report_to='none'\n"
        ")\n"
        "trainer = Trainer(model=model, args=training_args, train_dataset=ds_train, eval_dataset=ds_val, data_collator=collator)\n"
        "trainer.train()"
    )
    story.append(code_box(bert_pipe))
    story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 6: INSTRUCTION TUNING & LABEL LOSS MASKING (LESSON 26)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 6: GPT-2 Instruction Tuning & Label Loss Masking (Lesson 26)", ch_style))
    story.append(Paragraph(
        "In instruction tuning, the model must **never be penalized for predicting the prompt tokens**; loss must be computed exclusively on the target assistant completion. This is achieved using PyTorch's `ignore_index=-100`:",
        body_style
    ))
    
    story.append(callout_box(
        "<b>Mathematical Reason for -100:</b> PyTorch's <code>nn.CrossEntropyLoss(ignore_index=-100)</code> unconditionally zeroes out the loss contribution for any index with label <code>-100</code>. No gradient flows back from prompt tokens, so the model learns purely to formulate responses given prompts.",
        "Crucial Target Masking Principle"
    ))
    story.append(Spacer(1, 4))
    
    gpt2_inst = (
        "# Lesson 26: Custom Delimiters, Token Resizing & Target Masking\n"
        "USER_TOKEN, ASSISTANT_TOKEN = '<|user|>', '<|assistant|>'\n"
        "tokenizer.add_special_tokens({\n"
        "    'pad_token': tokenizer.eos_token,\n"
        "    'additional_special_tokens': [USER_TOKEN, ASSISTANT_TOKEN]\n"
        "})\n"
        "# MUST resize embedding matrix so vocabulary size matches model weights!\n"
        "model.resize_token_embeddings(len(tokenizer))\n\n"
        "def tokenize_and_mask_labels(example, max_len=512):\n"
        "    encoding = tokenizer(example['text'], truncation=True, max_length=max_len, padding='max_length')\n"
        "    input_ids = encoding['input_ids']\n"
        "    attention_mask = encoding['attention_mask']\n"
        "    labels = [-100] * len(input_ids) # Initialize all labels to -100 (ignored)\n\n"
        "    assistant_id = tokenizer.convert_tokens_to_ids(ASSISTANT_TOKEN)\n"
        "    if assistant_id in input_ids:\n"
        "        start_idx = input_ids.index(assistant_id) + 1\n"
        "        for i in range(start_idx, len(input_ids)):\n"
        "            if attention_mask[i] == 1:\n"
        "                labels[i] = input_ids[i] # Compute loss ONLY on real assistant tokens!\n"
        "    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}"
    )
    story.append(code_box(gpt2_inst))
    story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 7: CHAT TEMPLATES & PHI-3.5 (LESSON 27)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 7: Modern Chat LLMs & Jinja Chat Templates (Lesson 27)", ch_style))
    story.append(Paragraph(
        "Modern LLMs (e.g., Microsoft Phi-3.5-mini-instruct, Llama-3) standardize multi-turn conversational dialogue using Jinja2 Chat Templates stored directly in `tokenizer_config.json`:",
        body_style
    ))
    
    phi_inst = (
        "# Lesson 27: Applying Native Chat Templates & Seq2Seq Dynamic Collation\n"
        "from transformers import DataCollatorForSeq2Seq\n\n"
        "messages = [\n"
        "    {'role': 'system', 'content': 'You are an expert diagnostic assistant.'},\n"
        "    {'role': 'user', 'content': 'What causes chlorosis in citrus leaves?'}\n"
        "]\n"
        "# Automatically renders exact special tokens: <|system|>...<|end|><|user|>...<|end|><|assistant|>\n"
        "prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n\n"
        "# DataCollatorForSeq2Seq dynamically pads batch AND replaces label padding with -100:\n"
        "data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding='longest')"
    )
    story.append(code_box(phi_inst))
    story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 8: PEFT & LORA MATHEMATICS (LESSON 28)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 8: Parameter-Efficient Fine-Tuning (PEFT) & LoRA Math (Lesson 28)", ch_style))
    story.append(Paragraph(
        "Full fine-tuning of multi-billion parameter models is computationally prohibitive. "
        "<b>LoRA (Low-Rank Adaptation)</b> freezes the pretrained base weight $W_0 \\in \\mathbb{R}^{d \\times k}$ and factorizes updates into two low-rank matrices $B \\in \\mathbb{R}^{d \\times r}$ and $A \\in \\mathbb{R}^{r \\times k}$ where $r \\ll \\min(d, k)$:",
        body_style
    ))
    
    lora_math_box = (
        "                          LORA MATHEMATICAL FORMULATION\n\n"
        "           h = W_0 · x + Delta_W · x  ==>  h = W_0 · x + (alpha / r) * (B · A) · x\n\n"
        "   1. Base Weight W_0: FROZEN during training (no gradient computation).\n"
        "   2. Matrix A (d x r): Initialized with Gaussian distribution N(0, sigma^2).\n"
        "   3. Matrix B (r x k): Initialized to EXACT ZERO so Delta_W = B · A = 0 at step 0.\n"
        "   4. Scaling Factor (alpha / r): Stabilizes training when rank r is modified."
    )
    story.append(code_box(lora_math_box))
    story.append(Spacer(1, 4))
    
    lora_code = (
        "# Lesson 28: PEFT LoRA Adapter Setup\n"
        "from peft import LoraConfig, get_peft_model, TaskType\n\n"
        "lora_config = LoraConfig(\n"
        "    r=8,                                  # Rank dimension\n"
        "    lora_alpha=16,                        # Scaling factor (multiplier = alpha/r = 2.0)\n"
        "    lora_dropout=0.05,                    # Regularization dropout on LoRA layers\n"
        "    task_type=TaskType.CAUSAL_LM,\n"
        "    target_modules=['qkv_proj', 'o_proj'],# Target attention projections\n"
        "    bias='none'\n"
        ")\n"
        "peft_model = get_peft_model(base_model, lora_config)\n"
        "peft_model.print_trainable_parameters()\n"
        "# Trainable params: 3,028,998 || All params: 3,824,108,544 || Trainable%: 0.0792% (>99.9% savings!)"
    )
    story.append(code_box(lora_code))
    story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 9: QUANTIZATION & QLORA (LESSON 29)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 9: Quantization & QLoRA with SFTTrainer (Lesson 29)", ch_style))
    story.append(Paragraph(
        "Quantization compresses continuous 32-bit floating point parameters into lower precision discrete integer bins ($X_{\\text{quant}} = \\text{round}(X / \\text{scale}) + \\text{zero\\_point}$):",
        body_style
    ))
    
    # Quantization comparison table
    q_data = [
        [Paragraph("<b>Precision</b>", tbl_header_style), Paragraph("<b>Bits / Param</b>", tbl_header_style), Paragraph("<b>VRAM (7B Model)</b>", tbl_header_style), Paragraph("<b>Quantization Mechanism</b>", tbl_header_style)],
        [Paragraph("<b>FP32</b> (Standard)", tbl_cell_style), Paragraph("32 bits", tbl_cell_style), Paragraph("~28.0 GB", tbl_cell_style), Paragraph("Full single precision IEEE-754 floating point.", tbl_cell_style)],
        [Paragraph("<b>FP16 / BF16</b>", tbl_cell_style), Paragraph("16 bits", tbl_cell_style), Paragraph("~14.0 GB", tbl_cell_style), Paragraph("Half precision; BF16 preserves 8-bit dynamic range exponent.", tbl_cell_style)],
        [Paragraph("<b>INT8 (8-bit)</b>", tbl_cell_style), Paragraph("8 bits", tbl_cell_style), Paragraph("~7.0 GB", tbl_cell_style), Paragraph("Vector-wise quantization with outlier thresholding (BitsAndBytes).", tbl_cell_style)],
        [Paragraph("<b>NF4 (4-bit QLoRA)</b>", tbl_cell_style), Paragraph("4 bits", tbl_cell_style), Paragraph("~3.8 GB", tbl_cell_style), Paragraph("Information-theoretically optimal NormalFloat4 for Gaussian weights.", tbl_cell_style)],
    ]
    q_tbl = Table(q_data, colWidths=[110, 80, 110, 222])
    q_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_ACCENT),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(q_tbl)
    story.append(Spacer(1, 4))
    
    qlora_code = (
        "# Lesson 29: 8-Bit Quantized Loading & SFTTrainer with Sample Packing\n"
        "from transformers import BitsAndBytesConfig, AutoModelForCausalLM\n"
        "from trl import SFTConfig, SFTTrainer\n\n"
        "# 1. Configure 8-bit quantization\n"
        "bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)\n"
        "model = AutoModelForCausalLM.from_pretrained('gpt2-xl', quantization_config=bnb_config, device_map='auto')\n\n"
        "# 2. Configure Supervised Fine-Tuning\n"
        "sft_args = SFTConfig(\n"
        "    output_dir='./qlora-alpaca',\n"
        "    dataset_text_field='formatted_text',\n"
        "    max_seq_length=512,\n"
        "    packing=True,                   # Packs multiple short examples into 512 chunks (3-5x faster!)\n"
        "    num_train_epochs=3,\n"
        "    per_device_train_batch_size=4,\n"
        "    gradient_accumulation_steps=4, # Effective batch size = 16\n"
        "    learning_rate=2e-4,\n"
        "    logging_steps=10\n"
        ")\n"
        "trainer = SFTTrainer(model=model, args=sft_args, train_dataset=dataset, peft_config=lora_config, processing_class=tokenizer)\n"
        "trainer.train()"
    )
    story.append(code_box(qlora_code))
    story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 10: PRODUCTION DEPLOYMENT (LESSONS 30–32)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 10: Local Production: Merging, GGUF & Ollama (Lessons 30–32)", ch_style))
    story.append(Paragraph(
        "Deploying fine-tuned models for local, low-latency edge inference involves fusing LoRA weights, converting into single-file binary GGUF format with `llama.cpp`, and packaging with Ollama:",
        body_style
    ))
    
    prod_pipe = (
        "# Step 1: Fuse LoRA adapter into base model weights\n"
        "# W_merged = W_0 + (alpha / r) * B * A\n"
        "merged_model = model.merge_and_unload()\n"
        "merged_model.save_pretrained('./phi-3.5-mini-merged', safe_serialization=True)\n"
        "tokenizer.save_pretrained('./phi-3.5-mini-merged')\n\n"
        "# Step 2: Convert to GGUF binary format using llama.cpp\n"
        "# python3 llama.cpp/convert_hf_to_gguf.py ./phi-3.5-mini-merged --outfile ./phi-3.5-f16.gguf --outtype f16\n\n"
        "# Step 3: Ollama Modelfile Definition\n"
        "# FROM ./phi-3.5-f16.gguf\n"
        "# TEMPLATE \"\"\"{{ if .System }}<|system|>\n"
        "# {{ .System }}<|end|>\n"
        "# {{ end }}{{ if .Prompt }}<|user|>\n"
        "# {{ .Prompt }}<|end|>\n"
        "# {{ end }}<|assistant|>\n"
        "# \"\"\"\n"
        "# SYSTEM \"\"\"You are an expert AI assistant specialized in instruction-following.\"\"\"\n"
        "# PARAMETER temperature 0.7\n"
        "# PARAMETER stop \"<|end|>\"\n\n"
        "# Step 4: Register and run in Ollama\n"
        "# ollama create phi-3.5-custom -f Modelfile\n"
        "# ollama run phi-3.5-custom"
    )
    story.append(code_box(prod_pipe))
    story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # CHAPTER 11: COMPLETE MASTER FUNCTION & API ENCYCLOPEDIA
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Chapter 11: Master Function, Class & Method Encyclopedia", ch_style))
    story.append(Paragraph(
        "Complete reference guide covering every critical class, function, method, and hyperparameter across the entire curriculum:",
        body_style
    ))
    
    encyclopedia_data = [
        [Paragraph("<b>Function / Class</b>", tbl_header_style), Paragraph("<b>Library</b>", tbl_header_style), Paragraph("<b>Exact Purpose, Parameters &amp; Best Practices</b>", tbl_header_style)],
        [Paragraph("<code>Value(data)</code>", tbl_cell_code_style), Paragraph("Custom Autograd", tbl_cell_style), Paragraph("Scalar autograd node tracking computational graph operations (<code>+</code>, <code>*</code>, <code>tanh</code>) and computing backward derivatives.", tbl_cell_style)],
        [Paragraph("<code>nn.Linear(in, out)</code>", tbl_cell_code_style), Paragraph("PyTorch", tbl_cell_style), Paragraph("Applies affine transformation $y = x W^T + b$. Initializes weights with Kaiming uniform by default.", tbl_cell_style)],
        [Paragraph("<code>nn.BatchNorm1d(dim)</code>", tbl_cell_code_style), Paragraph("PyTorch", tbl_cell_style), Paragraph("Normalizes mini-batch activations to mean 0 and variance 1 with learnable scale $\\gamma$ and shift $\\beta$.", tbl_cell_style)],
        [Paragraph("<code>Conv1d(..., dilation)</code>", tbl_cell_code_style), Paragraph("PyTorch", tbl_cell_style), Paragraph("1D Convolution layer. Setting dilation $>1$ expands receptive field exponentially for WaveNet sequence modeling.", tbl_cell_style)],
        [Paragraph("<code>AutoTokenizer</code>", tbl_cell_code_style), Paragraph("transformers", tbl_cell_style), Paragraph("Instantiates the exact tokenizer architecture from model configuration (BPE, WordPiece, SentencePiece).", tbl_cell_style)],
        [Paragraph("<code>apply_chat_template()</code>", tbl_cell_code_style), Paragraph("transformers", tbl_cell_style), Paragraph("Applies model's official Jinja template to role-based message lists (system/user/assistant).", tbl_cell_style)],
        [Paragraph("<code>resize_token_embeddings()</code>", tbl_cell_code_style), Paragraph("transformers", tbl_cell_style), Paragraph("Resizes model vocabulary matrix when custom special tokens are registered to prevent index out of bounds.", tbl_cell_style)],
        [Paragraph("<code>get_memory_footprint()</code>", tbl_cell_code_style), Paragraph("transformers", tbl_cell_style), Paragraph("Returns active VRAM/RAM consumed by model weights in bytes to verify quantization savings.", tbl_cell_style)],
        [Paragraph("<code>DataCollatorWithPadding</code>", tbl_cell_code_style), Paragraph("transformers", tbl_cell_style), Paragraph("Pads mini-batch tensors dynamically to the maximum sequence length of that specific batch.", tbl_cell_style)],
        [Paragraph("<code>DataCollatorForSeq2Seq</code>", tbl_cell_code_style), Paragraph("transformers", tbl_cell_style), Paragraph("Dynamically pads inputs and fills target label pad positions with -100 to ignore loss on padding tokens.", tbl_cell_style)],
        [Paragraph("<code>LoraConfig</code>", tbl_cell_code_style), Paragraph("peft", tbl_cell_style), Paragraph("Defines low-rank decomposition rank $r$, scaling factor $\\alpha$, dropout, and target layers (<code>qkv_proj</code>).", tbl_cell_style)],
        [Paragraph("<code>get_peft_model()</code>", tbl_cell_code_style), Paragraph("peft", tbl_cell_style), Paragraph("Injects trainable LoRA low-rank decomposition matrices while freezing all base model parameters.", tbl_cell_style)],
        [Paragraph("<code>merge_and_unload()</code>", tbl_cell_code_style), Paragraph("peft", tbl_cell_style), Paragraph("Fuses trained LoRA delta weights permanently into base model weights: $W_{new} = W_0 + \\frac{\\alpha}{r}BA$.", tbl_cell_style)],
        [Paragraph("<code>BitsAndBytesConfig</code>", tbl_cell_code_style), Paragraph("transformers", tbl_cell_style), Paragraph("Enables 8-bit vector quantization (<code>load_in_8bit</code>) or 4-bit NormalFloat4 (<code>load_in_4bit</code>) loading.", tbl_cell_style)],
        [Paragraph("<code>SFTTrainer</code>", tbl_cell_code_style), Paragraph("trl", tbl_cell_style), Paragraph("High-level Supervised Fine-Tuning trainer with support for sample packing, dataset text fields, and LoRA.", tbl_cell_style)],
        [Paragraph("<code>convert_hf_to_gguf.py</code>", tbl_cell_code_style), Paragraph("llama.cpp", tbl_cell_style), Paragraph("Translates Hugging Face SafeTensors checkpoints into single-file binary GGUF format for CPU/Metal/GPU.", tbl_cell_style)],
        [Paragraph("<code>ollama create / run</code>", tbl_cell_code_style), Paragraph("Ollama CLI", tbl_cell_style), Paragraph("Compiles a local containerized model from a <code>Modelfile</code> and executes real-time streaming inference.", tbl_cell_style)],
    ]
    
    ency_tbl = Table(encyclopedia_data, colWidths=[140, 78, 304])
    ency_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(ency_tbl)
    
    # Build Document
    doc.build(story, canvasmaker=MasterNumberedCanvas)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    build_master_pdf()
