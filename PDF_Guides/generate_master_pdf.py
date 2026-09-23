#!/usr/bin/env python3
"""
Generates the comprehensive PDF guide: Transformers_FineTuning_LLM_Master_Guide.pdf
Covers Hugging Face Transformers, BERT, GPT-2, Phi-3.5, LoRA, QLoRA, SFTTrainer, GGUF, Ollama.
"""

import sys
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable, Preformatted
)
from reportlab.pdfgen import canvas

# ─────────────────────────────────────────────────────────────────────────────
# NUMBERED CANVAS FOR HEADER & FOOTER
# ─────────────────────────────────────────────────────────────────────────────
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_header_footer(self, page_count):
        self.saveState()
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#4A5568"))
            self.drawString(54, 750, "TRANSFORMERS, LoRA, QLoRA & LLM DEPLOYMENT MASTER GUIDE")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)

        # Footer (all pages)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(54, 38, "Deep Learning Course Masterclass • Comprehensive Reference")
        
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 38, page_str)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        
        self.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# BUILD SCRIPT
# ─────────────────────────────────────────────────────────────────────────────
def build_pdf(filename="Transformers_FineTuning_LLM_Master_Guide.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    C_PRIMARY   = colors.HexColor("#0F172A") # Slate 900
    C_ACCENT    = colors.HexColor("#1E40AF") # Blue 800
    C_TEAL      = colors.HexColor("#0D9488") # Teal 600
    C_DARK      = colors.HexColor("#1E293B") # Slate 800
    C_TEXT      = colors.HexColor("#334155") # Slate 700
    C_BG_CODE   = colors.HexColor("#F8FAFC") # Slate 50
    C_BORDER    = colors.HexColor("#E2E8F0") # Slate 200
    C_CALLOUT_BG= colors.HexColor("#EFF6FF") # Blue 50
    C_CALLOUT_BD= colors.HexColor("#3B82F6") # Blue 500

    # Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=C_PRIMARY,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=C_ACCENT,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=C_ACCENT,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=C_DARK,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=C_TEXT,
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=C_TEXT,
        leftIndent=12,
        spaceAfter=3
    )

    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#0F172A")
    )

    callout_style = ParagraphStyle(
        'CalloutText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.2,
        leading=11.5,
        textColor=colors.HexColor("#1E3A8A")
    )

    tbl_header_style = ParagraphStyle(
        'TblHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    tbl_cell_style = ParagraphStyle(
        'TblCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=10,
        textColor=C_TEXT
    )

    tbl_cell_code_style = ParagraphStyle(
        'TblCellCode',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#1E40AF")
    )

    story = []

    def code_box(code_text):
        p = Preformatted(code_text.strip(), code_style)
        t = Table([[p]], colWidths=[504])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), C_BG_CODE),
            ('BOX', (0,0), (-1,-1), 0.5, C_BORDER),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 7),
            ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ]))
        return t

    def callout_box(text):
        p = Paragraph(text, callout_style)
        t = Table([[p]], colWidths=[504])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), C_CALLOUT_BG),
            ('LINELEFT', (0,0), (0,-1), 3, C_CALLOUT_BD),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#BFDBFE")),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        return t

    # ─────────────────────────────────────────────────────────────────────────
    # HEADER & TITLE BLOCK
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 4))
    story.append(Paragraph("🚀 Transformers, LoRA, QLoRA & LLM Deployment", title_style))
    story.append(Paragraph("<b>The Complete Master Reference Guide:</b> From Hugging Face Fundamentals to High-Performance Local Production with Ollama", subtitle_style))
    
    # Metadata Badge Box
    meta_html = "<b>Target Models:</b> BERT, GPT-2, Microsoft Phi-3.5-mini-instruct, Llama &bull; <b>Key Stack:</b> transformers, datasets, peft, trl, bitsandbytes, llama.cpp, ollama"
    meta_p = Paragraph(meta_html, ParagraphStyle('MetaText', parent=styles['Normal'], fontName='Helvetica', fontSize=7.5, leading=10.5, textColor=colors.HexColor("#475569")))
    meta_table = Table([[meta_p]], colWidths=[504])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F1F5F9")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=8))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1: ARCHITECTURAL FOUNDATIONS
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("1. Core Architectural Foundations: Encoders vs. Decoders", h1_style))
    story.append(Paragraph(
        "Transformers fall into two primary structural families based on attention mechanics and objective formulation:",
        body_style
    ))
    
    arch_diagram = (
        "                    TRANSFORMER TAXONOMY\n"
        "                             │\n"
        "         ┌───────────────────┴───────────────────┐\n"
        "         ▼                                       ▼\n"
        "   ENCODER-ONLY                            DECODER-ONLY\n"
        " (e.g., BERT, RoBERTa)                 (e.g., GPT-2, Phi-3.5, Llama)\n"
        " ─────────────────────                 ─────────────────────────────\n"
        " • Bidirectional Attention             • Causal (Unidirectional) Masked Attention\n"
        " • Attends to past + future            • Attends strictly to past tokens (autoregressive)\n"
        " • Training: Masked LM (MLM)           • Training: Next Token Prediction (NTP / Causal LM)\n"
        " • Best: Classification, NER, Vectors  • Best: Text Generation, Reasoning, Chatbots"
    )
    story.append(code_box(arch_diagram))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("<b>Mathematical Attention Intuition:</b>", h2_style))
    story.append(Paragraph(
        "&bull; <b>Encoders (BERT):</b> Self-attention is calculated over all $N$ tokens simultaneously: "
        "<code>Attention(Q,K,V) = softmax((Q K^T) / sqrt(d_k)) V</code>. Every token directly attends to every other token.<br/>"
        "&bull; <b>Decoders (GPT/Phi):</b> Enforces a lower-triangular causal attention mask ($-\\infty$ in the upper triangle before softmax). Token $t$ cannot look ahead to token $t+1$, ensuring causal autoregressive generation.",
        bullet_style
    ))
    story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: HUGGING FACE ECOSYSTEM & CORE APIS
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("2. The Hugging Face Ecosystem & Core APIs", h1_style))
    story.append(Paragraph(
        "The Hugging Face pipeline forms a standardized workflow from data ingest to tokenization, model loading, and evaluation:",
        body_style
    ))
    
    hf_pipe = (
        "┌────────────────────────────────────────────────────────────────────────┐\n"
        "│ datasets ──► AutoTokenizer ──► Pretrained Model ──► Trainer ──► HF Hub  │\n"
        "│ (Arrow/JSON) (BPE/WordPiece)   (PyTorch/SafeTensors) (Optimization)    │\n"
        "└────────────────────────────────────────────────────────────────────────┘"
    )
    story.append(code_box(hf_pipe))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("<b>Core Tokenizer Outputs:</b>", h2_style))
    story.append(Paragraph(
        "&bull; <code>input_ids</code>: Integer IDs mapping tokens to vocab indices (e.g. <code>[101, 2054, 102]</code>).<br/>"
        "&bull; <code>attention_mask</code>: 1 for authentic tokens, 0 for padded filler tokens.<br/>"
        "&bull; <code>apply_chat_template()</code>: Uses Jinja2 templates stored in <code>tokenizer_config.json</code> to apply precise conversational role delimiters (e.g., <code>&lt;|user|&gt;</code>, <code>&lt;|assistant|&gt;</code>).",
        bullet_style
    ))
    story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3: ENCODER FINE-TUNING (BERT)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("3. Encoder Fine-Tuning: BERT for Sequence Classification", h1_style))
    story.append(Paragraph(
        "For classification tasks, BERT extracts the pooled <code>[CLS]</code> token embedding (768-dim vector) and passes it to a linear feedforward head with cross-entropy loss over target labels:",
        body_style
    ))
    
    bert_code = (
        "import torch\n"
        "from transformers import (AutoTokenizer, AutoModelForSequenceClassification,\n"
        "                          DataCollatorWithPadding, Trainer, TrainingArguments)\n"
        "from sklearn.metrics import accuracy_score, f1_score\n\n"
        "tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')\n"
        "model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)\n\n"
        "# Dynamic mini-batch padding (pads each batch only to its longest item)\n"
        "data_collator = DataCollatorWithPadding(tokenizer=tokenizer)\n\n"
        "def compute_metrics(eval_pred):\n"
        "    logits, labels = eval_pred\n"
        "    preds = logits.argmax(axis=-1)\n"
        "    return {'acc': accuracy_score(labels, preds), 'f1': f1_score(labels, preds, average='macro')}\n\n"
        "training_args = TrainingArguments(\n"
        "    output_dir='./bert_classifier',\n"
        "    num_train_epochs=3,\n"
        "    per_device_train_batch_size=16,\n"
        "    learning_rate=2e-5,          # Standard fine-tuning learning rate\n"
        "    weight_decay=0.01,          # L2 regularization\n"
        "    warmup_ratio=0.1,           # Linear warmup over first 10% steps\n"
        "    eval_strategy='epoch',\n"
        "    save_strategy='epoch',\n"
        "    fp16=torch.cuda.is_available(),\n"
        "    report_to='none'\n"
        ")"
    )
    story.append(code_box(bert_code))
    story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4: DECODER INSTRUCTION TUNING & LABEL MASKING
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("4. Decoder Instruction Fine-Tuning: GPT-2 & Label Loss Masking", h1_style))
    story.append(Paragraph(
        "In instruction tuning, the model must <b>only be penalized for predicting the assistant's response</b>, not the user prompt. We mask prompt token labels with <code>-100</code>:",
        body_style
    ))
    
    masking_diagram = (
        "Tokens:  [ <|user|>, What, is, 2+2?, <|assistant|>,  4,  <|end|> ]\n"
        "Labels:  [   -100,    -100, -100, -100,     -100,     4,   <|end|> ]\n"
        "         └────────── MASKED (-100) ─────────────┘ └── LOSS COMPUTED ──┘"
    )
    story.append(code_box(masking_diagram))
    story.append(Spacer(1, 4))
    
    story.append(callout_box(
        "<b>Why -100?</b> PyTorch's <code>nn.CrossEntropyLoss(ignore_index=-100)</code> ignores targets set to -100 by default. Zero gradient is backpropagated through prompt tokens."
    ))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("<b>Adding Special Tokens & Embedding Resizing:</b>", h2_style))
    gpt2_code = (
        "# 1. Define and add custom tokens\n"
        "USER_TOKEN, ASSISTANT_TOKEN = '<|user|>', '<|assistant|>'\n"
        "tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token, 'additional_special_tokens': [USER_TOKEN, ASSISTANT_TOKEN]})\n\n"
        "# 2. Critical: resize model embedding matrix to fit new vocab size!\n"
        "model.resize_token_embeddings(len(tokenizer))\n\n"
        "# 3. Target Label Masking Function\n"
        "def prepare_input(example, max_length=512):\n"
        "    encoding = tokenizer(example['text'], truncation=True, max_length=max_length, padding='max_length')\n"
        "    labels = [-100] * len(encoding['input_ids'])\n"
        "    assistant_id = tokenizer.convert_tokens_to_ids(ASSISTANT_TOKEN)\n"
        "    if assistant_id in encoding['input_ids']:\n"
        "        pos = encoding['input_ids'].index(assistant_id)\n"
        "        for i in range(pos + 1, len(encoding['input_ids'])):\n"
        "            if encoding['attention_mask'][i] == 1:\n"
        "                labels[i] = encoding['input_ids'][i]\n"
        "    return {'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask'], 'labels': labels}"
    )
    story.append(code_box(gpt2_code))
    story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5: MODERN CHAT LLMS & TEMPLATES
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("5. Modern Chat LLMs: Phi-3.5-mini & Chat Templates", h1_style))
    story.append(Paragraph(
        "Modern instruction-tuned models (e.g. Microsoft Phi-3.5-mini, Llama-3) use Jinja templates and dynamic sequence-to-sequence collation:",
        body_style
    ))
    
    phi_code = (
        "messages = [\n"
        "    {'role': 'system', 'content': 'You are an agricultural expert.'},\n"
        "    {'role': 'user', 'content': 'How do I identify and treat powdery mildew?'}\n"
        "]\n"
        "prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n"
        "# Output: '<|system|>\\nYou are...<|end|>\\n<|user|>\\nHow do...<|end|>\\n<|assistant|>\\n'\n\n"
        "# Dynamic Seq2Seq batch padding (automatically converts pad positions in labels to -100)\n"
        "from transformers import DataCollatorForSeq2Seq\n"
        "collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding='longest')"
    )
    story.append(code_box(phi_code))
    story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 6: PEFT & LORA MATHEMATICS
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("6. Parameter-Efficient Fine-Tuning (PEFT) & LoRA Mathematics", h1_style))
    story.append(Paragraph(
        "Full fine-tuning updates the entire weight matrix $W_0 \\in \\mathbb{R}^{d \\times k}$. "
        "<b>LoRA (Low-Rank Adaptation)</b> freezes $W_0$ and decomposes updates into two low-rank matrices $B \\in \\mathbb{R}^{d \\times r}$ and $A \\in \\mathbb{R}^{r \\times k}$ ($r \\ll \\min(d, k)$):",
        body_style
    ))
    
    lora_diagram = (
        "               h = W_0 x + (alpha / r) * (B · A) x\n\n"
        "   Input x (dim d)\n"
        "      │\n"
        "      ├───► [ Frozen Base Weights W_0 (d x k) ] ───────────┐\n"
        "      │                                                     ▼\n"
        "      └───► [ Matrix A (d x r) ] ──► [ Matrix B (r x k) ] ──(+)──► Output h\n"
        "               (Gaussian Init)           (Zero Init)"
    )
    story.append(code_box(lora_diagram))
    story.append(Spacer(1, 4))
    
    story.append(callout_box(
        "<b>Parameter Reduction Example:</b> For $d=4096$, $r=8$:<br/>"
        "&bull; Full weight matrix: $4096 \\times 4096 = 16,777,216$ params.<br/>"
        "&bull; LoRA matrices: $(4096 \\times 8) + (8 \\times 4096) = 65,536$ params (<b>99.6% parameter reduction!</b>)."
    ))
    story.append(Spacer(1, 4))
    
    lora_code = (
        "from peft import LoraConfig, get_peft_model\n\n"
        "lora_config = LoraConfig(\n"
        "    r=8,                             # Low-rank dimension\n"
        "    lora_alpha=16,                   # Scaling multiplier (scale = alpha / r = 2.0)\n"
        "    lora_dropout=0.05,               # Regularization dropout for LoRA layers\n"
        "    bias='none',\n"
        "    task_type='CAUSAL_LM',\n"
        "    target_modules=['qkv_proj', 'o_proj'] # Targeted attention projection layers\n"
        ")\n"
        "model = get_peft_model(base_model, lora_config)\n"
        "model.print_trainable_parameters()\n"
        "# Trainable: ~3.0M / 3.82B (0.079% of model weights)"
    )
    story.append(code_box(lora_code))
    story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 7: QUANTIZATION & QLORA WITH SFTTRAINER
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("7. Quantization & QLoRA with SFTTrainer", h1_style))
    story.append(Paragraph(
        "Quantization maps 32-bit floating point weights into lower precision discrete bins ($X_{\\text{quant}} = \\text{round}(X / \\text{scale}) + \\text{zero\\_point}$):",
        body_style
    ))
    
    # Precision Table
    prec_data = [
        [Paragraph("<b>Precision</b>", tbl_header_style), Paragraph("<b>Bits / Weight</b>", tbl_header_style), Paragraph("<b>7B Model VRAM</b>", tbl_header_style), Paragraph("<b>Quality Retention</b>", tbl_header_style)],
        [Paragraph("<b>FP32</b> (Standard)", tbl_cell_style), Paragraph("32 bits", tbl_cell_style), Paragraph("~28.0 GB", tbl_cell_style), Paragraph("100% (Baseline)", tbl_cell_style)],
        [Paragraph("<b>FP16 / BF16</b>", tbl_cell_style), Paragraph("16 bits", tbl_cell_style), Paragraph("~14.0 GB", tbl_cell_style), Paragraph("99.9%", tbl_cell_style)],
        [Paragraph("<b>INT8 (8-bit)</b>", tbl_cell_style), Paragraph("8 bits", tbl_cell_style), Paragraph("~7.0 GB", tbl_cell_style), Paragraph("99.5%", tbl_cell_style)],
        [Paragraph("<b>NF4 (4-bit QLoRA)</b>", tbl_cell_style), Paragraph("4 bits", tbl_cell_style), Paragraph("~3.8 GB", tbl_cell_style), Paragraph("98.5%", tbl_cell_style)],
    ]
    prec_table = Table(prec_data, colWidths=[120, 100, 120, 164])
    prec_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_ACCENT),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(prec_table)
    story.append(Spacer(1, 4))

    qlora_code = (
        "from transformers import BitsAndBytesConfig, AutoModelForCausalLM\n"
        "from trl import SFTConfig, SFTTrainer\n\n"
        "# 1. Load base model in 8-bit quantization\n"
        "bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)\n"
        "model = AutoModelForCausalLM.from_pretrained('gpt2-xl', quantization_config=bnb_config, device_map='auto')\n\n"
        "# 2. Configure Supervised Fine-Tuning with SFTTrainer\n"
        "sft_config = SFTConfig(\n"
        "    output_dir='./qlora_output',\n"
        "    dataset_text_field='formatted_text',\n"
        "    max_seq_length=512,\n"
        "    packing=True,                   # Efficiently packs short sequences into 512 chunks\n"
        "    num_train_epochs=3,\n"
        "    per_device_train_batch_size=4,\n"
        "    gradient_accumulation_steps=4,\n"
        "    learning_rate=2e-4,\n"
        "    logging_steps=10\n"
        ")\n"
        "trainer = SFTTrainer(model=model, args=sft_config, train_dataset=dataset, peft_config=lora_config, processing_class=tokenizer)\n"
        "trainer.train()"
    )
    story.append(code_box(qlora_code))
    story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 8: FULL DEPLOYMENT LIFECYCLE (MERGE, GGUF, OLLAMA)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("8. Full Deployment Lifecycle: Merging, GGUF & Ollama", h1_style))
    story.append(Paragraph(
        "Deploying fine-tuned models locally involves fusing LoRA adapters, binary GGUF quantization, and Ollama containerization:",
        body_style
    ))
    
    deploy_flow = (
        "┌───────────────────────────┐     ┌───────────────────────────┐     ┌──────────────────────────┐\n"
        "│ 1. Merge LoRA with Base   │ ──► │ 2. Convert to GGUF Binary │ ──► │ 3. Local Ollama Runtime │\n"
        "│    model.merge_and_unload │     │    llama.cpp conversion   │     │    Modelfile + CLI / API │\n"
        "└───────────────────────────┘     └───────────────────────────┘     └──────────────────────────┘"
    )
    story.append(code_box(deploy_flow))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("<b>Step-by-Step Production Commands:</b>", h2_style))
    deploy_code = (
        "# Step 1: Merge adapters into standalone model\n"
        "merged_model = model.merge_and_unload()\n"
        "merged_model.save_pretrained('./phi-3.5-merged', safe_serialization=True)\n"
        "tokenizer.save_pretrained('./phi-3.5-merged')\n\n"
        "# Step 2: Convert to GGUF using llama.cpp\n"
        "# python3 llama.cpp/convert_hf_to_gguf.py ./phi-3.5-merged --outfile phi-3.5-f16.gguf --outtype f16\n\n"
        "# Step 3: Modelfile definition\n"
        "# FROM ./phi-3.5-f16.gguf\n"
        "# TEMPLATE \"\"\"{{ if .System }}<|system|>\\n{{ .System }}<|end|>\\n{{ end }}\n"
        "# {{ if .Prompt }}<|user|>\\n{{ .Prompt }}<|end|>\\n{{ end }}<|assistant|>\\n\"\"\"\n"
        "# PARAMETER temperature 0.7\n"
        "# PARAMETER stop \"<|end|>\"\n\n"
        "# Step 4: Register and run in Ollama\n"
        "# ollama create my-phi3 -f Modelfile\n"
        "# ollama run my-phi3"
    )
    story.append(code_box(deploy_code))
    story.append(Spacer(1, 6))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 9: MASTER API CHEAT SHEET & GLOSSARY
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("9. Master API Cheat Sheet & Function Glossary", h1_style))
    story.append(Paragraph(
        "Quick reference table covering all critical functions, classes, and parameters encountered throughout the course:",
        body_style
    ))
    
    glossary_data = [
        [Paragraph("<b>Function / Class</b>", tbl_header_style), Paragraph("<b>Module</b>", tbl_header_style), Paragraph("<b>Exact Purpose &amp; Critical Parameters</b>", tbl_header_style)],
        [Paragraph("<code>AutoTokenizer.from_pretrained</code>", tbl_cell_code_style), Paragraph("transformers", tbl_cell_style), Paragraph("Loads tokenizer vocabulary &amp; configuration from model checkpoint.", tbl_cell_style)],
        [Paragraph("<code>apply_chat_template</code>", tbl_cell_code_style), Paragraph("transformers", tbl_cell_style), Paragraph("Formats role messages (system/user/assistant) into model's Jinja template.", tbl_cell_style)],
        [Paragraph("<code>resize_token_embeddings</code>", tbl_cell_code_style), Paragraph("transformers", tbl_cell_style), Paragraph("Expands embedding matrix when new special tokens are registered.", tbl_cell_style)],
        [Paragraph("<code>get_memory_footprint</code>", tbl_cell_code_style), Paragraph("transformers", tbl_cell_style), Paragraph("Returns active memory occupied by model parameters in bytes.", tbl_cell_style)],
        [Paragraph("<code>DataCollatorWithPadding</code>", tbl_cell_code_style), Paragraph("transformers", tbl_cell_style), Paragraph("Dynamically pads mini-batches to max batch sequence length.", tbl_cell_style)],
        [Paragraph("<code>DataCollatorForSeq2Seq</code>", tbl_cell_code_style), Paragraph("transformers", tbl_cell_style), Paragraph("Dynamically pads inputs and fills label pad positions with -100.", tbl_cell_style)],
        [Paragraph("<code>LoraConfig</code>", tbl_cell_code_style), Paragraph("peft", tbl_cell_style), Paragraph("Defines rank ($r$), alpha ($\\alpha$), dropout, and target layers (e.g. <code>qkv_proj</code>).", tbl_cell_style)],
        [Paragraph("<code>get_peft_model</code>", tbl_cell_code_style), Paragraph("peft", tbl_cell_style), Paragraph("Wraps PyTorch base model with trainable low-rank LoRA matrices.", tbl_cell_style)],
        [Paragraph("<code>merge_and_unload</code>", tbl_cell_code_style), Paragraph("peft", tbl_cell_style), Paragraph("Fuses LoRA adapter weights permanently into base model weights.", tbl_cell_style)],
        [Paragraph("<code>BitsAndBytesConfig</code>", tbl_cell_code_style), Paragraph("transformers", tbl_cell_style), Paragraph("Enables 8-bit (<code>load_in_8bit</code>) or 4-bit (<code>NF4</code>) quantized model loading.", tbl_cell_style)],
        [Paragraph("<code>SFTTrainer</code>", tbl_cell_code_style), Paragraph("trl", tbl_cell_style), Paragraph("High-level Supervised Fine-Tuning trainer supporting sample packing &amp; PEFT.", tbl_cell_style)],
        [Paragraph("<code>convert_hf_to_gguf.py</code>", tbl_cell_code_style), Paragraph("llama.cpp", tbl_cell_style), Paragraph("Converts Hugging Face checkpoints to GGUF binary format for llama.cpp.", tbl_cell_style)],
        [Paragraph("<code>ollama create / run</code>", tbl_cell_code_style), Paragraph("CLI", tbl_cell_style), Paragraph("Builds and executes local LLM containers with hardware acceleration.", tbl_cell_style)],
    ]
    
    glossary_table = Table(glossary_data, colWidths=[150, 74, 280])
    glossary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_PRIMARY),
        ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(glossary_table)
    
    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    build_pdf()
