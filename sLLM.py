#
###############################################################################################
# ___  _    _    __  __       _____  ___   ___   ___  _  _  ___  ___ #
#/ __|| |  | |  |  \/  |     |_   _|| _ \ / _ \ |_ _|| \| || __|| _ \#
#\__ \| |_ | |_ | |\/| |       | |  |   // /_\ \ | | | .` || _| |   /#
#|___/|___||___||_|  |_|       |_|  |_|_\\_/ \_/|___||_|\_||___||_|_\#

#
###############################################################################################
# ___  __   __           ___  _  _  _   _ __   __ ___   ___        ___    ___   ___  _   _ #
#| _ ) \ \ / /          / __|| || || | | |\ \ / /| _ \ / _ \      | _ )  / _ \ / __|| | | |#
#| _ \  \_| /           \__ \| __ || |_| | \ V / |   /| (_) |     | _ \ / /_\ \\__ \| |_| |#
#|___/   |_|            |___/|_||_| \___/   \_/  |_|_\ \___/      |___/ \_/ \_/|___/ \___/ #
###########################################################################################
###########################################################################################


#####################################################
# Clean Documentation by PY_CLEAN_DOCUMENT v 1.0
# (C) Shuvro Basu, 2026
#####################################################

# !/usr/bin/env python3
"""
Python Code LLM Trainer v4.0 - Professional Navy Edition
- Modern navy blue theme with side panel navigation
- Comprehensive progress tracking for all operations
- Tooltips for all settings
- Proper state management
- Colored icons throughout
- Thread-safe GUI with no freezing
- Mauve E0B0FF
- Beige F5F5DC
- Maroon 800000
Files	d_model	layers	heads	d_ff	batch	epochs	~Params	~VRAM
150k	512    	6	    8	    2048	128	    10	    140M	~10GB
300k	768	    12	    12	    3072	128	    4	    200M	~13GB
450k	1024	12	    16	    4096	96	    3	    350M	~15GB
    110M Model on 16GB
    d_model = 768
    n_heads = 12
    n_layers = 12
    d_ff = 3072
    context = 512
    stride = 256
    batch = 48
    precision = bf16
    epochs = 4

Training Strategy:
    Do NOT retrain tokenizer.
    Load your last checkpoint.
    Add new files to the folder.
    Increase Epochs to 3.
    Start Training.
    The system will scan the folder (seeing old + new files), encode everything using the saved tokenizer
    (make sure python_llm_tokenizer.json is in the checkpoint folder), and continue training where it left off, learning
    from both old (review) and new data.
"""

import os
import re
import json
import time
import math
import threading
import queue
import configparser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, fields
from typing import List, Dict, Optional, Callable, Any, Tuple
from collections import Counter
import warnings
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import nullcontext
import multiprocessing
import sys

warnings.filterwarnings('ignore')

# ============================================================================
# DEPENDENCIES CHECK
# ============================================================================
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.amp import autocast, GradScaler

    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
    CUDA_AVAILABLE = torch.cuda.is_available()
    CUDA_VERSION = torch.version.cuda if CUDA_AVAILABLE else "N/A"
    GPU_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "None"
    print(f"[✓] PyTorch {TORCH_VERSION} | CUDA {CUDA_VERSION} | GPU: {GPU_NAME}")
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_VERSION = "Not Found"
    CUDA_AVAILABLE = False
    CUDA_VERSION = "N/A"
    GPU_NAME = "None"
    print("[✗] PyTorch not found - Training disabled")

try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    FAST_TOKENIZER = True
    print("[✓] Fast tokenizer (HuggingFace)")
except ImportError:
    FAST_TOKENIZER = False
    print("[!] Fast tokenizer unavailable - Using simple tokenizer")

# Helper for multiprocessing - MUST be at top level
def _global_process_encode(args):
    chunk_data, path = args
    from tokenizers import Tokenizer
    import numpy as np

    # Load tokenizer in worker process
    t = Tokenizer.from_file(path)

    # Batch encode
    encs = t.encode_batch(chunk_data)

    # Flatten to numpy
    flat_ids = [tid for e in encs for tid in e.ids]
    return np.array(flat_ids, dtype=np.int32)


# ============================================================================
# DATA REFINEMENT HELPERS (ported from data_refiner.py)
# ============================================================================

import hashlib
import unicodedata
from collections import Counter

# Boilerplate patterns for text cleaning
_BOILERPLATE_PATTERNS = [
    r"^\s*copyright\b", r"^\s*all rights reserved\b", r"^\s*isbn\b",
    r"^\s*published by\b", r"^\s*project \b",
    r"^\s*\*\*\*\s*start of", r"^\s*\*\*\*\s*end of",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE)

# Character replacements
_COMMON_REPLACEMENTS = {"\u00a0": " ", "\u200b": "", "\ufeff": ""}
_FICTION_REPLACEMENTS = {""": '"', """: '"', "'": "'", "'": "'", "—": "-", "–": "-"}


def _normalize_common(text: str) -> str:
    """Normalize unicode and line endings."""
    text = unicodedata.normalize("NFKC", text)
    for k, v in _COMMON_REPLACEMENTS.items():
        text = text.replace(k, v)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def clean_text_fiction(text: str) -> str:
    """Clean fiction/book text: remove boilerplate, normalize quotes."""
    text = _normalize_common(text)
    for k, v in _FICTION_REPLACEMENTS.items():
        text = text.replace(k, v)
    lines = text.split("\n")
    out = []
    for line in lines:
        raw = line.strip()
        if not raw:
            out.append("")
            continue
        if _BOILERPLATE_RE.search(raw):
            continue
        out.append(raw)
    text = "\n".join(out)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def clean_code(text: str) -> str:
    """Clean code: preserve indentation, minimal changes."""
    text = _normalize_common(text)
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{8,}", "\n\n\n\n\n\n\n", text)
    return text.strip("\n")


def clean_docs(text: str) -> str:
    """Clean markdown/docs: preserve formatting, remove boilerplate."""
    text = _normalize_common(text)
    lines = text.split("\n")
    out = []
    for line in lines:
        line = line.rstrip()
        raw = line.strip()
        if raw and _BOILERPLATE_RE.search(raw):
            continue
        out.append(line)
    text = "\n".join(out)
    text = re.sub(r"\n{5,}", "\n\n\n\n", text)
    return text.strip()


def clean_by_mode(text: str, mode: str, ext: str) -> str:
    """Clean text based on mode."""
    if mode == "fiction":
        return clean_text_fiction(text)
    if mode == "code":
        return clean_code(text)
    if mode == "docs":
        return clean_docs(text)
    if mode == "auto":
        code_exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".c", ".h", ".cpp", ".hpp", ".java", ".go", ".rs"}
        doc_exts = {".md", ".rst"}
        if ext.lower() in code_exts:
            return clean_code(text)
        if ext.lower() in doc_exts:
            return clean_docs(text)
        return clean_text_fiction(text)
    return text  # mode == "none"


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_quality_metrics(text: str, is_code: bool) -> dict:
    """Compute quality metrics for text or code."""
    n_chars = len(text)
    n_lines = text.count("\n") + 1 if text else 0
    if n_chars == 0:
        return {"chars": 0, "lines": 0, "words": 0, "score": 0}
    
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\t")
    printable_ratio = _safe_div(printable, n_chars)
    
    # Entropy
    char_counts = Counter(text)
    total = sum(char_counts.values())
    entropy = 0.0
    for c in char_counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    
    # Repeated lines
    lines = text.split("\n")
    line_counts = Counter(ln.strip().lower() for ln in lines if ln.strip())
    max_repeat = max(line_counts.values()) if line_counts else 0
    max_line_repeat_frac = _safe_div(max_repeat, len(lines))
    
    if is_code:
        # Code-specific metrics
        tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", text)
        unique_ratio = _safe_div(len(set(tokens)), len(tokens)) if tokens else 0
        punct = len(re.findall(r"[^\w\s]", text))
        punct_ratio = _safe_div(punct, n_chars)
        long_lines = sum(1 for ln in lines if len(ln) > 180)
        long_line_frac = _safe_div(long_lines, n_lines)
        
        return {
            "chars": n_chars, "lines": n_lines, "words": len(tokens),
            "printable_ratio": printable_ratio, "entropy": entropy,
            "max_line_repeat_frac": max_line_repeat_frac,
            "unique_ratio": unique_ratio, "punct_ratio": punct_ratio,
            "long_line_frac": long_line_frac, "is_code": True
        }
    else:
        # Text-specific metrics
        words = re.findall(r"[A-Za-z0-9']+", text)
        n_words = len(words)
        unique_word_ratio = _safe_div(len(set(w.lower() for w in words)), n_words)
        letters = sum(ch.isalpha() for ch in text)
        letter_ratio = _safe_div(letters, n_chars)
        single_char = sum(1 for w in words if len(w) == 1)
        single_char_ratio = _safe_div(single_char, n_words)
        
        return {
            "chars": n_chars, "lines": n_lines, "words": n_words,
            "printable_ratio": printable_ratio, "entropy": entropy,
            "max_line_repeat_frac": max_line_repeat_frac,
            "unique_ratio": unique_word_ratio, "letter_ratio": letter_ratio,
            "single_char_ratio": single_char_ratio, "is_code": False
        }


def compute_quality_score(metrics: dict, min_chars: int = 500) -> tuple:
    """Compute quality score 0-100 and list of reasons for deductions."""
    score = 100
    reasons = []
    
    if metrics["chars"] < min_chars:
        score -= 35
        reasons.append("too_short")
    
    if metrics.get("printable_ratio", 1.0) < 0.95:
        score -= 20
        reasons.append("low_printable")
    
    if metrics.get("entropy", 10) < 3.5:
        score -= 10
        reasons.append("low_entropy")
    
    if metrics.get("max_line_repeat_frac", 0) > 0.05:
        score -= 10
        reasons.append("repeated_lines")
    
    if metrics.get("is_code"):
        if metrics.get("punct_ratio", 0) < 0.03:
            score -= 10
            reasons.append("low_punct")
        if metrics.get("long_line_frac", 0) > 0.25:
            score -= 12
            reasons.append("many_long_lines")
    else:
        if metrics.get("letter_ratio", 1) < 0.55:
            score -= 15
            reasons.append("low_letters")
        if metrics.get("unique_ratio", 1) < 0.12:
            score -= 12
            reasons.append("low_diversity")
        if metrics.get("single_char_ratio", 0) > 0.06:
            score -= 10
            reasons.append("many_single_chars")
    
    return max(0, min(100, score)), reasons


def sha256_hash(text: str) -> str:
    """SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8-sig")).hexdigest()


def md5_hash(text: str) -> str:
    """MD5 hash of text."""
    return hashlib.md5(text.encode("utf-8-sig")).hexdigest()


def build_fingerprint(text: str, chunk_size: int, use_lines: bool) -> set:
    """Build fingerprint for near-duplicate detection."""
    if use_lines:
        parts = text.split("\n")
    else:
        parts = text.split()
    
    if len(parts) < chunk_size * 2:
        return {md5_hash(text)}
    
    hashes = set()
    sep = "\n" if use_lines else " "
    for i in range(0, len(parts) - chunk_size + 1, chunk_size):
        chunk = sep.join(parts[i:i + chunk_size])
        hashes.add(md5_hash(chunk))
    return hashes


def jaccard_similarity(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def clean_by_mode(text: str, mode: str, ext: str = "") -> str:
    """Apply text cleaning based on mode.
    
    Modes:
    - auto: Detect based on file extension
    - code: Clean code files (remove excessive comments, normalize whitespace)
    - fiction: Clean prose (remove chapter headers, normalize quotes)
    - docs: Clean documentation (remove markdown cruft, normalize headers)
    - none: Return text unchanged
    """
    if mode == "none":
        return text
    
    # Auto-detect based on extension
    if mode == "auto":
        code_exts = {".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".rb", ".php"}
        if ext.lower() in code_exts:
            mode = "code"
        elif ext.lower() in {".md", ".rst"}:
            mode = "docs"
        else:
            mode = "fiction"
    
    if mode == "code":
        # Code cleaning: remove excessive blank lines, normalize indentation
        lines = text.split("\n")
        cleaned = []
        blank_count = 0
        for line in lines:
            if not line.strip():
                blank_count += 1
                if blank_count <= 2:  # Allow max 2 consecutive blank lines
                    cleaned.append("")
            else:
                blank_count = 0
                cleaned.append(line.rstrip())  # Remove trailing whitespace
        return "\n".join(cleaned)
    
    elif mode == "fiction":
        # Fiction cleaning: normalize quotes, remove chapter markers
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('—', '-').replace('–', '-')
        # Remove common boilerplate patterns
        text = re.sub(r'^Chapter\s+\d+\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^\*\s*\*\s*\*\s*$', '', text, flags=re.MULTILINE)
        # Normalize multiple newlines
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        return text.strip()
    
    elif mode == "docs":
        # Documentation cleaning: remove markdown formatting cruft
        # Remove excessive heading markers
        text = re.sub(r'^#{4,}\s*', '### ', text, flags=re.MULTILINE)
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        # Normalize code block markers
        text = re.sub(r'```\w*\n', '```\n', text)
        # Remove trailing whitespace
        lines = [line.rstrip() for line in text.split("\n")]
        return "\n".join(lines)
    
    return text


# ============================================================================
# ICONS (Unicode with color hints)
# ============================================================================

# -----------------------------------------#
# Class Name : Icons
# Calls: None
# -----------------------------------------#
class Icons:
    """Colored unicode icons for the application."""
    # Navigation
    DATA = "📂"
    MODEL = "🧠"
    TRAINING = "🎯"
    CHECKPOINT = "💾"
    HELP = "❓"
    TEST = "🧪"

    # Actions
    PLAY = "▶️"
    PAUSE = "⏸️"
    STOP = "⏹️"
    SAVE = "💾"
    LOAD = "📥"
    BROWSE = "📁"
    SCAN = "🔍"
    CLEAR = "🗑️"
    COPY = "📋"
    PLOT = "📊"
    CHART = "📈"
    REFRESH = "🔄"
    SETTINGS = "⚙️"
    WAIT = "⏳"
    TARGET = "🎯"

    # Status
    SUCCESS = "✅"
    ERROR = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    LOADING = "⏳"
    ROCKET = "🚀"
    SPARKLE = "✨"
    FIRE = "🔥"
    BOLT = "⚡"
    ADD = "➕"

    # Indicators
    CHECK = "✓"
    CROSS = "✗"
    ARROW_RIGHT = "→"
    ARROW_DOWN = "↓"
    BULLET = "•"
    STAR = "★"

    # Hardware
    GPU = "🎮"
    CPU = "💻"
    MEMORY = "🧮"
    FOLDER = "📁"
    FILE = "📄"
    CODE = "💻"


########################## END OF CLASS Icons ################################


# ============================================================================
# PROFESSIONAL NAVY THEME
# ============================================================================

# -----------------------------------------#
# Class Name : NavyTheme
# Calls: apply, root.configure, root.option_add, style.configure, style.map, style.theme_use, ttk.Style
# -----------------------------------------#
class NavyTheme:
    """Professional navy blue theme configuration."""

    # Primary colors
    NAVY_DARKEST = "#0b1320"
    NAVY_DARK = "#162030"
    NAVY_MEDIUM = "#253448"
    NAVY_LIGHT = "#324a5f"
    NAVY_LIGHTER = "#415a77"

    # Accent colors
    ACCENT_BLUE = "#3b82f6"
    ACCENT_RED = "#ef4444"
    ACCENT_CYAN = "#48cae4"
    ACCENT_LIGHT = "#90e0ef"

    # Semantic colors
    SUCCESS = "#4ade80"
    SUCCESS_DIM = "#22c55e"
    WARNING = "#fbbf24"
    WARNING_DIM = "#f59e0b"
    ERROR = "#f87171"
    ERROR_DIM = "#ef4444"
    INFO = "#60a5fa"

    # Text colors
    TEXT_PRIMARY = "#e2e8f0"
    TEXT_SECONDARY = "#94a3b8"
    TEXT_DIM = "#64748b"
    TEXT_ACCENT = "#7dd3fc"

    # Borders
    BORDER_LIGHT = "#3d5a80"
    BORDER_DARK = "#1e3a5f"

    # Backgrounds
    BG_INPUT = "#1e3048"
    BG_HOVER = "#2d4a6a"
    BG_SELECTED = "#3d5a80"
    BG_CARD = "#162032"

    # Fonts
    FONT_FAMILY = "Segoe UI"
    FONT_MONO = "Cascadia Code"
    FONT_FALLBACK_MONO = "Consolas"

    # Sizes
    FONT_SIZE_SMALL = 9
    FONT_SIZE_NORMAL = 10
    FONT_SIZE_LARGE = 11
    FONT_SIZE_HEADER = 14
    FONT_SIZE_TITLE = 18

    # Sidebar
    SIDEBAR_WIDTH = 250
    SIDEBAR_BG = "#0d1b2a"
    SIDEBAR_HOVER = "#1b263b"
    SIDEBAR_SELECTED = "#253448"

    @classmethod
    # ---------------------------------------------------#
    # Method name: apply
    # ---------------------------------------------------#
    def apply(cls, root: tk.Tk):
        """Apply the navy theme to the application."""
        style = ttk.Style()

        try:
            style.theme_use('clam')
        except tk.TclError:
            style.theme_use('default')

        # Base configuration
        style.configure(".",
                        background=cls.NAVY_DARK,
                        foreground=cls.TEXT_PRIMARY,
                        font=(cls.FONT_FAMILY, cls.FONT_SIZE_NORMAL),
                        borderwidth=0,
                        focuscolor=cls.ACCENT_BLUE
                        )

        # Frames
        style.configure("TFrame", background=cls.NAVY_DARK)
        style.configure("Card.TFrame", background=cls.BG_CARD)
        style.configure("Sidebar.TFrame", background=cls.SIDEBAR_BG)
        style.configure("Content.TFrame", background=cls.NAVY_DARK)

        # Labels
        style.configure("TLabel",
                        background=cls.NAVY_DARK,
                        foreground=cls.TEXT_PRIMARY
                        )
        style.configure("Title.TLabel",
                        font=(cls.FONT_FAMILY, cls.FONT_SIZE_TITLE, "bold"),
                        foreground=cls.TEXT_PRIMARY
                        )
        style.configure("Header.TLabel",
                        font=(cls.FONT_FAMILY, cls.FONT_SIZE_HEADER, "bold"),
                        foreground=cls.TEXT_PRIMARY
                        )
        style.configure("SubHeader.TLabel",
                        font=(cls.FONT_FAMILY, cls.FONT_SIZE_LARGE, "bold"),
                        foreground=cls.TEXT_SECONDARY
                        )
        style.configure("Dim.TLabel", foreground=cls.TEXT_DIM)
        style.configure("Accent.TLabel", foreground=cls.ACCENT_CYAN)
        style.configure("Success.TLabel", foreground=cls.SUCCESS)
        style.configure("Warning.TLabel", foreground=cls.WARNING)
        style.configure("Error.TLabel", foreground=cls.ERROR)
        style.configure("Info.TLabel", foreground=cls.INFO)
        style.configure("Card.TLabel", background=cls.BG_CARD)
        style.configure("Sidebar.TLabel",
                        background=cls.SIDEBAR_BG,
                        foreground=cls.TEXT_SECONDARY
                        )

        # Buttons
        style.configure("TButton",
                        background=cls.NAVY_LIGHT,
                        foreground=cls.TEXT_PRIMARY,
                        padding=(12, 8),
                        font=(cls.FONT_FAMILY, cls.FONT_SIZE_NORMAL)
                        )
        style.map("TButton",
                  background=[
                      ("active", cls.BG_HOVER),
                      ("disabled", cls.NAVY_MEDIUM)
                  ],
                  foreground=[("disabled", cls.TEXT_DIM)]
                  )

        style.configure("Accent.TButton",
                        background=cls.ACCENT_BLUE,
                        foreground="white",
                        padding=(16, 10),
                        font=(cls.FONT_FAMILY, cls.FONT_SIZE_NORMAL, "bold")
                        )
        style.map("Accent.TButton",
                  background=[
                      ("active", cls.ACCENT_CYAN),
                      ("disabled", cls.NAVY_LIGHT)
                  ]
                  )

        style.configure("Success.TButton",
                        background=cls.SUCCESS_DIM,
                        foreground="white",
                        padding=(12, 8)
                        )
        style.map("Success.TButton",
                  background=[("active", cls.SUCCESS)]
                  )

        style.configure("Danger.TButton",
                        background=cls.ERROR_DIM,
                        foreground="white",
                        padding=(12, 8)
                        )
        style.map("Danger.TButton",
                  background=[("active", cls.ERROR)]
                  )

        style.configure("Sidebar.TButton",
                        background=cls.SIDEBAR_BG,
                        foreground=cls.TEXT_SECONDARY,
                        padding=(15, 12),
                        font=(cls.FONT_FAMILY, cls.FONT_SIZE_NORMAL),
                        anchor="w"
                        )
        style.map("Sidebar.TButton",
                  background=[
                      ("active", cls.SIDEBAR_HOVER),
                      ("selected", cls.SIDEBAR_SELECTED)
                  ],
                  foreground=[("active", cls.TEXT_PRIMARY)]
                  )

        # Entry
        style.configure("TEntry",
                        fieldbackground=cls.BG_INPUT,
                        foreground=cls.TEXT_PRIMARY,
                        insertcolor=cls.TEXT_PRIMARY,
                        padding=10
                        )
        style.map("TEntry",
                  fieldbackground=[("focus", cls.NAVY_MEDIUM)]
                  )

        # Combobox
        style.configure("TCombobox",
                        fieldbackground=cls.BG_INPUT,
                        background=cls.NAVY_LIGHT,
                        foreground=cls.TEXT_PRIMARY,
                        padding=8
                        )
        style.map("TCombobox",
                  fieldbackground=[("focus", cls.NAVY_MEDIUM)]
                  )

        # LabelFrame
        style.configure("TLabelframe",
                        background=cls.BG_CARD,
                        bordercolor=cls.BORDER_LIGHT,
                        relief="solid"
                        )
        style.configure("TLabelframe.Label",
                        background=cls.BG_CARD,
                        foreground=cls.ACCENT_CYAN,
                        font=(cls.FONT_FAMILY, cls.FONT_SIZE_NORMAL, "bold"),
                        padding=(5, 2)
                        )

        # Checkbutton & Radiobutton
        style.configure("TCheckbutton",
                        background=cls.BG_CARD,
                        foreground=cls.TEXT_PRIMARY,
                        font=(cls.FONT_FAMILY, cls.FONT_SIZE_NORMAL)
                        )
        style.map("TCheckbutton",
                  background=[("active", cls.BG_CARD)],
                  foreground=[("active", cls.ACCENT_CYAN)]
                  )

        style.configure("TRadiobutton",
                        background=cls.BG_CARD,
                        foreground=cls.TEXT_PRIMARY,
                        font=(cls.FONT_FAMILY, cls.FONT_SIZE_NORMAL)
                        )
        style.map("TRadiobutton",
                  background=[("active", cls.BG_CARD)],
                  foreground=[("active", cls.ACCENT_CYAN)]
                  )

        # Progressbar

        style.configure("TProgressbar",
                        background=cls.ACCENT_BLUE,
                        troughcolor=cls.NAVY_MEDIUM,
                        borderwidth=0,
                        thickness=8
                        )
        style.configure("Success.Horizontal.TProgressbar",
                        background=cls.SUCCESS
                        )
        style.configure("Warning.Horizontal.TProgressbar",
                        background=cls.WARNING
                        )

        # Notebook (Tabs)
        style.configure("TNotebook",
                        background=cls.NAVY_DARK,
                        borderwidth=0
                        )
        style.configure("TNotebook.Tab",
                        background=cls.NAVY_MEDIUM,
                        foreground=cls.TEXT_SECONDARY,
                        padding=(16, 10),
                        font=(cls.FONT_FAMILY, cls.FONT_SIZE_NORMAL)
                        )
        style.map("TNotebook.Tab",
                  background=[("selected", cls.BG_CARD)],
                  foreground=[("selected", cls.TEXT_PRIMARY)]
                  )

        # Radiobutton
        style.configure("TRadiobutton",
                        background=cls.NAVY_DARK,
                        foreground=cls.TEXT_PRIMARY
                        )
        style.configure("Card.TRadiobutton",
                        background=cls.BG_CARD
                        )

        # Checkbutton
        style.configure("TCheckbutton",
                        background=cls.NAVY_DARK,
                        foreground=cls.TEXT_PRIMARY
                        )
        style.configure("Card.TCheckbutton",
                        background=cls.BG_CARD
                        )

        # Scrollbar
        style.configure("TScrollbar",
                        background=cls.NAVY_MEDIUM,
                        troughcolor=cls.NAVY_DARK,
                        borderwidth=0,
                        arrowsize=14
                        )
        style.map("TScrollbar",
                  background=[("active", cls.NAVY_LIGHT)]
                  )

        # Separator
        style.configure("TSeparator",
                        background=cls.BORDER_LIGHT
                        )

        # Scale
        style.configure("TScale",
                        background=cls.NAVY_DARK,
                        troughcolor=cls.NAVY_MEDIUM
                        )

        # Spinbox
        style.configure("TSpinbox",
                        fieldbackground=cls.BG_INPUT,
                        background=cls.NAVY_LIGHT,
                        foreground=cls.TEXT_PRIMARY
                        )

        # Treeview colors
        style.configure("Treeview",
                        background=cls.BG_INPUT,
                        foreground=cls.TEXT_PRIMARY,
                        fieldbackground=cls.BG_INPUT,
                        borderwidth=0)

        style.map("Treeview",
                  background=[("selected", cls.ACCENT_BLUE)],
                  foreground=[("selected", "white")])

        style.configure("Treeview.Heading",
                        background=cls.NAVY_MEDIUM,
                        foreground=cls.TEXT_PRIMARY,
                        font=(cls.FONT_FAMILY, 9, "bold"))

        # Configure root window
        root.configure(bg=cls.NAVY_DARKEST)
        root.option_add("*TCombobox*Listbox.background", cls.BG_INPUT)
        root.option_add("*TCombobox*Listbox.foreground", cls.TEXT_PRIMARY)
        root.option_add("*TCombobox*Listbox.selectBackground", cls.ACCENT_BLUE)


    @classmethod
    def style_card(cls, frame, elevation=1):
        """Apply card styling - NEW but doesn't break old code"""
        frame.configure(style="Card.TFrame")
        frame.padding = (20, 15)


    @classmethod
    def apply_treeview_styling(cls, tree):
        """Professional treeview styling - NEW"""
        tree.tag_configure("duplicate", background="#3d1f1f", foreground="#ff9999")
        tree.tag_configure("low_quality", background="#3d3a1f", foreground="#ffff99")


########################## END OF CLASS NavyTheme ################################


# ============================================================================
# TOOLTIP CLASS
# ============================================================================

# -----------------------------------------#
# Class Name : ToolTip
# Calls: __init__, _schedule_show, _cancel_schedule, _show, _hide, create, frame.pack, inner_frame.pack, label.pack, tk.Frame, tk.Label, tk.Toplevel, tw.wm_attributes, tw.wm_geometry, tw.wm_overrideredirect, widget.bind
# -----------------------------------------#
class ToolTip:
    """Professional tooltip with navy theme styling."""

    DELAY = 100  # ms before showing
    WRAP_LENGTH = 300

    # ---------------------------------------------------#
    # Method name: __init__
    # ---------------------------------------------------#
    def __init__(self, widget: tk.Widget, text: str, delay: int = None):
        self.widget = widget
        self.text = text
        self.delay = delay or self.DELAY
        self.tooltip_window = None
        self.schedule_id = None

        widget.bind("<Enter>", self._schedule_show)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    # ---------------------------------------------------#
    # Method name: _schedule_show
    # ---------------------------------------------------#
    def _schedule_show(self, event=None):
        self._cancel_schedule()
        self.schedule_id = self.widget.after(self.delay, self._show)

    # ---------------------------------------------------#
    # Method name: _cancel_schedule
    # ---------------------------------------------------#
    def _cancel_schedule(self):
        if self.schedule_id:
            self.widget.after_cancel(self.schedule_id)
            self.schedule_id = None

    # ---------------------------------------------------#
    # Method name: _show
    # ---------------------------------------------------#
    def _show(self, event=None):
        if self.tooltip_window:
            return

        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.wm_attributes("-topmost", True)

        # Create tooltip frame with border
        frame = tk.Frame(
            tw,
            background=NavyTheme.ACCENT_BLUE,
            padx=1,
            pady=1
        )
        frame.pack()

        inner_frame = tk.Frame(
            frame,
            background=NavyTheme.NAVY_MEDIUM,
            padx=10,
            pady=6
        )
        inner_frame.pack()

        label = tk.Label(
            inner_frame,
            text=self.text,
            justify="left",
            background=NavyTheme.NAVY_MEDIUM,
            foreground=NavyTheme.TEXT_PRIMARY,
            font=(NavyTheme.FONT_FAMILY, NavyTheme.FONT_SIZE_SMALL),
            wraplength=self.WRAP_LENGTH
        )
        label.pack()

    # ---------------------------------------------------#
    # Method name: _hide
    # ---------------------------------------------------#
    def _hide(self, event=None):
        self._cancel_schedule()
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

    @staticmethod
    # ---------------------------------------------------#
    # Method name: create
    # ---------------------------------------------------#
    def create(widget: tk.Widget, text: str) -> 'ToolTip':
        """Factory method for creating tooltips."""
        return ToolTip(widget, text)


########################## END OF CLASS ToolTip ################################


# ============================================================================
# TOAST NOTIFICATION SYSTEM
# ============================================================================

class ToastManager:
    """Non-blocking notification system replacing intrusive messageboxes."""
    
    def __init__(self, parent):
        self.parent = parent
        self.active_toasts = []
        self.max_toasts = 4
        
    def show(self, message: str, level: str = "info", duration: int = 3000):
        """
        Show toast notification.
        
        Args:
            message: Text to display
            level: "info", "success", "warning", or "error"
            duration: Time in ms before auto-dismiss (0 = permanent)
        """
        colors = {
            "info": (NavyTheme.ACCENT_BLUE, "#e8f4fc", "ℹ"),
            "success": ("#28a745", "#d4edda", "✓"),
            "warning": ("#ffc107", "#fff3cd", "⚠"),
            "error": ("#dc3545", "#f8d7da", "✕")
        }
        bg, border_color, icon = colors.get(level, colors["info"])
        
        # Limit active toasts
        if len(self.active_toasts) >= self.max_toasts:
            self._dismiss(self.active_toasts[0])
        
        # Create toast window
        toast = tk.Toplevel(self.parent)
        toast.overrideredirect(True)
        toast.attributes("-topmost", True)
        toast.attributes("-alpha", 0.0)  # Start invisible for fade-in
        
        # Content frame with border
        outer = tk.Frame(toast, bg=bg, padx=2, pady=2)
        outer.pack(fill="both", expand=True)
        
        inner = tk.Frame(outer, bg=NavyTheme.NAVY_MEDIUM, padx=15, pady=12)
        inner.pack(fill="both", expand=True)
        
        # Icon and message
        content = tk.Frame(inner, bg=NavyTheme.NAVY_MEDIUM)
        content.pack(fill="x")
        
        tk.Label(
            content, 
            text=icon, 
            bg=NavyTheme.NAVY_MEDIUM, 
            fg=bg,
            font=(NavyTheme.FONT_FAMILY, 14)
        ).pack(side="left", padx=(0, 10))
        
        tk.Label(
            content,
            text=message,
            bg=NavyTheme.NAVY_MEDIUM,
            fg=NavyTheme.TEXT_PRIMARY,
            font=(NavyTheme.FONT_FAMILY, 10),
            wraplength=280,
            justify="left"
        ).pack(side="left", fill="x", expand=True)
        
        # Close button for longer toasts
        if duration == 0 or duration > 5000:
            close_btn = tk.Label(
                content, 
                text="×", 
                bg=NavyTheme.NAVY_MEDIUM,
                fg=NavyTheme.TEXT_DIM,
                font=(NavyTheme.FONT_FAMILY, 14),
                cursor="hand2"
            )
            close_btn.pack(side="right", padx=(10, 0))
            close_btn.bind("<Button-1>", lambda e: self._dismiss(toast))
        
        # Position (bottom-right stack)
        self._position_toast(toast)
        
        # Fade in
        self._fade_in(toast)
        
        # Auto-dismiss
        if duration > 0:
            toast.after(duration, lambda: self._dismiss(toast))
        
        # Click to dismiss
        toast.bind("<Button-1>", lambda e: self._dismiss(toast))
        
        self.active_toasts.append(toast)
        
    def _position_toast(self, toast):
        """Calculate position based on existing toasts."""
        self.parent.update_idletasks()
        toast.update_idletasks()
        
        toast_height = 60
        margin = 20
        stack_offset = len(self.active_toasts) * (toast_height + 10)
        
        x = self.parent.winfo_x() + self.parent.winfo_width() - 340 - margin
        y = self.parent.winfo_y() + self.parent.winfo_height() - toast_height - margin - stack_offset
        
        toast.geometry(f"320x{toast_height}+{x}+{y}")
        
    def _fade_in(self, toast, alpha=0.0):
        """Fade in animation."""
        if alpha < 0.95:
            alpha += 0.15
            try:
                toast.attributes("-alpha", alpha)
                toast.after(20, lambda: self._fade_in(toast, alpha))
            except tk.TclError:
                pass  # Toast was destroyed
        else:
            try:
                toast.attributes("-alpha", 1.0)
            except tk.TclError:
                pass
        
    def _dismiss(self, toast):
        """Remove toast with fade effect."""
        if toast not in self.active_toasts:
            return
            
        self.active_toasts.remove(toast)
        
        def fade_out(alpha=1.0):
            if alpha > 0.1:
                alpha -= 0.15
                try:
                    toast.attributes("-alpha", alpha)
                    toast.after(20, lambda: fade_out(alpha))
                except tk.TclError:
                    pass
            else:
                try:
                    toast.destroy()
                except tk.TclError:
                    pass
                # Reposition remaining toasts
                self._reposition_all()
                    
        fade_out()
        
    def _reposition_all(self):
        """Reposition all active toasts after one is dismissed."""
        for i, toast in enumerate(self.active_toasts):
            try:
                toast_height = 60
                margin = 20
                stack_offset = i * (toast_height + 10)
                
                x = self.parent.winfo_x() + self.parent.winfo_width() - 340 - margin
                y = self.parent.winfo_y() + self.parent.winfo_height() - toast_height - margin - stack_offset
                
                toast.geometry(f"320x{toast_height}+{x}+{y}")
            except tk.TclError:
                pass
                
    def info(self, message: str, duration: int = 3000):
        """Shortcut for info toast."""
        self.show(message, "info", duration)
        
    def success(self, message: str, duration: int = 3000):
        """Shortcut for success toast."""
        self.show(message, "success", duration)
        
    def warning(self, message: str, duration: int = 4000):
        """Shortcut for warning toast."""
        self.show(message, "warning", duration)
        
    def error(self, message: str, duration: int = 5000):
        """Shortcut for error toast."""
        self.show(message, "error", duration)


########################## END OF CLASS ToastManager ################################


# ============================================================================
# STATUS BAR
# ============================================================================

class StatusBar(ttk.Frame):
    """Professional status bar with multiple information zones."""
    
    def __init__(self, parent, app_state=None):
        super().__init__(parent, style="StatusBar.TFrame")
        self.app_state = app_state
        self._build_ui()
        
    def _build_ui(self):
        """Build status bar layout."""
        # Configure style if not exists
        style = ttk.Style()
        style.configure("StatusBar.TFrame", background=NavyTheme.NAVY_DARKEST)
        style.configure("StatusBar.TLabel", 
                       background=NavyTheme.NAVY_DARKEST,
                       foreground=NavyTheme.TEXT_DIM,
                       font=(NavyTheme.FONT_FAMILY, 9))
        
        # Left: Status icon and text
        left_frame = ttk.Frame(self, style="StatusBar.TFrame")
        left_frame.pack(side="left", padx=10, pady=5)
        
        self.status_icon = ttk.Label(left_frame, text="⦿", 
                                    style="StatusBar.TLabel",
                                    font=(NavyTheme.FONT_FAMILY, 10))
        self.status_icon.pack(side="left", padx=(0, 5))
        
        self.status_text = ttk.Label(left_frame, text="Ready", 
                                    style="StatusBar.TLabel")
        self.status_text.pack(side="left")
        
        # Separator
        ttk.Separator(self, orient="vertical").pack(side="left", fill="y", padx=10, pady=3)
        
        # Center: Current step/activity
        self.step_label = ttk.Label(self, text="", style="StatusBar.TLabel")
        self.step_label.pack(side="left", padx=10)
        
        # Right side container
        right_frame = ttk.Frame(self, style="StatusBar.TFrame")
        right_frame.pack(side="right", padx=10, pady=5)
        
        # VRAM
        self.vram_label = ttk.Label(right_frame, text="VRAM: --", 
                                   style="StatusBar.TLabel")
        self.vram_label.pack(side="right", padx=(10, 0))
        
        # GPU temperature (if available)
        self.gpu_temp_label = ttk.Label(right_frame, text="", 
                                       style="StatusBar.TLabel")
        self.gpu_temp_label.pack(side="right", padx=(10, 0))
        
        # Time/clock
        self.time_label = ttk.Label(right_frame, text="", 
                                   style="StatusBar.TLabel")
        self.time_label.pack(side="right")
        self._update_time()
        
    def _update_time(self):
        """Update time display."""
        try:
            now = datetime.now().strftime("%H:%M")
            self.time_label.config(text=now)
            self.after(30000, self._update_time)  # Update every 30s
        except Exception:
            pass
        
    def set_state(self, state: str, message: str = ""):
        """Update status with appropriate icon and color."""
        state_map = {
            "idle": ("⦿", NavyTheme.SUCCESS, "Ready"),
            "training": ("◉", NavyTheme.WARNING, "Training"),
            "paused": ("⏸", NavyTheme.ACCENT_CYAN, "Paused"),
            "error": ("⚠", NavyTheme.ERROR, "Error"),
            "scanning": ("⟳", NavyTheme.ACCENT_BLUE, "Scanning"),
            "processing": ("⟳", NavyTheme.ACCENT_BLUE, "Processing"),
            "saving": ("💾", NavyTheme.ACCENT_CYAN, "Saving"),
        }
        
        icon, color, default_msg = state_map.get(state.lower(), ("◌", NavyTheme.TEXT_DIM, state))
        
        self.status_icon.config(text=icon, foreground=color)
        self.status_text.config(text=message or default_msg)
        
    def set_step(self, current: int, total: int, name: str):
        """Show current pipeline step."""
        self.step_label.config(text=f"Step {current}/{total}: {name}")
        
    def set_vram(self, used_gb: float, total_gb: float):
        """Update VRAM display."""
        percent = (used_gb / total_gb * 100) if total_gb > 0 else 0
        color = NavyTheme.SUCCESS if percent < 70 else NavyTheme.WARNING if percent < 90 else NavyTheme.ERROR
        self.vram_label.config(text=f"VRAM: {used_gb:.1f}/{total_gb:.0f}GB", foreground=color)
        
    def clear_step(self):
        """Clear step indicator."""
        self.step_label.config(text="")


########################## END OF CLASS StatusBar ################################


# ============================================================================
# COLLAPSIBLE SECTION
# ============================================================================

class CollapsibleSection(ttk.LabelFrame):
    """Expandable/collapsible settings section."""
    
    def __init__(self, parent, title: str, expanded: bool = True, **kwargs):
        super().__init__(parent, text=f"  {title}", **kwargs)
        
        self.expanded = expanded
        self.title = title
        
        # Content frame (what users pack their widgets into)
        self.content = ttk.Frame(self)
        
        # Toggle button styled as label
        self.toggle_btn = tk.Label(
            self,
            text="−" if expanded else "+",
            bg=NavyTheme.NAVY_DARK,
            fg=NavyTheme.ACCENT_CYAN,
            font=(NavyTheme.FONT_FAMILY, 12, "bold"),
            cursor="hand2",
            padx=5
        )
        self.toggle_btn.place(relx=1.0, x=-10, y=0, anchor="ne")
        self.toggle_btn.bind("<Button-1>", lambda e: self.toggle())
        
        if expanded:
            self.content.pack(fill="both", expand=True, padx=10, pady=(5, 10))
            
    def toggle(self):
        """Toggle expanded/collapsed state."""
        self.expanded = not self.expanded
        if self.expanded:
            self.content.pack(fill="both", expand=True, padx=10, pady=(5, 10))
            self.toggle_btn.config(text="−")
        else:
            self.content.pack_forget()
            self.toggle_btn.config(text="+")
            
    def expand(self):
        """Force expand."""
        if not self.expanded:
            self.toggle()
            
    def collapse(self):
        """Force collapse."""
        if self.expanded:
            self.toggle()


########################## END OF CLASS CollapsibleSection ################################


# ============================================================================
# COLORED PROGRESS BAR
# ============================================================================

class ColoredProgressBar(tk.Canvas):
    def __init__(self, parent, width=300, height=20, bg="#162030", **kwargs):
        super().__init__(parent, width=width, height=height, bg=bg, highlightthickness=0, **kwargs)
        self.width_val = width
        self.height_val = height
        self.max_val = 100
        # Spectrum colors
        self.colors = ["#ef4444", "#f97316", "#facc15", "#84cc16", "#22c55e", "#06b6d4", "#3b82f6", "#6366f1", "#8b5cf6", "#d946ef"] 

    def set(self, value):
        self.delete("all")
        if value <= 0: return
        
        # Get actual width
        w = self.winfo_width()
        if w <= 1: w = self.width_val
        
        # Calculate fill width
        pct = min(max(value, 0), self.max_val) / self.max_val
        fill_w = w * pct
        
        # Draw rainbow segments
        seg_w = 20 # 20px segments
        count = int(fill_w / seg_w) + 1
        
        for i in range(count):
            c = self.colors[i % len(self.colors)]
            x1 = i * seg_w
            x2 = min((i+1)*seg_w, fill_w)
            if x2 > x1:
                self.create_rectangle(x1, 0, x2, self.height_val, fill=c, outline="")


class SidebarButton(tk.Canvas):
    def __init__(self, parent, text, icon, command, icon_color="#ffffff", **kwargs):
        super().__init__(parent, height=40, highlightthickness=0, **kwargs)
        self.text_str = text
        self.icon_str = icon
        self.icon_color = icon_color
        self.command = command
        self.is_selected = False
        self.is_disabled = False
        self.is_hovered = False
        
        # Colors
        self.bg_selected = "#22c55e" # Green
        self.bg_normal = "#bae6fd"   # Light Blue
        self.bg_disabled = "#ffffff"
        self.bg_hover = "#7dd3fc"
        
        self.fg_selected = "#ffffff"
        self.fg_normal = "#0f172a"
        self.fg_disabled = "#94a3b8"
        
        # Bind events
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Configure>", lambda e: self.redraw())
        
        self.redraw()

    def configure(self, **kwargs):
        if 'state' in kwargs:
            self.set_disabled(kwargs['state'] == 'disabled')
        super().configure(**kwargs)

    def redraw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w <= 1: w = 200
        
        # Determine BG/FG
        if self.is_disabled:
            bg = self.bg_disabled
            fg = self.fg_disabled
            icon_fill = self.fg_disabled
        elif self.is_selected:
            bg = self.bg_selected
            fg = self.fg_selected
            icon_fill = "#ffffff" # White icon when selected
        else:
            bg = self.bg_normal
            fg = self.fg_normal
            icon_fill = self.icon_color if self.icon_color else self.fg_normal
            
        super().configure(bg=bg)
        
        # Hover Border
        if self.is_hovered and not self.is_disabled:
            self.create_rectangle(2, 2, w-2, h-2, outline="black", dash=(2, 2), width=1)
            
        # Draw Content
        text_x = 20
        # Red Triangle if selected
        if self.is_selected:
            self.create_text(text_x, h//2, text="▶", fill="#ef4444", anchor="w", font=(NavyTheme.FONT_FAMILY, 12, "bold"))
            text_x += 20
        
        # Draw Icon separately for color
        self.create_text(text_x, h//2, text=self.icon_str, fill=icon_fill, anchor="w", font=(NavyTheme.FONT_FAMILY, 14))
        
        # Draw Text
        self.create_text(text_x + 30, h//2, text=self.text_str, fill=fg, anchor="w", font=(NavyTheme.FONT_FAMILY, 10))

    def set_selected(self, selected):
        self.is_selected = selected
        self.redraw()
        
    def set_disabled(self, disabled):
        self.is_disabled = disabled
        if disabled:
            self.unbind("<Button-1>")
            self.unbind("<Enter>")
            self.unbind("<Leave>")
        else:
            self.bind("<Button-1>", self._on_click)
            self.bind("<Enter>", self._on_enter)
            self.bind("<Leave>", self._on_leave)
        self.redraw()

    def _on_click(self, event):
        if not self.is_disabled and self.command:
            self.command()

    def _on_enter(self, event):
        if not self.is_disabled:
            self.is_hovered = True
            self.redraw()

    def _on_leave(self, event):
        if not self.is_disabled:
            self.is_hovered = False
            self.redraw()


# ============================================================================
# RICH TEXT RENDERER
# ============================================================================
class RichTextRenderer:
    """
    Advanced Markup Renderer for Tkinter Text Widgets.
    Supports:
    - [b]Bold[/b], [i]Italic[/i], [u]Underline[/u]
    - [color:NAME_OR_HEX]...[/color]
    - [icon:NAME] (Mapped to Icons class)
    - [link:SLUG]Text[/link] (Clickable links)
    - [img:path/to/image.png] (Inline images)
    - [h1], [h2], [mono] blocks
    """
    def __init__(self, text_widget, theme_class):
        self.text_widget = text_widget
        self.theme = theme_class
        self.images = [] # Prevent garbage collection
        self._configure_tags()

    def _configure_tags(self):
        """Setup base tags."""
        font_family = self.theme.FONT_FAMILY
        
        # Headers
        self.text_widget.tag_config("h1", font=(font_family, 14, "bold"), foreground=self.theme.ACCENT_CYAN, spacing3=10)
        self.text_widget.tag_config("h2", font=(font_family, 11, "bold"), foreground=self.theme.ACCENT_BLUE, spacing3=5)
        
        # Styles
        self.text_widget.tag_config("b", font=(font_family, 10, "bold"))
        self.text_widget.tag_config("i", font=(font_family, 10, "italic"))
        self.text_widget.tag_config("u", underline=True)
        self.text_widget.tag_config("mono", font=("Consolas", 10))
        
        # Colors (Predefined)
        self.text_widget.tag_config("dim", foreground=self.theme.TEXT_DIM)
        self.text_widget.tag_config("cyan", foreground=self.theme.ACCENT_CYAN)
        self.text_widget.tag_config("green", foreground=self.theme.SUCCESS)
        self.text_widget.tag_config("red", foreground=self.theme.ERROR)
        self.text_widget.tag_config("yellow", foreground=self.theme.WARNING)
        self.text_widget.tag_config("accent", foreground=self.theme.ACCENT_LIGHT)
        self.text_widget.tag_config("text", foreground=self.theme.TEXT_PRIMARY)

    def render(self, content):
        """Parse and render content string."""
        self.images.clear() # Clear old references
        self.text_widget.config(state="normal")
        self.text_widget.delete("1.0", "end")
        
        lines = content.split('\n')
        active_block = "text"
        
        for line in lines:
            line_str = line.rstrip() # Keep indentation if needed, strip newline
            
            # Block Detection (Simple State Machine)
            if line_str == "[mono]":
                active_block = "mono"
                continue
            if line_str == "[/mono]":
                active_block = "text"
                continue
            
            # Line-level formatting (Headers)
            current_base_tag = active_block
            if active_block == "text":
                if line_str.startswith("# "):
                    current_base_tag = "h1"
                    line_str = line_str[2:]
                    self._create_anchor(line_str)
                elif line_str.startswith("## "):
                    current_base_tag = "h2"
                    line_str = line_str[3:]
                    self._create_anchor(line_str)
            
            # Inline Parsing
            self._render_line(line_str, current_base_tag)
            self.text_widget.insert("end", "\n")
            
        self.text_widget.config(state="disabled")

    def _create_anchor(self, text):
        """Create a mark for linking to this section."""
        clean = re.sub(r'\[.*?\]', '', text).strip()
        slug = re.sub(r'[^a-z0-9]+', '_', clean.lower()).strip('_')
        if slug:
            self.text_widget.mark_set(slug, "insert")
            self.text_widget.mark_gravity(slug, "left")

    def _render_line(self, line, base_tag):
        """Process inline tags for a single line."""
        # 1. Escape Handling: Hide literal brackets
        # Replace \[ with \u0000 and \] with \u0001 temporarily
        line = line.replace("\\[", "\u0000").replace("\\]", "\u0001")
        
        # Split by tags: [tag] or [/tag] or [tag:val]
        parts = re.split(r'(\[/?[\w:#\./]+\])', line)
        
        active_tags = {base_tag} # Set of active tags
        
        for part in parts:
            if not part: continue
            
            # Restore escapes in content
            part_content = part.replace("\u0000", "[").replace("\u0001", "]")
            
            # Closing Tag
            if part.startswith("[/") and part.endswith("]"):
                tag_name = part[2:-1]  # [/b] -> b
                
                if tag_name == "link":
                    # Remove dynamic link tag
                    to_remove = [t for t in active_tags if t.startswith("link_")]
                    for t in to_remove: active_tags.discard(t)
                    
                elif tag_name == "color":
                    # Remove any active color tags (fg_...)
                    to_remove = [t for t in active_tags if t.startswith("fg_") or t in ["dim", "accent", "red", "green", "blue", "cyan", "yellow"]]
                    for t in to_remove: active_tags.discard(t)
                    
                elif tag_name in active_tags:
                    active_tags.discard(tag_name)
                    
            # Opening Tag
            elif part.startswith("[") and part.endswith("]"):
                content = part[1:-1] # b or color:red
                
                # Check if it looks like a tag (no spaces usually)
                if " " in content:
                    # Treat as literal text if it contains spaces (e.g. [some text])
                    self.text_widget.insert("end", part_content, list(active_tags))
                    continue
                    
                if ":" in content:
                    # Tag with value
                    type_, val = content.split(":", 1)
                    
                    if type_ == "icon":
                        icon_char = getattr(Icons, val, "?")
                        self.text_widget.insert("end", icon_char + " ", list(active_tags))
                        
                    elif type_ == "color":
                        # Logic: 
                        # 1. New color tag comes in.
                        # 2. Tkinter allows multiple tags. Last one usually wins priority if defined later,
                        #    but safe bet is to remove existing color tags to avoid conflict.
                        # Remove existing colors from active set first (nested color override)
                        existing_colors = [t for t in active_tags if t.startswith("fg_") or t in ["dim", "accent", "red", "green", "blue", "cyan", "yellow"]]
                        for t in existing_colors: active_tags.discard(t)
                        
                        color_tag = f"fg_{val}"
                        # Check if hex or named
                        if val.startswith("#"):
                            self.text_widget.tag_config(color_tag, foreground=val)
                            self.text_widget.tag_raise(color_tag) # Ensure overrides base text
                            active_tags.add(color_tag)
                        else:
                            # Fallback to predefined text tag if exists
                            if val in ["red", "green", "blue", "cyan", "yellow", "dim", "accent"]:
                                active_tags.add(val)
                                self.text_widget.tag_raise(val) # Ensure overrides base text
                            else:
                                # Treat as named color supported by Tkinter
                                self.text_widget.tag_config(color_tag, foreground=val)
                                self.text_widget.tag_raise(color_tag) # Ensure overrides base text
                                active_tags.add(color_tag)
                        
                    elif type_ == "link":
                        link_tag = f"link_{val}"
                        self.text_widget.tag_config(link_tag, foreground=self.theme.ACCENT_CYAN, underline=True)
                        self._bind_link(link_tag, val)
                        active_tags.add(link_tag)
                        
                    elif type_ == "img":
                        self._insert_image(val, active_tags)
                    
                    else:
                         # Unknown tag type, treat as text
                         self.text_widget.insert("end", part_content, list(active_tags))
                        
                else:
                    # Simple Tag (b, i, u)
                    active_tags.add(content)
                    
            # Text Content
            else:
                self.text_widget.insert("end", part_content, list(active_tags))

    def _bind_link(self, tag, target):
        """Bind click event for link."""
        def _scroll(e):
            try: self.text_widget.see(target)
            except: pass
        
        self.text_widget.tag_bind(tag, "<Button-1>", _scroll)
        self.text_widget.tag_bind(tag, "<Enter>", lambda e: self.text_widget.config(cursor="hand2"))
        self.text_widget.tag_bind(tag, "<Leave>", lambda e: self.text_widget.config(cursor=""))

    def _insert_image(self, rel_path, tags):
        """Load and insert image."""
        if os.path.exists(rel_path):
            try:
                img = tk.PhotoImage(file=rel_path)
                # Naive scale down if huge
                if img.width() > 500:
                    scale = img.width() // 500
                    img = img.subsample(scale, scale)
                
                self.images.append(img) # Keep ref
                self.text_widget.image_create("end", image=img)
                self.text_widget.insert("end", " ", list(tags)) # Spacer
            except Exception:
                self.text_widget.insert("end", f"[IMG:{rel_path}]", list(tags))




class ValidatedEntry(ttk.Frame):
    """Entry field with inline validation and error display."""
    
    def __init__(self, parent, label: str = "", validator=None, 
                 required: bool = False, var=None, width: int = 15, **kwargs):
        super().__init__(parent)
        
        self.validator = validator  # Function: (value) -> (is_valid: bool, error_msg: str)
        self.required = required
        self.valid = True
        self._error_msg = ""
        
        # Label (optional)
        if label:
            self.label = ttk.Label(self, text=label)
            self.label.pack(side="left", padx=(0, 5))
        
        # Variable
        self.var = var if var else tk.StringVar()
        
        # Entry with indicator frame
        entry_frame = ttk.Frame(self)
        entry_frame.pack(side="left", fill="x", expand=True)
        
        self.entry = ttk.Entry(entry_frame, textvariable=self.var, width=width)
        self.entry.pack(side="left", fill="x", expand=True)
        
        # Validation indicator (small colored bar)
        self.indicator = tk.Canvas(entry_frame, width=3, height=20, 
                                  bg=NavyTheme.NAVY_DARK, 
                                  highlightthickness=0)
        self.indicator.pack(side="right", padx=(2, 0))
        
        # Error tooltip
        self.error_tooltip = None
        
        # Bind validation
        self.var.trace_add("write", self._on_change)
        self.entry.bind("<FocusOut>", self._on_focus_out)
        
    def _on_change(self, *args):
        """Validate on every change."""
        self._validate()
        
    def _on_focus_out(self, event=None):
        """Validate when leaving field."""
        self._validate()
        
    def _validate(self) -> bool:
        """Run validation and update UI."""
        value = self.var.get()
        
        # Required check
        if self.required and not value.strip():
            self._show_invalid("Required field")
            return False
            
        # Custom validator
        if self.validator and value:
            try:
                result = self.validator(value)
                if isinstance(result, tuple):
                    is_valid, msg = result
                else:
                    is_valid, msg = result, ""
                    
                if not is_valid:
                    self._show_invalid(msg or "Invalid value")
                    return False
            except Exception as e:
                self._show_invalid(str(e))
                return False
                
        self._show_valid()
        return True
        
    def _show_valid(self):
        """Show valid state."""
        self.valid = True
        self._error_msg = ""
        self.indicator.config(bg=NavyTheme.NAVY_DARK)
        self.entry.config(style="TEntry")
        self.event_generate("<<Valid>>")
        
    def _show_invalid(self, msg: str):
        """Show invalid state with error indicator."""
        self.valid = False
        self._error_msg = msg
        self.indicator.config(bg=NavyTheme.ERROR)
        # Could add red border style here
        self.event_generate("<<Invalid>>")
        
    def get(self) -> str:
        """Get current value."""
        return self.var.get()
        
    def set(self, value: str):
        """Set value."""
        self.var.set(value)
        
    def is_valid(self) -> bool:
        """Return current validation state."""
        return self._validate()
        
    def get_error(self) -> str:
        """Get current error message."""
        return self._error_msg


########################## END OF CLASS ValidatedEntry ################################


# ============================================================================
# SHORTCUT MANAGER
# ============================================================================

class ShortcutManager:
    """Manage keyboard shortcuts with visual hints."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.shortcuts: Dict[str, Tuple[Callable, str]] = {}
        
    def register(self, keystroke: str, callback: Callable, description: str):
        """
        Register a keyboard shortcut.
        
        Args:
            keystroke: e.g., "Ctrl+S", "Ctrl+Shift+P", "F1", "Escape"
            callback: Function to call
            description: Human-readable description
        """
        self.shortcuts[keystroke] = (callback, description)
        
        # Convert to Tk binding format
        tk_binding = self._convert_keystroke(keystroke)
        self.root.bind(tk_binding, lambda e, cb=callback: self._execute(cb))
        
    def _convert_keystroke(self, keystroke: str) -> str:
        """Convert human-readable keystroke to Tk format."""
        # "Ctrl+S" -> "<Control-s>"
        # "Ctrl+Shift+P" -> "<Control-Shift-p>"
        # "F1" -> "<F1>"
        
        parts = keystroke.split("+")
        tk_parts = []
        
        for part in parts:
            part = part.strip()
            if part.lower() == "ctrl":
                tk_parts.append("Control")
            elif part.lower() == "alt":
                tk_parts.append("Alt")
            elif part.lower() == "shift":
                tk_parts.append("Shift")
            elif part.lower() == "escape":
                tk_parts.append("Escape")
            elif part.lower() == "return" or part.lower() == "enter":
                tk_parts.append("Return")
            elif part.startswith("F") and part[1:].isdigit():
                tk_parts.append(part)  # F1, F2, etc.
            else:
                tk_parts.append(part.lower())
                
        return "<" + "-".join(tk_parts) + ">"
        
    def _execute(self, callback: Callable):
        """Execute callback and prevent event propagation."""
        try:
            callback()
        except Exception as e:
            print(f"[Shortcut Error] {e}")
        return "break"
        
    def show_shortcuts_dialog(self):
        """Display searchable shortcuts reference dialog."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Keyboard Shortcuts")
        dlg.geometry("450x400")
        dlg.configure(bg=NavyTheme.NAVY_DARK)
        dlg.transient(self.root)
        dlg.grab_set()
        
        # Header
        tk.Label(
            dlg, 
            text="⌨ Keyboard Shortcuts",
            bg=NavyTheme.NAVY_DARK,
            fg=NavyTheme.TEXT_PRIMARY,
            font=(NavyTheme.FONT_FAMILY, 14, "bold")
        ).pack(pady=15)
        
        # Search
        search_frame = ttk.Frame(dlg)
        search_frame.pack(fill="x", padx=20, pady=(0, 10))
        
        search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=search_var)
        search_entry.pack(fill="x")
        search_entry.insert(0, "🔍 Search...")
        search_entry.bind("<FocusIn>", lambda e: search_entry.delete(0, "end") if search_entry.get().startswith("🔍") else None)
        
        # Shortcuts list
        list_frame = ttk.Frame(dlg)
        list_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Create list with columns
        columns = ("shortcut", "description")
        tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=12)
        tree.heading("shortcut", text="Shortcut")
        tree.heading("description", text="Description")
        tree.column("shortcut", width=120)
        tree.column("description", width=280)
        
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Populate
        for key, (_, desc) in sorted(self.shortcuts.items()):
            tree.insert("", "end", values=(key, desc))
            
        # Filter function
        def filter_shortcuts(*args):
            query = search_var.get().lower()
            if query.startswith("🔍"):
                query = ""
            
            for item in tree.get_children():
                tree.delete(item)
                
            for key, (_, desc) in sorted(self.shortcuts.items()):
                if query in key.lower() or query in desc.lower():
                    tree.insert("", "end", values=(key, desc))
                    
        search_var.trace_add("write", filter_shortcuts)
        
        # Close button
        ttk.Button(dlg, text="Close", command=dlg.destroy).pack(pady=15)
        
        # Center dialog
        dlg.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dlg.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{x}+{y}")
        
    def get_all_shortcuts(self) -> Dict[str, str]:
        """Get all registered shortcuts as dict."""
        return {k: v[1] for k, v in self.shortcuts.items()}


########################## END OF CLASS ShortcutManager ################################


# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================

@dataclass
# -----------------------------------------#
# Class Name : ModelConfig
# Type: Data Class
# Calls: validate, errors.append
# -----------------------------------------#

class ModelConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    vocab_size: int = 32000
    max_seq_len: int = 2048
    dropout: float = 0.1
    bias: bool = True
    gradient_checkpointing: bool = False  # <--- NEW

    # ---------------------------------------------------#
    # Method name: validate
    # ---------------------------------------------------#
    def validate(self) -> List[str]:
        errors = []
        if self.d_model % self.n_heads != 0:
            errors.append(f"Embedding dim ({self.d_model}) must be divisible by heads ({self.n_heads})")
        if self.d_model < 64:
            errors.append(f"Embedding dim should be at least 64")
        if self.n_layers < 1:
            errors.append(f"Layers must be at least 1")
        if self.vocab_size < 100:
            errors.append(f"Vocab size should be at least 100")
        if not 0 <= self.dropout < 1:
            errors.append(f"Dropout must be between 0 and 1")
        if self.max_seq_len < 32:
            errors.append(f"Max sequence length should be at least 32")
        return errors


########################## END OF CLASS ModelConfig ################################


@dataclass
# -----------------------------------------#
# Class Name : TrainingConfig
# Type: Data Class
# Calls: validate, errors.append
# -----------------------------------------#
class TrainingConfig:
    learning_rate: float = 3e-4
    batch_size: int = 64
    gradient_accumulation: int = 4  # ADD THIS
    epochs: int = 10
    context_length: int = 512
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    val_split: float = 0.1
    precision: str = "bf16"
    early_stopping: bool = True
    early_stopping_patience: int = 3
    log_interval: int = 5
    stride: int = 256
    validation_percent: float = 20.0  # Validation limit percentage

    # ---------------------------------------------------#
    # Method name: validate
    # ---------------------------------------------------#
    def validate(self) -> List[str]:
        errors = []
        if self.learning_rate <= 0 or self.learning_rate > 1:
            errors.append(f"Learning rate should be between 0 and 1")
        if self.batch_size < 1:
            errors.append(f"Batch size must be at least 1")
        if self.epochs < 1:
            errors.append(f"Epochs must be at least 1")
        if self.stride > self.context_length:
            errors.append(f"Stride should not exceed context length")
        if not 0 < self.val_split < 1:
            errors.append(f"Validation split must be between 0 and 1")
        if self.gradient_clip <= 0:
            errors.append(f"Gradient clip must be positive")
        return errors


########################## END OF CLASS TrainingConfig ################################


@dataclass
# -----------------------------------------#
# Class Name : CheckpointConfig
# Type: Data Class
# Calls: None
# -----------------------------------------#
class CheckpointConfig:
    output_dir: str = "./checkpoints"
    checkpoint_name: str = "python_llm"
    save_every_epochs: int = 1
    save_every_steps: int = 100
    save_optimizer: bool = True
    save_tokenizer: bool = True
    resume_from: str = ""


########################## END OF CLASS CheckpointConfig ################################


@dataclass
# -----------------------------------------#
# Class Name : NormalizationConfig
# Type: Data Class
# Calls: None
# -----------------------------------------#
class NormalizationConfig:
    include_comments: bool = True
    include_docstrings: bool = True
    normalize_whitespace: bool = True
    tabs_to_spaces: int = 4


########################## END OF CLASS NormalizationConfig ################################


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

# -----------------------------------------#
# Class Name : AppState
# Calls: __init__, state, state, progress, set_progress, sub_progress, set_sub_progress, add_listener, _notify, is_busy, can_start_training, reset, threading.Lock
# -----------------------------------------#
class AppState:
    """Centralized state management for the application."""

    # Application states
    IDLE = "idle"
    SCANNING = "scanning"
    PROCESSING = "processing"
    TOKENIZING = "tokenizing"
    ENCODING = "encoding"
    TRAINING = "training"
    PAUSED = "paused"
    SAVING = "saving"
    LOADING = "loading"
    TESTING = "testing"
    ERROR = "error"

    # ---------------------------------------------------#
    # Method name: __init__
    # ---------------------------------------------------#
    def __init__(self):
        self._state = self.IDLE
        self._lock = threading.Lock()
        self._listeners: List[Callable[[str, str], None]] = []
        self._progress = 0.0
        self._progress_message = ""
        self._sub_progress = 0.0
        self._sub_message = ""

        self._dirty_pages = set()  # NEW: track unsaved changes
        self._transition_history = []  # NEW: for back navigation

    def mark_dirty(self, page_id, is_dirty=True):
        """Track unsaved changes - NEW feature"""
        if is_dirty:
            self._dirty_pages.add(page_id)
        else:
            self._dirty_pages.discard(page_id)

    def is_dirty(self):
        """Check if any page has unsaved changes - NEW"""
        return len(self._dirty_pages) > 0

    def can_transition(self, new_state):
        """Guard state transitions - NEW validation"""
        # Add logic here, return True by default
        return True
    
    @property
    # ---------------------------------------------------#
    # Method name: state
    # ---------------------------------------------------#
    def state(self) -> str:
        with self._lock:
            return self._state

    @state.setter
    # ---------------------------------------------------#
    # Method name: state
    # ---------------------------------------------------#
    def state(self, value: str):
        with self._lock:
            old_state = self._state
            self._state = value
        self._notify(old_state, value)

    @property
    # ---------------------------------------------------#
    # Method name: progress
    # ---------------------------------------------------#
    def progress(self) -> Tuple[float, str]:
        with self._lock:
            return self._progress, self._progress_message

    # ---------------------------------------------------#
    # Method name: set_progress
    # ---------------------------------------------------#
    def set_progress(self, value: float, message: str = ""):
        with self._lock:
            self._progress = max(0.0, min(1.0, value))
            self._progress_message = message

    @property
    # ---------------------------------------------------#
    # Method name: sub_progress
    # ---------------------------------------------------#
    def sub_progress(self) -> Tuple[float, str]:
        with self._lock:
            return self._sub_progress, self._sub_message

    # ---------------------------------------------------#
    # Method name: set_sub_progress
    # ---------------------------------------------------#
    def set_sub_progress(self, value: float, message: str = ""):
        with self._lock:
            self._sub_progress = max(0.0, min(1.0, value))
            self._sub_message = message

    # ---------------------------------------------------#
    # Method name: add_listener
    # ---------------------------------------------------#
    def add_listener(self, callback: Callable[[str, str], None]):
        self._listeners.append(callback)

    # ---------------------------------------------------#
    # Method name: _notify
    # ---------------------------------------------------#
    def _notify(self, old_state: str, new_state: str):
        for listener in self._listeners:
            try:
                listener(old_state, new_state)
            except Exception:
                pass

    # ---------------------------------------------------#
    # Method name: is_busy
    # ---------------------------------------------------#
    def is_busy(self) -> bool:
        return self.state not in (self.IDLE, self.ERROR, self.PAUSED)

    # ---------------------------------------------------#
    # Method name: can_start_training
    # ---------------------------------------------------#
    def can_start_training(self) -> bool:
        return self.state in (self.IDLE, self.ERROR)

    # ---------------------------------------------------#
    # Method name: reset
    # ---------------------------------------------------#
    def reset(self):
        self.state = self.IDLE
        self.set_progress(0.0, "")
        self.set_sub_progress(0.0, "")


########################## END OF CLASS AppState ################################


# ============================================================================
# THREAD-SAFE GUI UPDATE QUEUE
# ============================================================================

# -----------------------------------------#
# Class Name : GUIUpdateQueue
# Calls: __init__, put, _process_updates, stop, queue.Queue
# -----------------------------------------#
class GUIUpdateQueue:
    """Thread-safe queue for GUI updates."""

    # ---------------------------------------------------#
    # Method name: __init__
    # ---------------------------------------------------#
    def __init__(self, root: tk.Tk):
        self.root = root
        self.queue: queue.Queue = queue.Queue()
        self._running = True
        self._process_updates()

    # ---------------------------------------------------#
    # Method name: put
    # ---------------------------------------------------#
    def put(self, func: Callable, *args, **kwargs):
        """Queue a function to be executed in the main thread."""
        self.queue.put((func, args, kwargs))

    # ---------------------------------------------------#
    # Method name: _process_updates
    # ---------------------------------------------------#
    def _process_updates(self):
        """Process queued updates in the main thread."""
        if not self._running:
            return

        try:
            while True:
                func, args, kwargs = self.queue.get_nowait()
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    print(f"[GUI Update Error] {e}")
        except queue.Empty:
            pass

        if self._running:
            self.root.after(16, self._process_updates)  # ~60fps

    # ---------------------------------------------------#
    # Method name: stop
    # ---------------------------------------------------#
    def stop(self):
        self._running = False


########################## END OF CLASS GUIUpdateQueue ################################


####################################
# ETA Calc Class
# ==================================
# -----------------------------------------#
# Class Name : ETACalculator
# Calls: __init__, reset, get_eta, format_progress, time.time
# -----------------------------------------#
class ETACalculator:
    """Calculate estimated time remaining."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.history = []  # List of (time, progress) tuples
        self.last_eta = None

    def get_eta(self, progress: float) -> str:
        """Get formatted ETA string."""
        if progress <= 0.01:
            return "Calculating..."

        now = time.time()

        # Add to history
        self.history.append((now, progress))

        # Keep only last 20 data points
        if len(self.history) > 20:
            self.history = self.history[-20:]

        # Need at least 2 points
        if len(self.history) < 2:
            return "Calculating..."

        # Calculate rate from recent history (last 10 points or all if fewer)
        recent = self.history[-10:]
        oldest_time, oldest_progress = recent[0]
        newest_time, newest_progress = recent[-1]

        delta_time = newest_time - oldest_time
        delta_progress = newest_progress - oldest_progress

        if delta_time < 0.5 or delta_progress <= 0:
            # Not enough change, use last known ETA
            if self.last_eta:
                return self.last_eta
            return "Calculating..."

        # Calculate rate and remaining time
        rate = delta_progress / delta_time
        remaining_progress = 1.0 - progress
        remaining_seconds = remaining_progress / rate

        # Sanity bounds
        remaining_seconds = max(0, min(remaining_seconds, 86400))

        # Format
        if remaining_seconds < 60:
            eta_str = f"{int(remaining_seconds)}s"
        elif remaining_seconds < 3600:
            m = int(remaining_seconds // 60)
            s = int(remaining_seconds % 60)
            eta_str = f"{m}m {s}s"
        else:
            h = int(remaining_seconds // 3600)
            m = int((remaining_seconds % 3600) // 60)
            eta_str = f"{h}h {m}m"

        self.last_eta = eta_str
        return eta_str

########################## END OF CLASS ETACalculator ################################


# ============================================================================
# TOKENIZER
# ============================================================================

# Replace BytePairTokenizer class with this optimized version:

# -----------------------------------------#
# Class Name : BytePairTokenizer
# Calls: __init__, train, encode, encode_batch, encode_all_parallel, decode, save, load, get_vocab_size, Tokenizer.from_file, all_arrays.append, all_tokens.extend, decoders.ByteLevel, json.dump, json.load, math.exp, models.BPE, np.array, np.concatenate, pre_tokenizers.ByteLevel, re.findall, thread.start, threading.Event, threading.Lock, threading.Thread, time.time, tokens.append, trainers.BpeTrainer, training_done.is_set, training_done.set, training_done.wait, word_freq.most_common, word_freq.update
# -----------------------------------------#
class BytePairTokenizer:
    """Optimized BPE Tokenizer with parallel encoding."""

    # ---------------------------------------------------#
    # Method name: __init__
    # ---------------------------------------------------#
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.tokenizer: Optional[Any] = None
        self.use_fast = FAST_TOKENIZER
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self._lock = threading.Lock()  # Only for simple tokenizer

    # ---------------------------------------------------#
    # Method name: train
    # ---------------------------------------------------#
    def train(self, texts: List[str],
              progress_callback: Optional[Callable[[float, str], None]] = None,
              stop_check: Optional[Callable[[], bool]] = None):

        if self.use_fast:
            if progress_callback:
                progress_callback(0.0, "Initializing BPE tokenizer...")

            self.tokenizer = Tokenizer(models.BPE(unk_token='<|unk|>'))
            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = decoders.ByteLevel()

            trainer = trainers.BpeTrainer(
                vocab_size=self.vocab_size,
                special_tokens=['<|pad|>', '<|unk|>', '<|startoftext|>', '<|endoftext|>'],
                min_frequency=2,
                show_progress=False
            )

            if progress_callback:
                progress_callback(0.05, "Preparing training data...")

            import threading

            training_done = threading.Event()
            training_error = [None]

            def train_thread():
                try:
                    self.tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
                except Exception as e:
                    training_error[0] = e
                finally:
                    training_done.set()

            thread = threading.Thread(target=train_thread, daemon=True)
            thread.start()

            start_time = time.time()
            estimated_time = len(texts) / 5000
            estimated_time = max(5, min(estimated_time, 120))

            while not training_done.is_set():
                elapsed = time.time() - start_time
                progress = min(0.95, 0.05 + 0.90 * (1 - math.exp(-elapsed / (estimated_time * 0.5))))

                if progress_callback:
                    remaining = max(0, estimated_time - elapsed)
                    if remaining < 60:
                        eta_str = f"{int(remaining)}s"
                    else:
                        eta_str = f"{int(remaining // 60)}m {int(remaining % 60)}s"
                    progress_callback(progress, f"Training BPE model... • ETA: {eta_str}")

                training_done.wait(timeout=0.2)

            if training_error[0]:
                raise training_error[0]

            # Get actual vocab size
            actual_vocab = self.tokenizer.get_vocab_size()
            self.vocab_size = actual_vocab

            if progress_callback:
                progress_callback(1.0, f"Tokenizer ready: {actual_vocab} tokens")

        else:
            # Simple tokenizer
            if progress_callback:
                progress_callback(0.0, "Building vocabulary...")

            self.vocab = {
                '<|pad|>': 0, '<|unk|>': 1,
                '<|startoftext|>': 2, '<|endoftext|>': 3
            }
            word_freq: Counter = Counter()

            total = len(texts)
            last_update = time.time()

            for i, text in enumerate(texts):
                word_freq.update(re.findall(r'\w+|[^\w\s]|\s+', text))

                now = time.time()
                if progress_callback and (now - last_update >= 0.1 or i == total - 1):
                    progress_callback((i + 1) / total * 0.8, f"Counting tokens: {i + 1:,}/{total:,}")
                    last_update = now

            if progress_callback:
                progress_callback(0.85, "Building vocabulary...")

            for token, _ in word_freq.most_common(self.vocab_size - 4):
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

            self.inverse_vocab = {v: k for k, v in self.vocab.items()}

            if progress_callback:
                progress_callback(1.0, f"Tokenizer ready: {len(self.vocab)} tokens")

    # ---------------------------------------------------#
    # Method name: encode
    # ---------------------------------------------------#
    def encode(self, text: str) -> List[int]:
        """Encode single text - no lock for fast tokenizer."""
        if self.use_fast and self.tokenizer:
            return self.tokenizer.encode(text).ids

        with self._lock:
            tokens = [self.bos_token_id]
            for t in re.findall(r'\w+|[^\w\s]|\s+', text):
                tokens.append(self.vocab.get(t, self.unk_token_id))
            tokens.append(self.eos_token_id)
            return tokens

    # ---------------------------------------------------#
    # Method name: encode_batch
    # ---------------------------------------------------#
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode multiple texts in parallel - MUCH faster."""
        if self.use_fast and self.tokenizer:
            # HuggingFace batch encoding - uses all CPU cores automatically
            encodings = self.tokenizer.encode_batch(texts)
            return [enc.ids for enc in encodings]

        # Fallback for simple tokenizer
        return [self.encode(text) for text in texts]

    # ---------------------------------------------------#
    # Method name: encode_all_parallel
    # ---------------------------------------------------#
    def encode_all_parallel(self, texts: List[str],
                            progress_callback: Optional[Callable[[float, str], None]] = None,
                            stop_check: Optional[Callable[[], bool]] = None) -> List[int]:
        """Encode all texts with true parallelism. Returns flat token list."""

        if not self.use_fast or not self.tokenizer:
            # Fallback
            all_tokens = []
            for i, text in enumerate(texts):
                if stop_check and stop_check():
                    return all_tokens
                all_tokens.extend(self.encode(text))
                if progress_callback and i % 1000 == 0:
                    progress_callback(i / len(texts), f"Encoding: {i:,}/{len(texts):,}")
            return all_tokens

        import numpy as np

        total = len(texts)

        # Enable parallelism in tokenizer
        if hasattr(self.tokenizer, 'enable_parallelism'):
            self.tokenizer.enable_parallelism(True)

        # Process in chunks to allow progress updates
        chunk_size = 20000  # Large chunks = less overhead
        all_arrays = []
        total_tokens = 0

        for start in range(0, total, chunk_size):
            if stop_check and stop_check():
                return []

            end = min(start + chunk_size, total)
            chunk = texts[start:end]

            # encode_batch uses Rust parallelism internally
            encoded = self.tokenizer.encode_batch(chunk)

            # Convert to numpy
            for enc in encoded:
                arr = np.array(enc.ids, dtype=np.int32)
                all_arrays.append(arr)
                total_tokens += len(arr)

            if progress_callback:
                progress_callback(end / total, f"Encoding: {end:,}/{total:,}")

        if stop_check and stop_check():
            return []

        # Single concatenation
        if progress_callback:
            progress_callback(0.95, f"Finalizing {total_tokens:,} tokens...")

        result = np.concatenate(all_arrays).tolist()

        if progress_callback:
            progress_callback(1.0, f"Encoded {total_tokens:,} tokens")

        return result

    # ---------------------------------------------------#
    # Method name: decode
    # ---------------------------------------------------#
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self.use_fast and self.tokenizer:
            filtered = [i for i in ids if i >= 4]
            return self.tokenizer.decode(filtered)
        return ''.join(self.inverse_vocab.get(i, '') for i in ids if i >= 4)

    # ---------------------------------------------------#
    # Method name: save
    # ---------------------------------------------------#
    def save(self, path: str):
        if self.use_fast and self.tokenizer:
            self.tokenizer.save(path)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'vocab': self.vocab, 'type': 'simple'}, f, indent=2)

    # ---------------------------------------------------#
    # Method name: load
    # ---------------------------------------------------#
    def load(self, path: str):
        try:
            self.tokenizer = Tokenizer.from_file(path)
            self.use_fast = True
            self.vocab_size = self.tokenizer.get_vocab_size()
        except Exception:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.vocab = {k: int(v) for k, v in data['vocab'].items()}
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}
            self.use_fast = False
            self.vocab_size = len(self.vocab)

    # ---------------------------------------------------#
    # Method name: get_vocab_size
    # ---------------------------------------------------#
    def get_vocab_size(self) -> int:
        if self.use_fast and self.tokenizer:
            return self.tokenizer.get_vocab_size()
        return len(self.vocab)


########################## END OF CLASS BytePairTokenizer ################################


# ============================================================================
# DATASET
# ============================================================================

if TORCH_AVAILABLE:
    # -----------------------------------------#
    # Class Name : CodeDataset
    # Calls: __init__, __len__, __getitem__, torch.tensor, torch.zeros
    # -----------------------------------------#
    class CodeDataset(Dataset):
        # ---------------------------------------------------#
        # Method name: __init__
        # ---------------------------------------------------#
        def __init__(self, token_ids: List[int], context_length: int,
                     stride: Optional[int] = None,
                     progress_callback: Optional[Callable[[float, str], None]] = None):
            self.context_length = context_length
            self.stride = stride if stride and stride > 0 else context_length
            self.stride = min(self.stride, context_length)

            if progress_callback:
                progress_callback(0.0, "Creating dataset tensors...")

            tokens = torch.tensor(token_ids, dtype=torch.long)
            n_tokens = len(tokens)

            if n_tokens <= context_length:
                self.n_samples = 1 if n_tokens > 1 else 0
            else:
                self.n_samples = max(1, (n_tokens - context_length) // self.stride)

            if self.n_samples == 0:
                raise ValueError(f"Not enough tokens ({n_tokens}) for context ({context_length})")

            self.x = torch.zeros((self.n_samples, context_length), dtype=torch.long)
            self.y = torch.zeros((self.n_samples, context_length), dtype=torch.long)

            for i in range(self.n_samples):
                start = i * self.stride
                end = start + context_length

                if end + 1 > n_tokens:
                    end = n_tokens - 1
                    start = end - context_length

                self.x[i] = tokens[start:start + context_length]
                self.y[i] = tokens[start + 1:start + context_length + 1]

                if progress_callback and i % 1000 == 0:
                    progress_callback(i / self.n_samples, f"Creating samples: {i:,}/{self.n_samples:,}")

            if progress_callback:
                progress_callback(1.0, f"Dataset ready: {self.n_samples:,} samples")

        # ---------------------------------------------------#
        # Method name: __len__
        # ---------------------------------------------------#
        def __len__(self) -> int:
            return self.n_samples

        # ---------------------------------------------------#
        # Method name: __getitem__
        # ---------------------------------------------------#
        def __getitem__(self, idx: int):
            return self.x[idx], self.y[idx]

########################## END OF CLASS CodeDataset ################################


# ============================================================================
# MODEL
# ============================================================================

if TORCH_AVAILABLE:
    # -----------------------------------------#
    # Class Name : MultiHeadAttention
    # Calls: __init__, forward, F.scaled_dot_product_attention, nn.Dropout, nn.Linear, out.transpose, t.view
    # -----------------------------------------#
    class MultiHeadAttention(nn.Module):
        # ---------------------------------------------------#
        # Method name: __init__
        # ---------------------------------------------------#
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.n_heads = config.n_heads
            self.head_dim = config.d_model // config.n_heads
            self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
            self.out = nn.Linear(config.d_model, config.d_model, bias=config.bias)
            self.dropout = nn.Dropout(config.dropout)

        # ---------------------------------------------------#
        # Method name: forward
        # ---------------------------------------------------#
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T, C = x.shape
            qkv = self.qkv(x).chunk(3, dim=-1)
            q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in qkv]
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.dropout.p if self.training else 0.0
            )
            return self.out(out.transpose(1, 2).contiguous().view(B, T, C))


    ########################## END OF CLASS MultiHeadAttention ################################


    class TransformerBlock(nn.Module):
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.ln1 = nn.LayerNorm(config.d_model)
            self.attn = MultiHeadAttention(config)
            self.ln2 = nn.LayerNorm(config.d_model)
            self.mlp = nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_ff, config.d_model),
                nn.Dropout(config.dropout)
            )
            self.use_checkpoint = config.gradient_checkpointing

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.use_checkpoint and self.training:
                return torch.utils.checkpoint.checkpoint(
                    self._forward_impl,
                    x,
                    use_reentrant=False,
                    preserve_rng_state=True  # ADD THIS
                )
            return self._forward_impl(x)

        def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    ########################## END OF CLASS TransformerBlock ################################

    # -----------------------------------------#
    # Class Name : GPTModel
    # Calls: __init__, _init_weights, forward, generate, count_parameters, F.softmax, logits.size, next_token.item, nn.Dropout, nn.Embedding, nn.LayerNorm, nn.Linear, nn.ModuleList, p.numel, torch.arange, torch.cat, torch.multinomial, torch.no_grad, torch.topk
    # -----------------------------------------#
    class GPTModel(nn.Module):
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.config = config
            self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
            self.drop = nn.Dropout(config.dropout)
            self.blocks = nn.ModuleList([
                TransformerBlock(config) for _ in range(config.n_layers)
            ])
            self.ln_f = nn.LayerNorm(config.d_model)
            self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.tok_emb.weight = self.head.weight
            self.apply(self._init_weights)

        def _init_weights(self, module: nn.Module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T = x.shape
            pos = torch.arange(T, device=x.device)
            x = self.drop(self.tok_emb(x) + self.pos_emb(pos))
            for block in self.blocks:
                x = block(x)
            return self.head(self.ln_f(x))

        @torch.no_grad()
        # ---------------------------------------------------#
        # Method name: generate
        # ---------------------------------------------------#
        def generate(self, idx: torch.Tensor, max_tokens: int = 100,
                     temperature: float = 0.8, top_k: int = 40,
                     repetition_penalty: float = 1.2) -> torch.Tensor:
            self.eval()
            for _ in range(max_tokens):
                idx_cond = idx[:, -self.config.max_seq_len:]
                logits = self(idx_cond)[:, -1]

                # Apply Repetition Penalty
                # >1.0 penalizes, <1.0 encourages
                if repetition_penalty != 1.0:
                    for i in range(idx.shape[0]):  # Batch loop
                        for token in set(idx[i].tolist()):
                            if logits[i, token] < 0:
                                logits[i, token] *= repetition_penalty
                            else:
                                logits[i, token] /= repetition_penalty

                # Temp scaling
                logits = logits / max(temperature, 1e-8)

                # Top-K
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1:]] = float('-inf')

                probs = F.softmax(logits, dim=-1)

                # Sample
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_token], dim=1)

                if next_token.item() == 3:  # EOS token
                    break
            return idx

        # ---------------------------------------------------#
        # Method name: count_parameters
        # ---------------------------------------------------#
        def count_parameters(self) -> int:
            return sum(p.numel() for p in self.parameters())


########################## END OF CLASS GPTModel ################################


# ============================================================================
# DATA PROCESSOR
# ============================================================================

# -----------------------------------------#
# Class Name : DataProcessor
# Calls: __init__, scan, process, content.replace, content.strip, executor.shutdown, executor.submit, folder_path.exists, folder_path.rglob, fp.relative_to, fp.stat, future.result, os.cpu_count, path.read_bytes, raw.decode, re.sub, time.time
# -----------------------------------------#
class DataProcessor:
    """Processes Python files with progress callbacks."""

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    # ---------------------------------------------------#
    # Method name: __init__
    # ---------------------------------------------------#
    def __init__(self, config: NormalizationConfig):
        self.config = config
        self.files: List[Dict[str, Any]] = []
        self.texts: List[str] = []
        self.files: List[Dict[str, Any]] = []

    # ---------------------------------------------------#
    # Method name: scan
    # ---------------------------------------------------#
    def scan(self, folder: str, progress_callback: Optional[Callable[[float, str], None]] = None) -> List[
        Dict[str, Any]]:
        """Scan folder for files with progress."""
        self.files = []
        folder_path = Path(folder)

        if not folder_path.exists():
            return self.files

        if progress_callback:
            progress_callback(0.0, "Scanning files...")

        # Expanded extensions list for general use
        extensions = {".py", ".txt", ".epub", ".html", ".js", ".c", ".cpp", ".h", ".java", ".cs", ".go", ".rs"}

        # First, collect all potential files to get total count
        try:
            # Use rglob("*") to get everything, then filter
            all_paths = list(folder_path.rglob("*"))
            total = len(all_paths)
        except OSError:
            total = 0
            all_paths = []

        for i, fp in enumerate(all_paths):
            if fp.is_file() and fp.suffix.lower() in extensions:
                try:
                    size = fp.stat().st_size
                    # No max size check here - we do it in filtering stage

                    self.files.append({
                        'path': str(fp),
                        'name': fp.name,
                        'relative': str(fp.relative_to(folder_path)),
                        'size': size
                    })
                except (OSError, PermissionError):
                    continue

            # Update progress every 1000 files to avoid UI lag
            if progress_callback and i % 1000 == 0:
                progress_callback(i / max(1, total), f"Found {len(self.files)} files...")

        if progress_callback:
            progress_callback(1.0, f"Found {len(self.files)} files")

        return self.files



    # ---------------------------------------------------#
    # Method name: process
    # ---------------------------------------------------#
    def process(self,
                progress_callback: Optional[Callable[[float, str], None]] = None,
                stop_check: Optional[Callable[[], bool]] = None) -> List[str]:
        """Process files with parallel reading and smooth progress."""
        total = len(self.files)

        if total == 0:
            return []

        if progress_callback:
            progress_callback(0.0, "Starting...")

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os

        # Limit workers to prevent Windows crashes - 4 is safe
        max_workers = min(4, os.cpu_count() or 4)

        def read_single_file(idx: int, f: Dict) -> Tuple[int, Optional[str]]:
            """Returns (index, content) for ordering."""
            try:
                content = None
                path = Path(f['path'])

                # Read in binary first, then decode - more stable on Windows
                raw = path.read_bytes()

                for enc in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        content = raw.decode(enc)
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue

                if content:
                    content = content.replace('\r\n', '\n').replace('\r', '\n')
                    content = content.replace('\t', ' ' * self.config.tabs_to_spaces)

                    if self.config.normalize_whitespace:
                        content = re.sub(r'\n{3,}', '\n\n', content)
                        content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)

                    if content.strip():
                        return (idx, f"# {f['relative']}\n{content.strip()}")
            except Exception:
                pass
            return (idx, None)

        results = [None] * total
        completed = 0
        last_callback_time = time.time()

        # Process in chunks to reduce memory pressure
        # chunk_size = 5000
        chunk_size = max(100, total_texts // (max_workers * 2))
        for chunk_start in range(0, total, chunk_size):
            if stop_check and stop_check():
                break

            chunk_end = min(chunk_start + chunk_size, total)
            chunk_files = [(i, self.files[i]) for i in range(chunk_start, chunk_end)]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(read_single_file, i, f): i
                    for i, f in chunk_files
                }

                for future in as_completed(futures):
                    if stop_check and stop_check():
                        executor.shutdown(wait=False, cancel_futures=True)
                        self.texts = [r for r in results if r is not None]
                        return self.texts

                    try:
                        idx, content = future.result(timeout=10)
                        if content:
                            results[idx] = content
                    except Exception:
                        pass

                    completed += 1

                    now = time.time()
                    if progress_callback and (now - last_callback_time >= 0.15 or completed == total):
                        progress_callback(completed / total, f"Reading: {completed:,}/{total:,}")
                        last_callback_time = now

        self.texts = [r for r in results if r is not None]

        if progress_callback:
            progress_callback(1.0, f"Processed {len(self.texts):,} files")

        return self.texts


########################## END OF CLASS DataProcessor ################################


# ============================================================================
# TRAINER
# ============================================================================

if TORCH_AVAILABLE:
    # -----------------------------------------#
    # Class Name : Trainer
    # Calls: __init__, train, _validate, _update_metrics, save_checkpoint, load_checkpoint, _log, stop, pause, F.cross_entropy, ckpt.get, logits.size, logits.view, loss.backward, loss.item, math.cos, math.exp, metrics.copy, os.makedirs, threading.Lock, threading.Thread, time.sleep, time.time, torch.compile, torch.device, torch.load, torch.no_grad, torch.save, x.to, y.to, y.view
    # -----------------------------------------#
    class Trainer:
        # ---------------------------------------------------#
        # Method name: __init__
        # ---------------------------------------------------#
        def __init__(self, model: GPTModel, tokenizer: BytePairTokenizer,
                     model_cfg: ModelConfig, train_cfg: TrainingConfig,
                     ckpt_cfg: CheckpointConfig, app_state: AppState):

            self.model = model
            self.tokenizer = tokenizer
            self.model_cfg = model_cfg
            self.train_cfg = train_cfg
            self.ckpt_cfg = ckpt_cfg
            self.app_state = app_state

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            # Enable TF32 and cuDNN optimizations
            if self.device.type == 'cuda':
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                print(f"[TRAINER] TF32 and cuDNN benchmark enabled")

                # Try torch.compile only if Triton is available (Linux mainly)
                import sys
                if sys.platform != 'win32' and hasattr(torch, 'compile'):
                    try:
                        self.model = torch.compile(self.model, mode='reduce-overhead')
                        print(f"[TRAINER] torch.compile enabled")
                    except Exception as e:
                        print(f"[TRAINER] torch.compile skipped: {e}")
                else:
                    print(f"[TRAINER] torch.compile skipped (Windows/unavailable)")

            # Use fused AdamW if available
            try:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=train_cfg.learning_rate,
                    weight_decay=train_cfg.weight_decay,
                    fused=self.device.type == 'cuda'
                )
                print(f"[TRAINER] Fused AdamW enabled")
            except TypeError:
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=train_cfg.learning_rate,
                    weight_decay=train_cfg.weight_decay
                )
                print(f"[TRAINER] Standard AdamW")

            self.use_amp = train_cfg.precision != 'fp32' and self.device.type == 'cuda'
            self.amp_dtype = torch.bfloat16 if train_cfg.precision == 'bf16' else torch.float16
            self.scaler = GradScaler() if train_cfg.precision == 'fp16' else None

            self.epoch = 0
            self.step = 0
            self.best_loss = float('inf')
            self.train_losses: List[float] = []
            self.val_losses: List[float] = []
            self.start_time = 0.0

            self._save_lock = threading.Lock()

            self.log_cb: Optional[Callable[[str], None]] = None
            self.metrics_cb: Optional[Callable[[Dict], None]] = None

            os.makedirs(ckpt_cfg.output_dir, exist_ok=True)

        # ---------------------------------------------------#
        # Method name: train
        # ---------------------------------------------------#
        def train(self, train_loader: DataLoader, val_loader: DataLoader):
            self.app_state.state = AppState.TRAINING
            self.start_time = time.time()

            total_steps = len(train_loader) * self.train_cfg.epochs
            warmup_steps = int(total_steps * self.train_cfg.warmup_ratio)
            patience = 0
            last_save_step = 0

            # Gradient accumulation
            grad_accum = getattr(self.train_cfg, 'gradient_accumulation', 1)
            effective_batch = self.train_cfg.batch_size * grad_accum

            self._log(f"{Icons.ROCKET} Training started")
            self._log(f"   Steps/epoch: {len(train_loader)} | Total: {total_steps:,}")
            self._log(f"   Batch: {self.train_cfg.batch_size} × {grad_accum} accum = {effective_batch} effective")
            self._log(f"   Context: {self.train_cfg.context_length}")

            if self.ckpt_cfg.resume_from and os.path.exists(self.ckpt_cfg.resume_from):
                self.load_checkpoint(self.ckpt_cfg.resume_from)

            last_metrics_time = 0
            last_log_time = 0

            for epoch in range(self.epoch, self.train_cfg.epochs):
                if self.app_state.state == AppState.IDLE:
                    break

                self.epoch = epoch
                self.model.train()
                epoch_loss = 0.0
                epoch_steps = 0
                accum_loss = 0.0

                self._log(f"\n{Icons.ARROW_RIGHT} Epoch {epoch + 1}/{self.train_cfg.epochs}")

                for batch_idx, (x, y) in enumerate(train_loader):
                    if self.app_state.state == AppState.IDLE:
                        break

                    while self.app_state.state == AppState.PAUSED:
                        time.sleep(0.1)

                    # x = x.to(self.device, non_blocking=True)
                    # y = y.to(self.device, non_blocking=True)
                    x = x.to(self.device, non_blocking=True).long()
                    y = y.to(self.device, non_blocking=True).long()
                    # Learning rate schedule
                    if self.step < warmup_steps:
                        lr = self.train_cfg.learning_rate * (self.step + 1) / warmup_steps
                    else:
                        progress = (self.step - warmup_steps) / max(1, total_steps - warmup_steps)
                        lr = self.train_cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

                    for pg in self.optimizer.param_groups:
                        pg['lr'] = lr

                    # Forward pass
                    try:
                        if self.use_amp:
                            with autocast('cuda', dtype=self.amp_dtype):
                                logits = self.model(x)
                                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                                loss = loss / grad_accum  # Scale loss

                            if self.scaler:
                                self.scaler.scale(loss).backward()
                            else:
                                loss.backward()
                        else:
                            logits = self.model(x)
                            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                            loss = loss / grad_accum
                            loss.backward()

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            if CUDA_AVAILABLE:
                                torch.cuda.empty_cache()
                            raise RuntimeError(f"OOM - reduce batch to {self.train_cfg.batch_size // 2}")
                        raise

                    accum_loss += loss.item() * grad_accum

                    # Optimizer step every grad_accum batches
                    if (batch_idx + 1) % grad_accum == 0:
                        if self.use_amp and self.scaler:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.gradient_clip)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.gradient_clip)
                            self.optimizer.step()

                        self.optimizer.zero_grad(set_to_none=True)

                        loss_val = accum_loss
                        accum_loss = 0.0
                        epoch_loss += loss_val
                        epoch_steps += 1
                        self.step += 1

                        # Update progress
                        overall_progress = self.step / (total_steps // grad_accum)
                        epoch_progress = (batch_idx + 1) / len(train_loader)
                        self.app_state.set_progress(overall_progress, f"Epoch {epoch + 1}/{self.train_cfg.epochs}")
                        self.app_state.set_sub_progress(epoch_progress, f"Step {self.step} | Loss: {loss_val:.4f}")

                        # Metrics every 2 seconds
                        now = time.time()
                        if now - last_metrics_time >= 2.0:
                            self._update_metrics(epoch, batch_idx, len(train_loader), loss_val, lr,
                                                 total_steps // grad_accum)
                            last_metrics_time = now

                        # Log every 30 seconds
                        if now - last_log_time >= 30.0 or self.step <= 3:
                            elapsed = now - self.start_time
                            if self.step > 0:
                                steps_per_sec = self.step / elapsed
                                remaining = (total_steps // grad_accum) - self.step
                                eta_sec = remaining / max(0.1, steps_per_sec)
                                eta_str = str(timedelta(seconds=int(eta_sec)))

                                if self.step <= 3:
                                    gpu_mem = torch.cuda.memory_allocated() / 1e9 if CUDA_AVAILABLE else 0
                                    self._log(f"   [DIAG] VRAM: {gpu_mem:.1f}GB | {steps_per_sec:.2f} step/s")

                                self._log(
                                    f"   Step {self.step:,} | Loss: {loss_val:.4f} | LR: {lr:.2e} | ETA: {eta_str}")
                            last_log_time = now

                        # Checkpoint
                        if self.ckpt_cfg.save_every_steps > 0:
                            if (self.step - last_save_step) >= self.ckpt_cfg.save_every_steps:
                                self.save_checkpoint(f"step_{self.step}")
                                last_save_step = self.step

                # End of epoch
                avg_loss = epoch_loss / max(1, epoch_steps)
                self.train_losses.append(avg_loss)

                val_loss = self._validate(val_loader)
                self.val_losses.append(val_loss)

                self._log(f"   Train: {avg_loss:.4f} | Val: {val_loss:.4f}")
                self._update_metrics(epoch, len(train_loader), len(train_loader), avg_loss, lr,
                                     total_steps // grad_accum)

                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    patience = 0
                    self.save_checkpoint("best", wait=True)
                    self._log(f"   {Icons.SUCCESS} New best: {val_loss:.4f}")
                else:
                    patience += 1

                self.save_checkpoint(f"epoch_{epoch + 1}")

                if self.train_cfg.early_stopping and patience >= self.train_cfg.early_stopping_patience:
                    self._log(f"{Icons.WARNING} Early stopping at epoch {epoch + 1}")
                    break

            self.save_checkpoint("final", wait=True)

            total_time = time.time() - self.start_time
            self._log(f"\n{Icons.SPARKLE} Training complete!")
            self._log(f"   Time: {timedelta(seconds=int(total_time))}")
            self._log(f"   Best Val Loss: {self.best_loss:.4f}")

            self.app_state.state = AppState.IDLE

        # ---------------------------------------------------#
        # Method name: _validate
        # ---------------------------------------------------#
        def _validate(self, loader: DataLoader) -> float:
            self.model.eval()
            total_loss = 0.0
            n_batches = 0

            with torch.no_grad():
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    if self.use_amp:
                        with autocast('cuda', dtype=self.amp_dtype):
                            loss = F.cross_entropy(
                                self.model(x).view(-1, self.model.config.vocab_size),
                                y.view(-1)
                            )
                    else:
                        loss = F.cross_entropy(
                            self.model(x).view(-1, self.model.config.vocab_size),
                            y.view(-1)
                        )
                    total_loss += loss.item()
                    n_batches += 1

            self.model.train()
            return total_loss / max(1, n_batches)

        # ---------------------------------------------------#
        # Method name: _update_metrics
        # ---------------------------------------------------#
        def _update_metrics(self, epoch: int, batch_idx: int, total_batches: int,
                            loss: float, lr: float, total_steps: int):
            elapsed = time.time() - self.start_time
            steps_per_sec = self.step / max(0.1, elapsed)
            eta = (total_steps - self.step) / max(0.1, steps_per_sec)
            tokens_per_sec = (self.step * self.train_cfg.batch_size *
                              self.train_cfg.context_length) / max(0.1, elapsed)

            gpu_mem = 0.0
            gpu_total = 1.0
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1e9
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9

            metrics = {
                'epoch': epoch + 1,
                'total_epochs': self.train_cfg.epochs,
                'step': batch_idx + 1,
                'total_steps': total_batches,
                'global_step': self.step,
                'total_global_steps': total_steps,
                'train_loss': loss,
                'val_loss': self.val_losses[-1] if self.val_losses else 0,
                'best_loss': self.best_loss if self.best_loss != float('inf') else 0,
                'lr': lr,
                'tokens_per_sec': tokens_per_sec,
                'steps_per_sec': steps_per_sec,
                'elapsed': elapsed,
                'eta': eta,
                'gpu_mem': gpu_mem,
                'gpu_total': gpu_total,
                'perplexity': math.exp(min(loss, 20))
            }

            if self.metrics_cb:
                try:
                    self.metrics_cb(metrics.copy())
                except Exception:
                    pass

        # ---------------------------------------------------#
        # Method name: save_checkpoint
        # ---------------------------------------------------#
        def save_checkpoint(self, tag: str, wait: bool = False):
            def _save():
                with self._save_lock:
                    path = os.path.join(self.ckpt_cfg.output_dir,
                                        f"{self.ckpt_cfg.checkpoint_name}_{tag}.pt")
                    checkpoint = {
                        'epoch': self.epoch,
                        'step': self.step,
                        'model': self.model.state_dict(),
                        'model_state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict() if self.ckpt_cfg.save_optimizer else None,
                        'best_loss': self.best_loss,
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses,
                        'model_cfg': asdict(self.model_cfg),
                        'train_cfg': asdict(self.train_cfg),
                    }
                    torch.save(checkpoint, path)

                    if self.ckpt_cfg.save_tokenizer and tag in ('final', 'best'):
                        tok_path = os.path.join(self.ckpt_cfg.output_dir,
                                                f"{self.ckpt_cfg.checkpoint_name}_tokenizer.json")
                        self.tokenizer.save(tok_path)

                    self._log(f"   {Icons.SAVE} Saved: {tag}")

            if wait:
                _save()
            else:
                threading.Thread(target=_save, daemon=True).start()

        # ---------------------------------------------------#
        # Method name: load_checkpoint
        # ---------------------------------------------------#
        def load_checkpoint(self, path: str):
            self._log(f"{Icons.LOAD} Loading checkpoint...")
            ckpt = torch.load(path, map_location=self.device, weights_only=False)

            state_dict = ckpt.get('model', ckpt.get('model_state_dict', {}))
            self.model.load_state_dict(state_dict)

            epoch = start_epoch
            self.epoch = ckpt.get('epoch', 0) + 1
            self.step = ckpt.get('step', 0)
            self.best_loss = ckpt.get('best_loss', float('inf'))
            self.train_losses = ckpt.get('train_losses', [])
            self.val_losses = ckpt.get('val_losses', [])

            self._log(f"   Resuming from epoch {self.epoch}, step {self.step}")

        # ---------------------------------------------------#
        # Method name: _log
        # ---------------------------------------------------#
        def _log(self, msg: str):
            print(msg)
            if self.log_cb:
                try:
                    self.log_cb(msg)
                except Exception:
                    pass

        # ---------------------------------------------------#
        # Method name: stop
        # ---------------------------------------------------#
        def stop(self):
            self._log(f"{Icons.STOP} Stopping...")
            self.app_state.state = AppState.IDLE

        # ---------------------------------------------------#
        # Method name: pause
        # ---------------------------------------------------#
        def pause(self):
            if self.app_state.state == AppState.TRAINING:
                self.app_state.state = AppState.PAUSED
                self._log(f"{Icons.PAUSE} Paused")
            elif self.app_state.state == AppState.PAUSED:
                self.app_state.state = AppState.TRAINING
                self._log(f"{Icons.PLAY} Resumed")

########################## END OF CLASS Trainer ################################


# ============================================================================
# DPO TRAINER - Direct Preference Optimization
# ============================================================================

if TORCH_AVAILABLE:
    # -----------------------------------------#
    # Class Name : DPOTrainer
    # Calls: __init__, _get_log_probs, dpo_loss, train, _save_checkpoint, _log
    # -----------------------------------------#
    class DPOTrainer:
        """
        Direct Preference Optimization trainer for RLHF.
        
        DPO fine-tunes a model using preference pairs (chosen vs rejected outputs).
        It learns to increase probability of chosen outputs while decreasing
        probability of rejected outputs, relative to a frozen reference model.
        """
        
        def __init__(self, model: GPTModel, ref_model: GPTModel,
                     tokenizer: BytePairTokenizer,
                     model_cfg: ModelConfig, train_cfg: TrainingConfig,
                     ckpt_cfg: CheckpointConfig, app_state: AppState):
            
            self.model = model  # Policy model (will be updated)
            self.ref_model = ref_model  # Reference model (frozen)
            self.tokenizer = tokenizer
            self.model_cfg = model_cfg
            self.train_cfg = train_cfg
            self.ckpt_cfg = ckpt_cfg
            self.app_state = app_state
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.ref_model.to(self.device)
            self.ref_model.eval()  # Freeze reference
            
            # Freeze ref model parameters
            for param in self.ref_model.parameters():
                param.requires_grad = False
            
            # DPO hyperparameters
            self.beta = 0.1  # KL penalty coefficient
            self.dpo_lr = train_cfg.learning_rate * 0.1  # Lower LR for DPO
            
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.dpo_lr,
                weight_decay=train_cfg.weight_decay
            )
            
            self.use_amp = train_cfg.precision != 'fp32' and self.device.type == 'cuda'
            self.amp_dtype = torch.bfloat16 if train_cfg.precision == 'bf16' else torch.float16
            
            self.log_cb: Optional[Callable[[str], None]] = None
            self.metrics_cb: Optional[Callable[[Dict], None]] = None
            
            os.makedirs(ckpt_cfg.output_dir, exist_ok=True)
        
        def _get_log_probs(self, model: GPTModel, input_ids: torch.Tensor, 
                          labels: torch.Tensor) -> torch.Tensor:
            """Get log probabilities for each token."""
            ctx = nullcontext() if model == self.model else torch.no_grad()
            with ctx:
                if self.use_amp:
                    with autocast('cuda', dtype=self.amp_dtype):
                        logits = model(input_ids)
                else:
                    logits = model(input_ids)
            
            # Shift for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Get log probs
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Gather the log probs for actual tokens
            token_log_probs = torch.gather(
                log_probs, 
                dim=-1, 
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Sum over sequence (average per token)
            return token_log_probs.sum(dim=-1)
        
        def dpo_loss(self, chosen_ids: torch.Tensor, rejected_ids: torch.Tensor,
                     chosen_labels: torch.Tensor, rejected_labels: torch.Tensor) -> Tuple[torch.Tensor, float, float, float]:
            """
            Compute DPO loss.
            
            DPO Loss = -log(sigmoid(beta * (log_pi(chosen) - log_pi(rejected) 
                                           - log_ref(chosen) + log_ref(rejected))))
            """
            # Policy model log probs
            pi_chosen = self._get_log_probs(self.model, chosen_ids, chosen_labels)
            pi_rejected = self._get_log_probs(self.model, rejected_ids, rejected_labels)
            
            # Reference model log probs
            ref_chosen = self._get_log_probs(self.ref_model, chosen_ids, chosen_labels)
            ref_rejected = self._get_log_probs(self.ref_model, rejected_ids, rejected_labels)
            
            # DPO loss
            pi_diff = pi_chosen - pi_rejected
            ref_diff = ref_chosen - ref_rejected
            
            loss = -F.logsigmoid(self.beta * (pi_diff - ref_diff)).mean()
            
            # Metrics
            with torch.no_grad():
                chosen_reward = (pi_chosen - ref_chosen).mean().item()
                rejected_reward = (pi_rejected - ref_rejected).mean().item()
                accuracy = (pi_diff > ref_diff).float().mean().item()
            
            return loss, chosen_reward, rejected_reward, accuracy
        
        def train(self, preference_data: List[Dict[str, str]]):
            """
            Train with DPO.
            
            preference_data: List of dicts with keys:
                - 'prompt': The input prompt
                - 'chosen': The preferred response
                - 'rejected': The dispreferred response
            """
            self.app_state.state = AppState.TRAINING
            start_time = time.time()
            
            self._log(f"{Icons.ROCKET} DPO Training started")
            self._log(f"   Preference pairs: {len(preference_data)}")
            self._log(f"   Beta (KL penalty): {self.beta}")
            self._log(f"   Learning rate: {self.dpo_lr}")
            
            total_steps = len(preference_data) * self.train_cfg.epochs
            step = 0
            
            for epoch in range(self.train_cfg.epochs):
                if self.app_state.state == AppState.IDLE:
                    break
                
                self.model.train()
                epoch_loss = 0.0
                epoch_acc = 0.0
                
                self._log(f"\n{Icons.ARROW_RIGHT} Epoch {epoch + 1}/{self.train_cfg.epochs}")
                
                # Shuffle data
                import random
                shuffled_data = preference_data.copy()
                random.shuffle(shuffled_data)
                
                for i, pair in enumerate(shuffled_data):
                    if self.app_state.state == AppState.IDLE:
                        break
                    
                    # Tokenize
                    prompt = pair['prompt']
                    chosen_text = prompt + pair['chosen']
                    rejected_text = prompt + pair['rejected']
                    
                    chosen_ids = torch.tensor(
                        [self.tokenizer.encode(chosen_text)[:self.model_cfg.max_seq_len]],
                        device=self.device
                    )
                    rejected_ids = torch.tensor(
                        [self.tokenizer.encode(rejected_text)[:self.model_cfg.max_seq_len]],
                        device=self.device
                    )
                    
                    # Pad to same length
                    max_len = max(chosen_ids.size(1), rejected_ids.size(1))
                    if chosen_ids.size(1) < max_len:
                        chosen_ids = F.pad(chosen_ids, (0, max_len - chosen_ids.size(1)))
                    if rejected_ids.size(1) < max_len:
                        rejected_ids = F.pad(rejected_ids, (0, max_len - rejected_ids.size(1)))
                    
                    # Forward + loss
                    self.optimizer.zero_grad()
                    
                    loss, chosen_r, rejected_r, acc = self.dpo_loss(
                        chosen_ids, rejected_ids,
                        chosen_ids, rejected_ids
                    )
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.gradient_clip)
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    epoch_acc += acc
                    step += 1
                    
                    # Progress
                    self.app_state.set_progress(step / total_steps, f"DPO Epoch {epoch + 1}")
                    
                    if i % 10 == 0:
                        self._log(f"   Step {step} | Loss: {loss.item():.4f} | Acc: {acc:.2%}")
                
                avg_loss = epoch_loss / len(preference_data)
                avg_acc = epoch_acc / len(preference_data)
                self._log(f"   Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.2%}")
                
                # Save checkpoint
                self._save_checkpoint(f"dpo_epoch_{epoch + 1}")
            
            self._save_checkpoint("dpo_final")
            
            total_time = time.time() - start_time
            self._log(f"\n{Icons.SPARKLE} DPO Training complete!")
            self._log(f"   Total time: {timedelta(seconds=int(total_time))}")
            
            self.app_state.state = AppState.IDLE
        
        def _save_checkpoint(self, tag: str):
            """Save DPO checkpoint."""
            path = os.path.join(
                self.ckpt_cfg.output_dir,
                f"{self.ckpt_cfg.checkpoint_name}_{tag}.pt"
            )
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_cfg': asdict(self.model_cfg),
            }, path)
            self._log(f"   {Icons.SAVE} Saved: {tag}")
        
        def _log(self, msg: str):
            """Log message to callback if available."""
            print(msg)
            if self.log_cb:
                try:
                    self.log_cb(msg)
                except Exception:
                    pass

    ########################## END OF CLASS DPOTrainer ################################


# ============================================================================
# SETTINGS TOOLTIPS
# ============================================================================

TOOLTIPS = {
    # Data
    'folder': "Select a folder containing Python (.py) files.\nAll subdirectories will be scanned recursively.\nRecommended: 100+ files for meaningful training.",

    # Model Architecture
    'd_model': "Embedding dimension - size of token embeddings.\nLarger = more capacity but slower.\nCommon values: 256, 512, 768, 1024\nMust be divisible by number of attention heads.",
    'n_heads': "Number of attention heads.\nMust divide evenly into embedding dim.\nCommon values: 4, 8, 12, 16\nHead dim = d_model / n_heads (ideally 64 or 128)",
    'n_layers': "Number of transformer layers.\nMore layers = deeper understanding.\nCommon: 4-12 for small, 12-48 for large models.",
    'd_ff': "Feed-forward network dimension.\nTypically 4x the embedding dimension.\nExample: 2048 for d_model=512",
    'vocab_size': "Maximum vocabulary size for tokenizer.\n32000 is standard for code.\nLarger = more precise but uses more memory.",
    'max_seq': "Maximum context length in tokens.\nLonger = more context but more VRAM.\n512 for 8GB GPU, 1024+ for 16GB+",
    'dropout': "Dropout probability for regularization.\nPrevents overfitting.\nTypical: 0.1 for most cases, 0.0 for large datasets.",

    # Training
    'lr': "Learning rate - step size for optimization.\n• 0.0003: Good starting point\n• 0.0001: More stable, slower\n• 0.00005: Fine-tuning",
    'batch_size': "Samples processed together.\nLarger = faster but more VRAM.\n8GB: 16-32, 16GB: 64-128, 24GB+: 128+",
    'epochs': "Complete passes through the dataset.\nMore = better learning but risk overfitting.\nUse early stopping to auto-stop.",
    'stride': "Step size for sliding window.\nSmaller = more samples but redundancy.\nTypically half of context length.",
    'warmup': "Fraction of training with LR warmup.\nStabilizes early training.\n0.1 (10%) is standard.",
    'grad_clip': "Maximum gradient norm.\nPrevents exploding gradients.\n1.0 is safe default.",
    'val_split': "Fraction for validation.\nUsed to monitor overfitting.\n0.1 (10%) is standard.",
    'precision': "Training precision:\n• BF16: Best for RTX 30xx+, fastest\n• FP16: Good for older GPUs\n• FP32: Full precision, slowest",
    'early_stopping': "Stop if validation loss stops improving.\nSaves time and prevents overfitting.",
    'patience': "Epochs to wait before early stopping.\nHigher = more tolerance for fluctuation.",

    # Checkpoints
    'output_dir': "Directory to save checkpoints.\nWill be created if it doesn't exist.",
    'ckpt_name': "Prefix for checkpoint filenames.\nExample: 'python_llm' creates 'python_llm_best.pt'",
    'save_steps': "Save checkpoint every N steps.\nSet to 0 to disable step checkpoints.\n100-500 for long training runs.",
    'resume': "Select a .pt checkpoint to resume.\nModel architecture should match.",
    'incremental': "Reuse existing tokenizer.\nFor adding new data without retraining tokenizer.\nRequires resume checkpoint with tokenizer.",
}

################################
# Class : HW Monitor
################################
class HardwareMonitor(ttk.Frame):
    """Real-time hardware graph with boxed sections."""

    def __init__(self, parent, width=180, height=240, update_ms=1000):
        super().__init__(parent, style="Sidebar.TFrame")
        self.width = width
        self.height = 60  # Height per individual graph
        self.update_ms = update_ms
        self.history_len = 60

        # Data stores
        self.cpu_data = [0] * self.history_len
        self.ram_data = [0] * self.history_len
        self.vram_data = [0] * self.history_len

        self._build_ui()
        self._start_monitoring()

    def _build_ui(self):
        # CPU Box
        self._create_graph_box("CPU Usage", NavyTheme.ACCENT_CYAN, "cpu_lbl", "cpu_canvas")

        # RAM Box
        self._create_graph_box("System RAM", NavyTheme.WARNING, "ram_lbl", "ram_canvas")

        # VRAM Box
        self._create_graph_box("GPU Memory", NavyTheme.SUCCESS, "vram_lbl", "vram_canvas")

    def _create_graph_box(self, title, color, lbl_attr, canvas_attr):
        # Container with border look
        container = tk.Frame(self, bg=NavyTheme.BORDER_DARK, padx=1, pady=1)
        container.pack(fill="x", padx=0, pady=4) # No padding

        inner = tk.Frame(container, bg=NavyTheme.NAVY_DARKEST)
        inner.pack(fill="both")

        # Header
        header = tk.Frame(inner, bg=NavyTheme.NAVY_MEDIUM, height=20)
        header.pack(fill="x")

        tk.Label(header, text=title, bg=NavyTheme.NAVY_MEDIUM, fg=color,
                 font=("Segoe UI", 7, "bold")).pack(side="left", padx=5)

        lbl = tk.Label(header, text="0%", bg=NavyTheme.NAVY_MEDIUM, fg="#fdfdfd",
                       font=("Segoe UI", 7))
        lbl.pack(side="right", padx=5)
        setattr(self, lbl_attr, lbl)

        # Canvas
        canvas = tk.Canvas(inner, width=self.width, height=self.height,
                           bg=NavyTheme.NAVY_DARKEST, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        setattr(self, canvas_attr, canvas)

        # Draw static grid initially
        self._draw_grid(canvas)
        
        # Bind resize
        canvas.bind("<Configure>", lambda e, c=canvas: self._on_resize(e, c))

    def _on_resize(self, event, canvas):
        self._draw_grid(canvas)

    def _draw_grid(self, canvas):
        canvas.delete("grid_line")
        w = canvas.winfo_width()
        if w <= 1: w = self.width
        h = self.height
        
        # Horizontal lines (25%, 50%, 75%)
        for i in range(1, 4):
            y = i * (h / 4)
            canvas.create_line(0, y, w, y, fill="#E0B0FF", width=1, dash=(2, 4), tag="grid_line")

        # Vertical lines
        for i in range(1, 5):
            x = i * (w / 5)
            canvas.create_line(x, 0, x, h, fill="#E0B0FF", width=1, dash=(2, 4), tag="grid_line")

    def _start_monitoring(self):
        self._update_data()

    def _update_data(self):
        try:
            import psutil

            # CPU
            # Show MAX usage of any single core
            cpu = max(psutil.cpu_percent(percpu=True))
            self.cpu_data.append(cpu)
            self.cpu_data.pop(0)
            self.cpu_lbl.config(text=f"{cpu:.0f}%")
            self._plot_line(self.cpu_canvas, self.cpu_data, NavyTheme.ACCENT_CYAN)

            # RAM
            ram = psutil.virtual_memory().percent
            self.ram_data.append(ram)
            self.ram_data.pop(0)
            self.ram_lbl.config(text=f"{ram:.0f}%")
            self._plot_line(self.ram_canvas, self.ram_data, NavyTheme.WARNING)

            # VRAM
            vram_pct = 0
            if TORCH_AVAILABLE and torch.cuda.is_available():
                mem = torch.cuda.memory_reserved(0)
                tot = torch.cuda.get_device_properties(0).total_memory
                vram_pct = (mem / tot) * 100
                self.vram_lbl.config(text=f"{vram_pct:.0f}%")

            self.vram_data.append(vram_pct)
            self.vram_data.pop(0)
            self._plot_line(self.vram_canvas, self.vram_data, NavyTheme.SUCCESS)

        except Exception:
            pass

        self.after(self.update_ms, self._update_data)


    def _plot_line(self, canvas, data, color):
        w = canvas.winfo_width()
        if w <= 1: w = self.width
        h = self.height
        
        # Clear old lines only (keep grid)
        canvas.delete("data_line")
        canvas.delete("data_fill")

        points = []
        if len(data) > 1:
            dx = w / (len(data) - 1)
    
            for i, val in enumerate(data):
                x = i * dx
                y = h - (val / 100 * h)
                points.extend([x, y])
    
            if len(points) >= 4:
                # Semi-transparent fill (simulated by stipple)
                poly_points = points + [w, h, 0, h]
                canvas.create_polygon(poly_points, fill=color, outline="",
                                      stipple="gray25", tag="data_fill")
                # Solid line
                canvas.create_line(points, fill=color, width=2, tag="data_line")

# ============================================================================
# MAIN GUI APPLICATION
# ============================================================================


# ============================================================
# DPO UTILITIES & HELPERS
# ============================================================

DOC_TYPES = {
    "Story/Prose": {"extensions": [".txt", ".md"], "mode": "prose"},
    "Poetry": {"extensions": [".txt", ".md"], "mode": "poem"},
    "Python Code": {"extensions": [".py"], "mode": "python"},
    "C/C++ Code": {"extensions": [".c", ".h", ".cpp", ".hpp"], "mode": "c_cpp"},
}

def read_file(path: str) -> str:
    """Read text file with encoding fallbacks."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except Exception:
        return ""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc).replace("\r\n", "\n")
        except Exception:
            continue
    return ""

def clean_text(s: str) -> str:
    """Basic text cleanup."""
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def is_junk(s: str) -> bool:
    """Check if text is boilerplate/junk."""
    t = s.lower().strip()
    if not t or len(t.split()) < 30: return True
    junk_phrases = ["all rights reserved", "table of contents", "copyright", "chapter ", "prologue", "epilogue", "project gutenberg"]
    for phrase in junk_phrases:
        if phrase in t: return True
    return False

def extract_samples(text: str, mode: str, count: int = 10) -> List[str]:
    """Extract good text samples from document."""
    text = clean_text(text)
    if not text: return []
    samples = []
    if mode in ("python", "c_cpp"):
        lines = text.split("\n")
        i = 0
        while i < len(lines) and len(samples) < count:
            if lines[i].strip() and not lines[i].strip().startswith(("#", "//", "/*")):
                block = []
                while i < len(lines) and len(block) < 10:
                    block.append(lines[i])
                    i += 1
                sample = "\n".join(block).strip()
                if len(sample.split()) >= 30: samples.append(sample)
            else: i += 1
    else:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        for p in paras:
            if not is_junk(p) and len(samples) < count: samples.append(p)
    return samples

def find_files_recursive(folder: str, extensions: List[str], max_files: int = 500, min_kb: int = 0, max_kb: int = 0) -> List[str]:
    """Find matching files in folder with size filtering."""
    files = []
    ext_set = set(e.lower() for e in extensions)
    min_bytes = min_kb * 1024
    max_bytes = max_kb * 1024 if max_kb > 0 else float('inf')
    for root, _, names in os.walk(folder):
        for name in names:
            if os.path.splitext(name)[1].lower() in ext_set:
                path = os.path.join(root, name)
                try:
                    size = os.path.getsize(path)
                    if min_bytes <= size <= max_bytes:
                        files.append(path)
                        if len(files) >= max_files: return files
                except OSError: continue
    return files


DPO_SETTINGS_FILE = "dpo_creator_settings.ini"
DPO_PROMPTS_FILE = "dpo_prompts.ini"
DPO_PAIRS_FILE = "dpo_pairs.json"
SFT_FILE = "sft_data.json"

class DPOSettings:
    """Simple settings manager for DPO Creator."""
    DEFAULTS = {
        "folder": "", "doc_type": "Story/Prose", "tokenizer_path": "",
        "model_path": "", "grader_path": "", "seed": "42",
        "max_tokens": "150", "min_file_kb": "1", "max_file_kb": "2048",
        "max_files": "500", "auto_grade": "0",
    }
    def __init__(self):
        self.data = dict(self.DEFAULTS)
        self._load()

    def _load(self):
        if os.path.exists(DPO_SETTINGS_FILE):
            try:
                cfg = configparser.ConfigParser()
                cfg.read(DPO_SETTINGS_FILE, encoding="utf-8")
                if cfg.has_section("Settings"):
                    for k, v in cfg.items("Settings"): self.data[k] = v
            except Exception: pass

    def save(self):
        try:
            cfg = configparser.ConfigParser()
            cfg.add_section("Settings")
            for k, v in self.data.items(): cfg.set("Settings", k, str(v))
            with open(DPO_SETTINGS_FILE, "w", encoding="utf-8") as f: cfg.write(f)
            return True
        except Exception: return False

    def get(self, key: str) -> str: return self.data.get(key, "")
    def get_int(self, key: str, default: int = 0) -> int:
        try: return int(self.data.get(key, default))
        except ValueError: return default
    def set(self, key: str, value): self.data[key] = str(value)

class AutoGrader:
    """Uses a small model to automatically rank outputs."""
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.loaded = False

    def load(self, model_path: str, device: str = "cuda"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
            is_local = os.path.isdir(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=is_local)
            if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
            
            kwargs = {"trust_remote_code": True, "local_files_only": is_local}
            if self.device.type == "cuda": kwargs["torch_dtype"] = torch.float16
            else: kwargs["torch_dtype"] = torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to(self.device)
            self.model.eval()
            self.loaded = True
            return True
        except Exception as e:
            print(f"Failed to load grader: {e}")
            return False

    def score_single(self, prompt: str, response: str) -> float:
        if not self.loaded: return 5.0
        full_text = f"{prompt}\n\n{response}"
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
        return max(0.0, min(10.0, 12.0 - loss * 2.0))

    def score_all(self, prompt: str, responses: List[str]) -> List[Tuple[str, float]]:
        labels = ["A", "B", "C"]
        results = []
        for i, resp in enumerate(responses):
            score = self.score_single(prompt, resp) if resp.strip() else 0.0
            results.append((labels[i], score))
        return results


# -----------------------------------------#
# Class Name : LLMTrainerGUI
# Calls: __init__, _on_close, _init_variables, _get_preset_config, _apply_preset, _auto_select_preset, _build_ui, _build_sidebar_content, _build_header, _show_page, _create_page_frame, _clear_cache, _build_data_page, _set_file_limit, _set_size_filter, _get_size_in_bytes, _get_filtered_files, _update_effective_files, _update_dataset_estimate, _set_file_limit, _update_effective_files, _update_dataset_estimate, _build_model_page, _build_training_page, _build_checkpoint_page, _build_progress_page, _build_help_page, _on_state_change, _update_state_ui, _start_monitors, _update_vram, _update_progress_display, _update_model_size_estimate, _browse_folder, _browse_output, _browse_resume, _scan_folder, _clear_vram, _log, _update_metrics_display, _get_configs, _start_training, _training_worker, _pause_training, _stop_training, _save_now, _plot_loss, _test_model, _save_settings, _load_settings, _load_settings_dialog, NavyTheme.apply, TOOLTIPS.get, ToolTip.create, actions_frame.pack, arch_frame.pack, auto_btn.pack, ax.grid, ax.legend, ax.plot, ax.set_facecolor, ax.set_title, ax.set_xlabel, ax.set_ylabel, ax.tick_params, b.configure, browse_btn.pack, btn.bind, btn.configure, btn.pack, btn_frame.pack, btn_row.pack, check_text.split, checkpoint_name.replace, ckpt.get, clear_cache_btn.pack, clear_vram_btn.pack, content.replace, content.strip, count_row.pack, current_tokens.extend, data_row.pack, datetime.now, dir_entry.pack, dir_row.pack, entry.grid, errors.append, es_check.pack, es_frame.pack, es_row.pack, estimate_frame.pack, eta_calc.get_eta, eta_calc.reset, filedialog.askdirectory, filedialog.askopenfilename, files.copy, filter_frame.pack, filtered.append, folder_entry.pack, folder_frame.pack, frame.pack_forget, gc.collect, gpu_config.copy, gpu_row.pack, grid.pack, header.pack, help_container.pack, help_scrollbar.config, help_scrollbar.pack, help_text.config, help_text.insert, help_text.pack, hint_lbl.grid, hyper_frame.pack, icon_map.get, incr_check.pack, json.dump, json.load, label.lower, lbl.grid, limit_entry.pack, load_settings_btn.pack, log_container.pack, log_frame.pack, log_scrollbar.config, log_scrollbar.pack, lr_combo.grid, main.add, main.pack, matplotlib.use, max_entry.pack, max_tokens_var.get, messagebox.askyesno, messagebox.showerror, messagebox.showinfo, messagebox.showwarning, metrics.get, metrics_frame.pack, min_entry.pack, multipliers.get, name_entry.pack, name_row.pack, out_frame.pack, output_frame.pack, output_text.delete, output_text.get, output_text.insert, output_text.pack, page_id.title, param_frame.pack, patience_entry.pack, plot_btn.pack, plt.show, plt.subplots, plt.tight_layout, prec_desc.pack, prec_frame.pack, prec_row.pack, preset_desc.pack, preset_frame.pack, presets.get, progress_frame.pack, prompt_entry.bind, prompt_entry.get, prompt_entry.insert, prompt_entry.pack, prompt_frame.pack, quality_row.pack, random.seed, random.shuffle, raw.decode, rb.pack, resume_entry.pack, resume_frame.pack, resume_row.pack, row1.pack, s.get, s.items, save_btn.pack, save_settings_btn.pack, scan_btn.pack, scrolledtext.ScrolledText, seed_check.pack, seed_entry.pack, shuffle_row.pack, size_frame.pack, size_row.pack, stats_frame.pack, status_label.config, status_label.pack, status_map.get, status_row.pack, steps_entry.pack, sys_frame.pack, temp_var.get, test_btn.pack, test_win.clipboard_append, test_win.clipboard_clear, test_win.configure, test_win.geometry, test_win.title, test_win.update, texts.append, thread.is_alive, thread.start, threading.Thread, time.time, title_frame.pack, titles.get, tk.BooleanVar, tk.Button, tk.StringVar, tk.Text, tk.Toplevel, token_counts.append, token_counts.extend, topk_var.get, torch.load, torch.no_grad, torch.tensor, torch.zeros, traceback.format_exc, ttk.Button, ttk.Checkbutton, ttk.Combobox, ttk.Entry, ttk.Frame, ttk.Label, ttk.LabelFrame, ttk.PanedWindow, ttk.Progressbar, ttk.Radiobutton, ttk.Scrollbar, ttk.Separator, unit_combo.pack, unit_combo.set, var.set, var.trace_add, warnings.append
# -----------------------------------------#
class LLMTrainerGUI:
    """Professional LLM Trainer with side panel navigation."""

    VERSION = "1.0"
    SETTINGS_FILE = "trainer_settings.json"
    INDEX_FILE = "sllm_projects.ini"

    # ---------------------------------------------------#
    # Method name: __init__
    # ---------------------------------------------------#
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"Python LLM Trainer v{self.VERSION}")
        self.root.title(f"Python LLM Trainer v{self.VERSION}")
        self.root.state("zoomed") # Start maximized
        self.root.minsize(1000, 700)
        
        try:
            self.root.iconbitmap("sllm2.ico")
        except: pass

        # Apply theme
        NavyTheme.apply(self.root)

        # State management
        self.app_state = AppState()
        self.gui_queue = GUIUpdateQueue(root)


        # Configurations
        self.model_cfg = ModelConfig()
        self.train_cfg = TrainingConfig()
        self.ckpt_cfg = CheckpointConfig()
        self.norm_cfg = NormalizationConfig()

        # Components
        self.processor = DataProcessor(self.norm_cfg)
        self.tokenizer: Optional[BytePairTokenizer] = None
        self.model: Optional[Any] = None
        self.trainer: Optional[Any] = None

        # Current page
        self.current_page = "data"
        self.page_buttons: Dict[str, ttk.Button] = {}
        self.page_frames: Dict[str, ttk.Frame] = {}

        # Stop flag - MUST be before _start_monitors()
        self._stop_requested = False
        
        # Prevent save dialog during initialization
        self._prevent_save = True

        # Build UI
        self._init_variables()
        self._build_ui()
        self._load_settings()
        self._update_state_ui()
        self._start_monitors()

        # State listener
        self.app_state.add_listener(self._on_state_change)

        # Professional GUI enhancements
        self.toast = ToastManager(self.root)
        self.shortcuts = ShortcutManager(self.root)
        self._setup_shortcuts()
        
        # Track dirty state for window title
        self._base_title = f"Python LLM Trainer v{self.VERSION}"
        
        # Cleanup
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------------------------------------------#
    # Method name: _on_close
    # ---------------------------------------------------#
    def _on_close(self):
        """Handle window close with dirty check."""
        # Check dirty state
        if self.app_state.is_dirty():
            result = messagebox.askyesnocancel(
                "Unsaved Changes", 
                "You have unsaved changes. Save before exit?"
            )
            if result is None:  # Cancel
                return
            elif result:  # Yes - save
                self._save_project_settings()
                
        if self.app_state.is_busy():
            if not messagebox.askyesno("Confirm", "Operation in progress. Exit anyway?"):
                return
            if self.trainer:
                self.trainer.stop()

        self.gui_queue.stop()
        self.root.destroy()

    # ---------------------------------------------------#
    # Method name: _setup_shortcuts
    # ---------------------------------------------------#
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        self.shortcuts.register("Ctrl+S", self._save_project_settings, "Save Project Settings")
        self.shortcuts.register("Ctrl+O", self._load_project_settings, "Load Project Settings")
        self.shortcuts.register("Ctrl+Return", self._start_training, "Start Training")
        self.shortcuts.register("Escape", self._stop_training, "Stop Training")
        self.shortcuts.register("F1", lambda: self._show_page("help"), "Show Help")
        self.shortcuts.register("Ctrl+Shift+S", self._show_shortcuts, "Show Shortcuts")
        self.shortcuts.register("Ctrl+1", lambda: self._show_page("data"), "Data Tab")
        self.shortcuts.register("Ctrl+2", lambda: self._show_page("model"), "Model Tab")
        self.shortcuts.register("Ctrl+3", lambda: self._show_page("training"), "Training Tab")
        self.shortcuts.register("Ctrl+4", lambda: self._show_page("progress"), "Progress Tab")
        
    # ---------------------------------------------------#
    # Method name: _show_shortcuts
    # ---------------------------------------------------#
    def _show_shortcuts(self):
        """Show keyboard shortcuts dialog."""
        self.shortcuts.show_shortcuts_dialog()
        
    # ---------------------------------------------------#
    # Method name: _update_title_dirty
    # ---------------------------------------------------#
    def _update_title_dirty(self):
        """Update window title with dirty indicator."""
        if not hasattr(self, '_base_title'):
            return  # Not yet initialized
            
        if self.app_state.is_dirty():
            self.root.title(f"● {self._base_title}")
        else:
            self.root.title(self._base_title)

    # ---------------------------------------------------#
    # Method name: _on_var_change
    # ---------------------------------------------------#
    def _on_var_change(self, page_id: str):
        """Handle variable changes to update dirty state."""
        if getattr(self, '_prevent_save', False):
            return
        self.app_state.mark_dirty(page_id, True)
        self._update_title_dirty()

    # ---------------------------------------------------#
    # Method name: _init_variables
    # ---------------------------------------------------#
    def _init_variables(self):
        """Initialize all tkinter variables."""
        
        def _track(var, page="settings"):
            """Helper to track dirty state."""
            var.trace_add("write", lambda *a: self._on_var_change(page))
            return var

        # Data
        self.folder_var = _track(tk.StringVar(), "data")
        self.file_limit_var = _track(tk.StringVar(value="0"), "data")
        self.random_seed_var = _track(tk.BooleanVar(value=True), "data")
        self.seed_value_var = _track(tk.StringVar(value="42"), "data")
        self.min_file_size_var = _track(tk.StringVar(value="0"), "data")  # 0 = no minimum
        self.max_file_size_var = _track(tk.StringVar(value="0"), "data")  # 0 = no maximum
        self.size_unit_var = _track(tk.StringVar(value="KB"), "data")  # KB or MB
        
        # Data Processing Options (Missing before)
        self.norm_cfg_whitespace = _track(tk.BooleanVar(value=True), "data")
        self.norm_cfg_comments = _track(tk.BooleanVar(value=True), "data")
        self.norm_cfg_docstrings = _track(tk.BooleanVar(value=True), "data")

        # Model
        self.d_model_var = _track(tk.StringVar(value="512"), "model")
        self.n_heads_var = _track(tk.StringVar(value="8"), "model")
        self.n_layers_var = _track(tk.StringVar(value="6"), "model")
        self.d_ff_var = _track(tk.StringVar(value="2048"), "model")
        self.vocab_size_var = _track(tk.StringVar(value="32000"), "model")
        self.max_seq_var = _track(tk.StringVar(value="512"), "model")
        self.dropout_var = _track(tk.StringVar(value="0.1"), "model")
        self.grad_ckpt_var = _track(tk.BooleanVar(value=False), "model")

        # Project
        self.project_name_var = _track(tk.StringVar(value="my_first_model"), "project")

        # Training
        self.lr_var = _track(tk.StringVar(value="0.0003"), "training")
        self.lr_options = ["0.001", "0.0005", "0.0003", "0.0001", "0.00005", "0.00001"]
        self.grad_accum_var = _track(tk.StringVar(value="1"), "training")

        self.batch_var = _track(tk.StringVar(value="128"), "training")
        self.epochs_var = _track(tk.StringVar(value="10"), "training")
        self.stride_var = _track(tk.StringVar(value="256"), "training")
        self.warmup_var = _track(tk.StringVar(value="0.1"), "training")
        self.grad_clip_var = _track(tk.StringVar(value="1.0"), "training")
        self.val_split_var = _track(tk.StringVar(value="0.1"), "training")
        self.precision_var = _track(tk.StringVar(value="bf16"), "training")
        self.early_stop_var = _track(tk.BooleanVar(value=True), "training")
        self.patience_var = _track(tk.StringVar(value="3"), "training")
        
        self._save_requested = False
        
        # Validation
        self.validation_enabled_var = _track(tk.BooleanVar(value=True), "training")
        self.validation_percent_var = _track(tk.StringVar(value="20"), "training")

        # Checkpoints
        self.output_var = _track(tk.StringVar(value="./checkpoints"), "checkpoint")
        self.ckpt_name_var = _track(tk.StringVar(value="python_llm"), "checkpoint")
        self.save_steps_var = tk.StringVar(value="100")
        self.resume_var = tk.StringVar()
        self.incremental_var = tk.BooleanVar(value=False)

        # Refinement Settings
        self.refine_min_size_var = tk.StringVar(value="0")  # Min file size (0 = no limit)
        self.refine_max_size_var = tk.StringVar(value="0")  # Max file size (0 = no limit)
        self.refine_size_unit_var = tk.StringVar(value="KB")
        self.refine_min_quality_var = tk.StringVar(value="50")  # Quality threshold 0-100
        self.refine_exact_dedup_var = tk.BooleanVar(value=True)
        self.refine_near_dedup_var = tk.BooleanVar(value=True)
        self.refine_near_threshold_var = tk.StringVar(value="0.90")
        self.refine_fingerprint_size_var = tk.StringVar(value="200")
        self.refine_cleaning_mode_var = tk.StringVar(value="auto")  # auto/fiction/code/docs/none
        self.refine_stop_flag = False
        
        # Context Extension Settings (separate from normal training)
        self.ctx_ext_checkpoint_var = tk.StringVar()  # Source checkpoint path
        self.ctx_ext_target_var = tk.StringVar(value="2048")  # Target context length
        self.ctx_ext_batch_var = tk.StringVar(value="8")  # Reduced batch for larger context
        self.ctx_ext_stride_var = tk.StringVar(value="1024")  # 50% overlap
        self.ctx_ext_accum_var = tk.StringVar(value="16")  # Higher accumulation
        self.ctx_ext_epochs_var = tk.StringVar(value="3")  # Fine-tuning epochs
        self.ctx_ext_lr_var = tk.StringVar(value="0.00002")  # Lower LR for fine-tuning
        self.ctx_ext_detected_context = tk.StringVar(value="Unknown")  # Detected from checkpoint

        # DPO Creator
        self.dpo_settings = DPOSettings()
        self.dpo_folder_var = tk.StringVar(value=self.dpo_settings.get("folder"))
        self.dpo_type_var = tk.StringVar(value=self.dpo_settings.get("doc_type"))
        self.dpo_min_kb_var = tk.StringVar(value=self.dpo_settings.get("min_file_kb"))
        self.dpo_max_kb_var = tk.StringVar(value=self.dpo_settings.get("max_file_kb"))
        self.dpo_max_files_var = tk.StringVar(value=self.dpo_settings.get("max_files"))
        self.dpo_mode_var = tk.StringVar(value=self.dpo_settings.get("mode") or "DPO")
        
        self.dpo_files = []
        self.dpo_samples = []
        self.dpo_current_file_idx = 0
        self.dpo_creator_model = None
        self.dpo_generator_tokenizer = None
        self.dpo_grader = None
        self.dpo_outputs = ["", "", ""]

    # ---------------------------------------------------#
    # Method name: _get_preset_config
    # ---------------------------------------------------#
    def _get_preset_config(self, preset_name: str, file_count: int = 0, gpu_vram_gb: float = 0) -> Dict[str, Any]:
        """Get preset configuration based on name and detected parameters."""

        # Auto-detect GPU VRAM if not provided
        if gpu_vram_gb <= 0 and TORCH_AVAILABLE and CUDA_AVAILABLE:
            try:
                gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            except Exception:
                gpu_vram_gb = 8

        # Estimate tokens from file count (avg ~2500 tokens per Python file)
        estimated_tokens = file_count * 2500 if file_count > 0 else 0

        presets = {
            # =================================================================
            # GPU-based presets
            # =================================================================
            "gpu_8gb": {
                "d_model": "256", "n_heads": "4", "n_layers": "6", "d_ff": "1024",
                "vocab_size": "32000", "max_seq": "512", "dropout": "0.1",
                "lr": "0.0003", "batch": "32", "epochs": "10", "stride": "256",
                "warmup": "0.1", "grad_clip": "1.0", "val_split": "0.1",
                "precision": "fp16", "early_stop": True, "patience": "3",
                "description": "Conservative settings for 8GB VRAM GPUs"
            },
            "gpu_12gb": {
                "d_model": "512", "n_heads": "8", "n_layers": "8", "d_ff": "2048",
                "vocab_size": "32000", "max_seq": "512", "dropout": "0.1",
                "lr": "0.0003", "batch": "64", "epochs": "10", "stride": "256",
                "warmup": "0.1", "grad_clip": "1.0", "val_split": "0.1",
                "precision": "bf16", "early_stop": True, "patience": "3",
                "description": "Balanced settings for 12GB VRAM GPUs"
            },
            "gpu_16gb": {
                "d_model": "512", "n_heads": "8", "n_layers": "8", "d_ff": "2048",
                "vocab_size": "32000", "max_seq": "1024", "dropout": "0.1",
                "lr": "0.0003", "batch": "32", "epochs": "6", "stride": "512",
                "warmup": "0.1", "grad_clip": "1.0", "val_split": "0.1",
                "precision": "bf16", "early_stop": True, "patience": "3",
                "description": "Optimized for 16GB VRAM GPUs"

            },
            "gpu_24gb": {
                "d_model": "1024", "n_heads": "16", "n_layers": "16", "d_ff": "4096",
                "vocab_size": "50000", "max_seq": "1024", "dropout": "0.1",
                "lr": "0.0003", "batch": "192", "epochs": "6", "stride": "512",
                "warmup": "0.1", "grad_clip": "1.0", "val_split": "0.1",
                "precision": "bf16", "early_stop": True, "patience": "3",
                "description": "High capacity for 24GB VRAM GPUs (RTX 4090 5090)"
            },

            # =================================================================
            # Dataset size presets
            # =================================================================
            "dataset_small": {
                "d_model": "256", "n_heads": "4", "n_layers": "4", "d_ff": "1024",
                "vocab_size": "16000", "max_seq": "512", "dropout": "0.15",
                "lr": "0.0005", "batch": "64", "epochs": "20", "stride": "256",
                "warmup": "0.1", "grad_clip": "1.0", "val_split": "0.15",
                "precision": "bf16", "early_stop": True, "patience": "5",
                "description": "For <50k files. Smaller model, more epochs, higher dropout."
            },
            "dataset_medium": {
                "d_model": "512", "n_heads": "8", "n_layers": "8", "d_ff": "2048",
                "vocab_size": "32000", "max_seq": "1024", "dropout": "0.1",
                "lr": "0.0003", "batch": "128", "epochs": "8", "stride": "512",
                "warmup": "0.1", "grad_clip": "1.0", "val_split": "0.1",
                "precision": "bf16", "early_stop": True, "patience": "3",
                "description": "For 50k-200k files. Balanced model size."
            },
            "dataset_large": {
                "d_model": "768", "n_heads": "12", "n_layers": "12", "d_ff": "3072",
                "vocab_size": "50000", "max_seq": "1024", "dropout": "0.1",
                "lr": "0.0003", "batch": "128", "epochs": "4", "stride": "512",
                "warmup": "0.1", "grad_clip": "1.0", "val_split": "0.05",
                "precision": "bf16", "early_stop": True, "patience": "2",
                "description": "For 200k-500k files. Larger model, fewer epochs."
            },
            "dataset_xlarge": {
                "d_model": "1024", "n_heads": "16", "n_layers": "16", "d_ff": "4096",
                "vocab_size": "50000", "max_seq": "1024", "dropout": "0.05",
                "lr": "0.0002", "batch": "96", "epochs": "3", "stride": "512",
                "warmup": "0.05", "grad_clip": "1.0", "val_split": "0.03",
                "precision": "bf16", "early_stop": True, "patience": "2",
                "description": "For 500k+ files. Large model, minimal epochs."
            },

            # =================================================================
            # Quality presets
            # =================================================================
            "fast_experiment": {
                "d_model": "256", "n_heads": "4", "n_layers": "4", "d_ff": "1024",
                "vocab_size": "16000", "max_seq": "256", "dropout": "0.1",
                "lr": "0.001", "batch": "128", "epochs": "3", "stride": "128",
                "warmup": "0.05", "grad_clip": "1.0", "val_split": "0.1",
                "precision": "bf16", "early_stop": False, "patience": "3",
                "description": "Quick test run. Small model, few epochs, fast iteration."
            },
            "balanced": {
                "d_model": "512", "n_heads": "8", "n_layers": "6", "d_ff": "2048",
                "vocab_size": "32000", "max_seq": "512", "dropout": "0.1",
                "lr": "0.0003", "batch": "64", "epochs": "10", "stride": "256",
                "warmup": "0.1", "grad_clip": "1.0", "val_split": "0.1",
                "precision": "bf16", "early_stop": True, "patience": "3",
                "description": "Good balance of quality and speed. Works on most GPUs."
            },
            "high_quality": {
                "d_model": "768", "n_heads": "12", "n_layers": "12", "d_ff": "3072",
                "vocab_size": "50000", "max_seq": "1024", "dropout": "0.1",
                "lr": "0.0002", "batch": "64", "epochs": "15", "stride": "512",
                "warmup": "0.1", "grad_clip": "1.0", "val_split": "0.1",
                "precision": "bf16", "early_stop": True, "patience": "5",
                "description": "Higher quality output. Longer training, larger model."
            },
        }

        return presets.get(preset_name, presets["balanced"])

    def _get_project_dir(self):
        """Get current project directory, create if missing."""
        # Clean name
        name = self.project_name_var.get().strip().replace(" ", "_")
        # Remove weird chars
        name = "".join(c for c in name if c.isalnum() or c in ('_', '-'))
        if not name: name = "default_project"

        path = os.path.join("projects", name)
        os.makedirs(path, exist_ok=True)
        return path

    # ---------------------------------------------------#
    # Method name: _apply_preset
    # ---------------------------------------------------#
    def _apply_preset(self, preset_name: str):
        """Apply a preset configuration."""
        file_count = len(self.processor.files) if self.processor.files else 0

        preset = self._get_preset_config(preset_name, file_count)

        # Apply all values
        self.d_model_var.set(preset["d_model"])
        self.n_heads_var.set(preset["n_heads"])
        self.n_layers_var.set(preset["n_layers"])
        self.d_ff_var.set(preset["d_ff"])
        self.vocab_size_var.set(preset["vocab_size"])
        self.max_seq_var.set(preset["max_seq"])
        self.dropout_var.set(preset["dropout"])
        self.lr_var.set(preset["lr"])
        self.batch_var.set(preset["batch"])
        self.epochs_var.set(preset["epochs"])
        self.stride_var.set(preset["stride"])
        self.warmup_var.set(preset["warmup"])
        self.grad_clip_var.set(preset["grad_clip"])
        self.val_split_var.set(preset["val_split"])
        self.precision_var.set(preset["precision"])
        self.early_stop_var.set(preset["early_stop"])
        self.patience_var.set(preset["patience"])

        self._log(f"{Icons.SPARKLE} Applied preset: {preset_name}")
        self._log(f"   {preset['description']}")

        # Update estimates
        self._update_model_size_estimate()

    # ---------------------------------------------------#
    # Method name: _auto_select_preset
    # ---------------------------------------------------#
    def _auto_select_preset(self):
        """Automatically select best preset based on GPU and dataset."""
        file_count = len(self.processor.files) if self.processor.files else 0

        # Detect GPU VRAM
        gpu_vram_gb = 0
        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            try:
                gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            except Exception:
                pass

        if file_count == 0:
            messagebox.showwarning("No Data", "Scan a folder first to auto-select optimal preset.")
            return

        # Select GPU preset
        if gpu_vram_gb >= 20:
            gpu_preset = "gpu_24gb"
        elif gpu_vram_gb >= 14:
            gpu_preset = "gpu_16gb"
        elif gpu_vram_gb >= 10:
            gpu_preset = "gpu_12gb"
        else:
            gpu_preset = "gpu_8gb"

        # Select dataset preset
        if file_count < 50000:
            data_preset = "dataset_small"
        elif file_count < 200000:
            data_preset = "dataset_medium"
        elif file_count < 500000:
            data_preset = "dataset_large"
        else:
            data_preset = "dataset_xlarge"

        # Merge: use GPU preset as base, adjust epochs/batch from data preset
        gpu_config = self._get_preset_config(gpu_preset, file_count, gpu_vram_gb)
        data_config = self._get_preset_config(data_preset, file_count, gpu_vram_gb)

        # Use GPU config for model size, data config for training params
        final_config = gpu_config.copy()
        final_config["epochs"] = data_config["epochs"]
        final_config["val_split"] = data_config["val_split"]
        final_config["patience"] = data_config["patience"]
        final_config["dropout"] = data_config["dropout"]

        # Apply
        self.d_model_var.set(final_config["d_model"])
        self.n_heads_var.set(final_config["n_heads"])
        self.n_layers_var.set(final_config["n_layers"])
        self.d_ff_var.set(final_config["d_ff"])
        self.vocab_size_var.set(final_config["vocab_size"])
        self.max_seq_var.set(final_config["max_seq"])
        self.dropout_var.set(final_config["dropout"])
        self.lr_var.set(final_config["lr"])
        self.batch_var.set(final_config["batch"])
        self.epochs_var.set(final_config["epochs"])
        self.stride_var.set(final_config["stride"])
        self.warmup_var.set(final_config["warmup"])
        self.grad_clip_var.set(final_config["grad_clip"])
        self.val_split_var.set(final_config["val_split"])
        self.precision_var.set(final_config["precision"])
        self.early_stop_var.set(final_config["early_stop"])
        self.patience_var.set(final_config["patience"])

        self._log(f"{Icons.SPARKLE} Auto-selected optimal settings:")
        self._log(f"   GPU: {gpu_vram_gb:.0f}GB → {gpu_preset}")
        self._log(f"   Dataset: {file_count:,} files → {data_preset}")
        self._log(f"   {final_config['description']}")

        self._update_model_size_estimate()

        messagebox.showinfo(
            "Preset Applied",
            f"Optimal settings applied for:\n\n"
            f"• GPU: {gpu_vram_gb:.0f}GB VRAM\n"
            f"• Dataset: {file_count:,} files\n\n"
            f"Review settings in Model and Training tabs."
        )

    # ---------------------------------------------------#
    # Method name: _apply_target_size
    # ---------------------------------------------------#
    def _apply_target_size(self, size_label: str):
        """Apply parameters for a specific target model size."""
        if not size_label or size_label == "Select Target Size":
            return

        # Definitions: (d_model, n_layers, n_heads, required_vram_gb)
        # VRAM estimate is rough for training (fp16/bf16 + optimizer + batch=8)
        configs = {
            "10M (Nano)":    (256, 6, 4, 2),
            "50M (Micro)":   (512, 6, 8, 4),
            "124M (Small)":  (768, 12, 12, 6),
            "350M (Medium)": (1024, 24, 16, 12),
            "774M (Large)":  (1280, 36, 20, 24),
            "1.5B (XL)":     (1600, 48, 25, 48),
        }

        cfg = configs.get(size_label)
        if not cfg:
            return
        
        d_model, n_layers, n_heads, req_vram = cfg

        # Hardware check
        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            try:
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                if vram < req_vram:
                    if not messagebox.askyesno(
                        "VRAM Warning", 
                        f"Target {size_label} typically requires ~{req_vram}GB VRAM for training.\n"
                        f"You have {vram:.1f}GB.\n\n"
                        "Apply anyway? (You may need to reduce batch size drastically)"
                    ):
                        return
            except:
                pass

        # Apply
        self.d_model_var.set(str(d_model))
        self.n_layers_var.set(str(n_layers))
        self.n_heads_var.set(str(n_heads))
        self.d_ff_var.set(str(d_model * 4))
        
        # Scaling adjustments for larger models
        if "774M" in size_label or "1.5B" in size_label:
             self.grad_ckpt_var.set(True) # Force gradient checkpointing for large models
             self._log(f"{Icons.INFO} Enabled Gradient Checkpointing for large model")

        self.max_seq_var.set("1024") # Standard context

        self._log(f"{Icons.SPARKLE} Applied target size: {size_label}")
        self._update_model_size_estimate()

    # ---------------------------------------------------#
    # Method name: _build_ui
    # ---------------------------------------------------#
    # ---------------------------------------------------#
    # Method name: _build_ui
    # ---------------------------------------------------#
    def _build_ui(self):
        """Build the main UI with side panel."""
        
        # Status bar at very bottom (Pack first to ensure visibility logic)
        self.status_bar = StatusBar(self.root, self.app_state)
        self.status_bar.pack(side="bottom", fill="x")
        
        # Main container with PanedWindow for resizable splitter
        main = ttk.PanedWindow(self.root, orient="horizontal")
        main.pack(fill="both", expand=True)

        # Sidebar
        self.sidebar_frame = ttk.Frame(main, style="Sidebar.TFrame", width=NavyTheme.SIDEBAR_WIDTH)
        self.sidebar_frame.pack_propagate(False)
        main.add(self.sidebar_frame, weight=0)

        # Content area
        content_container = ttk.Frame(main, style="Content.TFrame")
        main.add(content_container, weight=1)

        # Build sidebar content
        self._build_sidebar_content(self.sidebar_frame)

        # Header
        self._build_header(content_container)

        # Page container
        self.page_container = ttk.Frame(content_container)
        self.page_container.pack(fill="both", expand=True, padx=20, pady=10)

        # Build all pages
        self._build_data_page()
        self._build_refinement_page()
        self._build_model_page()
        self._build_training_page()
        self._build_context_extend_page()
        self._build_dpo_page()
        self._build_checkpoint_page()
        self._build_progress_page()

        # Show initial page
        self._show_page("data")
        
        # Enable save tracking
        self._prevent_save = False

    # ---------------------------------------------------#
    # Method name: _build_sidebar_content
    # ---------------------------------------------------#
    def _build_sidebar_content(self, sidebar: ttk.Frame):
        """Build the sidebar navigation content."""
        # sidebar = ttk.Frame(parent, style="Sidebar.TFrame", width=NavyTheme.SIDEBAR_WIDTH)
        # sidebar.pack(side="left", fill="y")
        # sidebar.pack_propagate(False)

        # Logo/Title
        title_frame = ttk.Frame(sidebar, style="Sidebar.TFrame")
        title_frame.pack(fill="x", pady=20, padx=15)

        lbl = ttk.Label(
            title_frame,
            text="  sLLM Trainer",
            style="Title.TLabel",
            compound="left"
        )
        # Load icon
        try:
            if os.path.exists("sllm2.png"):
                img = tk.PhotoImage(file="sllm2.png")
                # Resize if too big (assuming standard icon might be large)
                # Simple check: if width > 48, subsample
                if img.width() >= 64:
                    scale = img.width() // 32
                    img = img.subsample(scale, scale)
                self.logo_img = img # Keep reference
                lbl.config(image=self.logo_img)
        except Exception: pass
        
        lbl.pack(anchor="w")

        ttk.Label(
            title_frame,
            text=f"v{self.VERSION}",
            style="Sidebar.TLabel"
        ).pack(anchor="w")

        # Project Section
        proj_frame = ttk.Frame(sidebar, style="Sidebar.TFrame")
        proj_frame.pack(fill="x", padx=10, pady=(0, 10))

        # Project Section
        ttk.Label(sidebar, text="PROJECT", style="Sidebar.TLabel", font=("Segoe UI", 8, "bold")).pack(anchor="w",
                                                                                                      padx=10,
                                                                                                      pady=(15, 0))

        # Project Name Entry
        ttk.Entry(sidebar, textvariable=self.project_name_var).pack(fill="x", padx=10, pady=5)

        # Buttons Row
        proj_btns = ttk.Frame(sidebar, style="Sidebar.TFrame")
        proj_btns.pack(fill="x", padx=10, pady=2)

        # Load Button
        ttk.Button(proj_btns, text=f"{Icons.LOAD} Load", width=8,
                   command=self._load_project_settings).pack(side="left", padx=(0, 5))

        # Save Button
        ttk.Button(proj_btns, text=f"{Icons.SAVE} Save", width=8,
                   command=self._save_project_settings).pack(side="left")

        # Separator
        ttk.Separator(sidebar).pack(fill="x", padx=10, pady=10)

        # Navigation buttons
        self.nav_items = [
            ("data", Icons.DATA, "Dataset Setup", "#fcd34d"),        # Amber
            ("refine", Icons.SCAN, "Refine Data", "#22d3ee"),        # Cyan
            ("dpo", Icons.SPARKLE, "DPO/RLHF", "#e879f9"),           # Fuchsia
            ("model", Icons.MODEL, "Model Architecture", "#60a5fa"), # Blue
            ("training", Icons.TRAINING, "Training Configuration", "#fb923c"), # Orange
            ("context_extend", Icons.BOLT, "Increase Context", "#facc15"),     # Yellow
            ("checkpoint", Icons.CHECKPOINT, "Checkpoints & Export", "#4ade80"), # Green
            ("progress", Icons.FIRE, "Progress & Logs", "#f87171"),  # Red
        ]

        for page_id, icon, label, color in self.nav_items:
            btn = SidebarButton(
                sidebar,
                text=label,
                icon=icon,
                icon_color=color,
                command=lambda p=page_id: self._show_page(p)
            )
            btn.pack(fill="x", pady=1)
            self.page_buttons[page_id] = btn


        # Bottom section - System info
        ttk.Separator(sidebar).pack(fill="x", padx=10, pady=10, side="bottom")

        # Bottom section - System info
        sys_frame = ttk.Frame(sidebar, style="Sidebar.TFrame")
        sys_frame.pack(fill="x", side="bottom", padx=0, pady=10)

        # Hardware Monitor Graph
        self.monitor = HardwareMonitor(sys_frame, width=NavyTheme.SIDEBAR_WIDTH, height=180)
        self.monitor.pack(fill="both", expand=True, pady=5)

        # GPU Name
        gpu_name = GPU_NAME[:25] + "..." if len(GPU_NAME) > 25 else GPU_NAME
        ttk.Label(
            sys_frame,
            text=f"{Icons.GPU} {gpu_name}",
            style="Sidebar.TLabel",
            font=("Segoe UI", 8)
        ).pack(anchor="w", pady=(5, 0))

        ttk.Separator(sidebar).pack(fill="x", padx=10, pady=5, side="bottom")

    # ---------------------------------------------------#
    # Method name: _build_header
    # ---------------------------------------------------#
    def _build_header(self, parent: ttk.Frame):
        """Build the header bar."""
        header = ttk.Frame(parent)
        header.pack(fill="x", padx=20, pady=15)

        # Page title (will be updated)
        self.page_title = ttk.Label(header, text="Data", style="Header.TLabel")
        self.page_title.pack(side="left")

        # Action buttons
        btn_frame = ttk.Frame(header)
        btn_frame.pack(side="right")

        self.start_btn = ttk.Button(
            btn_frame,
            text=f"{Icons.PLAY} Start Training",
            style="Accent.TButton",
            command=self._start_training
        )
        self.start_btn.pack(side="left", padx=5)
        ToolTip.create(self.start_btn, "Start the training process")

        self.pause_btn = ttk.Button(
            btn_frame,
            text=f"{Icons.PAUSE} Pause",
            command=self._pause_training
        )
        self.pause_btn.pack(side="left", padx=5)
        ToolTip.create(self.pause_btn, "Pause/Resume training")

        self.stop_btn = ttk.Button(
            btn_frame,
            text=f"{Icons.STOP} Stop",
            style="Danger.TButton",
            command=self._stop_training
        )
        self.stop_btn.pack(side="left", padx=5)
        ToolTip.create(self.stop_btn, "Stop training (saves checkpoint)")

        # Divider
        ttk.Separator(btn_frame, orient="vertical").pack(side="left", fill="y", padx=10, pady=5)

        # Utility Buttons (Moved from Progress Page)
        self.plot_btn = ttk.Button(
            btn_frame, 
            text=f"{Icons.PLOT} Plot Loss", 
            command=self._plot_loss
        )
        self.plot_btn.pack(side="left", padx=5)
        
        self.test_btn = ttk.Button(
            btn_frame, 
            text=f"{Icons.TEST} Test Model", 
            command=self._test_model
        )
        self.test_btn.pack(side="left", padx=5)
        
        self.save_btn = ttk.Button(
            btn_frame, 
            text=f"{Icons.SAVE} Save Now", 
            command=self._save_now
        )
        self.save_btn.pack(side="left", padx=5)

        # Help Button
        self.help_btn = ttk.Button(
            btn_frame, 
            text=f"{Icons.HELP}", 
            width=4,
            style="Accent.TButton",
            command=self._open_help_window
        )
        self.help_btn.pack(side="right", padx=10)


    # ---------------------------------------------------#
    # Method name: _show_page
    # ---------------------------------------------------#
    def _show_page(self, page_id: str):
        """Show a specific page."""
        # Update button styles
        for pid, btn in self.page_buttons.items():
             btn.set_selected(pid == page_id)

        # Hide all pages
        for frame in self.page_frames.values():
            frame.pack_forget()

        # Show selected page
        if page_id in self.page_frames:
            self.page_frames[page_id].pack(fill="both", expand=True)

        # Update title
        titles = {
            "data": f"{Icons.DATA} Data Settings",
            "model": f"{Icons.MODEL} Model Architecture",
            "training": f"{Icons.TRAINING} Training Configuration",
            "dpo": f"{Icons.SPARKLE} DPO/RLHF Training",
            "checkpoint": f"{Icons.CHECKPOINT} Checkpoint Settings",
            "progress": f"{Icons.FIRE} Training Progress",
            "help": f"{Icons.HELP} Help & Documentation",
        }
        self.page_title.config(text=titles.get(page_id, page_id.title()))

        self.current_page = page_id

    # ---------------------------------------------------#
    # Method name: _create_page_frame
    # ---------------------------------------------------#
    def _create_page_frame(self, page_id: str) -> ttk.Frame:
        """Create and register a page frame."""
        frame = ttk.Frame(self.page_container)
        self.page_frames[page_id] = frame
        return frame

    # ---------------------------------------------------#
    # Method name: _clear_cache
    # ---------------------------------------------------#
    def _clear_cache(self):
        """Clear cached texts and tokenizer."""
        self._cached_texts = None
        self._cached_file_count = 0
        self._cached_tokenizer_vocab = 0
        self.tokenizer = None
        gc.collect()
        # self._log(f"{Icons.CLEAR} Cache cleared - next run will reprocess all files")
        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            torch.cuda.empty_cache()

        self._log(f"{Icons.CLEAR} Cache cleared - next run will reprocess all files")
        self._show_toast("Cache cleared. Next run will reprocess all files.", "success")

    # ---------------------------------------------------#
    # Method name: _build_data_page
    # ---------------------------------------------------#

    def _build_data_page(self):
        """Build the data settings page."""
        page = self._create_page_frame("data")

        # Dataset folder section
        folder_frame = ttk.LabelFrame(page, text=f"{Icons.FOLDER} Dataset Location", padding=20)
        folder_frame.pack(fill="x", pady=(0, 15))

        path_row = ttk.Frame(folder_frame)
        path_row.pack(fill="x")

        ttk.Entry(path_row, textvariable=self.folder_var).pack(side="left", fill="x", expand=True, padx=(0, 10))
        ttk.Button(path_row, text=f"{Icons.BROWSE} Browse", command=self._browse_folder, style="Accent.TButton").pack(side="left")

        # Scan settings
        scan_frame = ttk.LabelFrame(page, text=f"{Icons.SETTINGS} Scan Settings", padding=20)
        scan_frame.pack(fill="x", pady=(0, 15))

        # File limit row
        limit_row = ttk.Frame(scan_frame)
        limit_row.pack(fill="x", pady=(0, 10))

        ttk.Label(limit_row, text="Max files to use:").pack(side="left", padx=(0, 10))
        ttk.Entry(limit_row, textvariable=self.file_limit_var, width=10).pack(side="left")
        ttk.Label(limit_row, text="(0 = all)", style="Dim.TLabel").pack(side="left", padx=(5, 20))

        # Quick limit buttons
        ttk.Label(limit_row, text="Presets:").pack(side="left", padx=(10, 5))
        for val, label in [("1000", "1k"), ("10000", "10k"), ("50000", "50k"), ("0", "All")]:
            ttk.Button(limit_row, text=label, width=5, 
                       command=lambda v=val: self.file_limit_var.set(v)).pack(side="left", padx=2)

        # File size filter row
        size_row = ttk.Frame(scan_frame)
        size_row.pack(fill="x", pady=(10, 0))

        ttk.Label(size_row, text="File Size Filter:").pack(side="left", padx=(0, 10))
        ttk.Label(size_row, text="Min:", style="Dim.TLabel").pack(side="left")
        ttk.Entry(size_row, textvariable=self.min_file_size_var, width=8).pack(side="left", padx=5)
        
        ttk.Label(size_row, text="Max:", style="Dim.TLabel").pack(side="left", padx=(10, 0))
        ttk.Entry(size_row, textvariable=self.max_file_size_var, width=8).pack(side="left", padx=5)
        
        ttk.Combobox(size_row, textvariable=self.size_unit_var, values=["B", "KB", "MB"], width=5, state="readonly").pack(side="left", padx=5)

        # Quick size presets
        ttk.Label(size_row, text="Presets:", style="Dim.TLabel").pack(side="left", padx=(20, 5))
        for label, min_v, max_v, unit in [("Small (<10KB)", "0", "10", "KB"), ("Med (10-100KB)", "10", "100", "KB"), 
                                          ("Large (>100KB)", "100", "0", "KB")]:
            ttk.Button(size_row, text=label, width=12,
                       command=lambda m=min_v, x=max_v, u=unit: self._set_size_filter(m, x, u)).pack(side="left", padx=2)

        # Shuffle options
        shuffle_row = ttk.Frame(scan_frame)
        shuffle_row.pack(fill="x", pady=(15, 0))

        ttk.Checkbutton(shuffle_row, text="Shuffle files before selection", variable=self.random_seed_var).pack(side="left")
        
        ttk.Label(shuffle_row, text="Seed:").pack(side="left", padx=(20, 5))
        ttk.Entry(shuffle_row, textvariable=self.seed_value_var, width=8).pack(side="left")

        self.effective_files_label = ttk.Label(shuffle_row, text="", style="Accent.TLabel")
        self.effective_files_label.pack(side="left", padx=(30, 0))

        # Actions
        action_frame = ttk.Frame(page)
        action_frame.pack(fill="x", pady=20)

        # Scan button
        self.scan_btn = ttk.Button(action_frame, text=f"{Icons.SCAN} Scan Files", style="Accent.TButton", command=self._scan_folder)
        self.scan_btn.pack(side="left", padx=(0, 10))

        # Clear cache
        ttk.Button(action_frame, text=f"{Icons.CLEAR} Clear Cache", style="Danger.TButton", 
                   command=self._clear_cache).pack(side="right")
                   
        # Status/Estimates
        est_frame = ttk.LabelFrame(page, text=f"{Icons.INFO} Dataset Estimates", padding=20)
        est_frame.pack(fill="x")
        
        self.dataset_estimate_label = ttk.Label(est_frame, text="Scan to see estimates", style="Accent.TLabel")
        self.dataset_estimate_label.pack(anchor="w")

        # Scan Progress (Restored)
        self.scan_progress_frame = ttk.Frame(page)
        self.scan_progress_frame.pack(fill="x", pady=10)
        self.scan_progress_label = ttk.Label(self.scan_progress_frame, text="Ready", style="Dim.TLabel")
        self.scan_progress_label.pack(anchor="w")
        self.scan_progress_bar = ColoredProgressBar(self.scan_progress_frame, width=400, height=20, bg=NavyTheme.NAVY_DARKEST)
        self.scan_progress_bar.pack(fill="x", pady=(5, 0))

        # Bind updates
        for var in [self.file_limit_var, self.min_file_size_var, self.max_file_size_var, self.size_unit_var]:
            var.trace_add("write", lambda *args: self._update_effective_files())

    # Helper method for collapsible section
    def _create_collapsible_section(self, parent, title, expanded=True):
        section = CollapsibleSection(parent, title, expanded)
        return section, section.content

    # ---------------------------------------------------#
    # Method name: _set_file_limit
    # ---------------------------------------------------#
    def _set_file_limit(self, value: str):
        """Set file limit from preset button."""
        self.file_limit_var.set(value)

    # ---------------------------------------------------#
    # Method name: _set_size_filter
    # ---------------------------------------------------#
    def _set_size_filter(self, min_val: str, max_val: str, unit: str):
        """Set size filter from preset."""
        self.min_file_size_var.set(min_val)
        self.max_file_size_var.set(max_val)
        self.size_unit_var.set(unit)

    # ---------------------------------------------------#
    # Method name: _get_size_in_bytes
    # ---------------------------------------------------#
    def _get_size_in_bytes(self, value: str, unit: str) -> int:
        """Convert size value to bytes."""
        try:
            val = float(value)
        except ValueError:
            return 0

        multipliers = {"B": 1, "KB": 1024, "MB": 1024 * 1024}
        return int(val * multipliers.get(unit, 1))

    # ---------------------------------------------------#
    # Method name: _get_filtered_files
    # ---------------------------------------------------#
    def _get_filtered_files(self) -> List[Dict]:
        """Get files filtered by size and count limits."""
        if not self.processor.files:
            return []

        files = self.processor.files

        # Apply size filter
        unit = self.size_unit_var.get()
        min_size = self._get_size_in_bytes(self.min_file_size_var.get(), unit)
        max_size = self._get_size_in_bytes(self.max_file_size_var.get(), unit)

        if min_size > 0 or max_size > 0:
            filtered = []
            for f in files:
                size = f['size']
                if min_size > 0 and size < min_size:
                    continue
                if max_size > 0 and size > max_size:
                    continue
                filtered.append(f)
            files = filtered

        # Apply shuffle
        if self.random_seed_var.get():
            import random
            try:
                seed = int(self.seed_value_var.get())
            except ValueError:
                seed = 42
            files = files.copy()
            random.seed(seed)
            random.shuffle(files)

        # Apply count limit
        try:
            limit = int(self.file_limit_var.get())
        except ValueError:
            limit = 0

        if limit > 0:
            files = files[:limit]

        return files

    # ---------------------------------------------------#
    # Method name: _update_effective_files
    # ---------------------------------------------------#
    def _update_effective_files(self):
        """Update the effective files label."""
        if not self.processor.files:
            self.effective_files_label.config(text="")
            return

        total = len(self.processor.files)
        filtered = self._get_filtered_files()
        count = len(filtered)

        if count == total:
            self.effective_files_label.config(text=f"→ Using all {total:,} files")
        else:
            self.effective_files_label.config(text=f"→ Using {count:,} of {total:,} files")

        self._update_dataset_estimate()

    # ---------------------------------------------------#
    # Method name: _update_dataset_estimate
    # ---------------------------------------------------#
    def _update_dataset_estimate(self):
        """Update dataset size estimate."""
        if not self.processor.files:
            return

        filtered = self._get_filtered_files()
        effective_files = len(filtered)

        if effective_files == 0:
            self.dataset_estimate_label.config(
                text=f"{Icons.WARNING} No files match current filters"
            )
            return

        # Calculate actual size of filtered files
        total_size = sum(f['size'] for f in filtered)

        # Estimate tokens (~0.3 tokens per byte for Python code)
        estimated_tokens = int(total_size * 0.3)

        try:
            ctx = int(self.max_seq_var.get())
            stride = int(self.stride_var.get())
            estimated_samples = max(1, (estimated_tokens - ctx) // stride)
        except ValueError:
            estimated_samples = 0

        # Format tokens
        if estimated_tokens >= 1e9:
            token_str = f"{estimated_tokens / 1e9:.1f}B"
        elif estimated_tokens >= 1e6:
            token_str = f"{estimated_tokens / 1e6:.1f}M"
        else:
            token_str = f"{estimated_tokens / 1e3:.0f}k"

        # Format size
        if total_size >= 1e9:
            size_str = f"{total_size / 1e9:.1f}GB"
        elif total_size >= 1e6:
            size_str = f"{total_size / 1e6:.1f}MB"
        else:
            size_str = f"{total_size / 1e3:.0f}KB"

        self.dataset_estimate_label.config(
            text=f"{Icons.FILE} Files: {effective_files:,} ({size_str}) | "
                 f"{Icons.CODE} Est. tokens: ~{token_str} | "
                 f"{Icons.DATA} Est. samples: ~{estimated_samples:,}"
        )



    # ---------------------------------------------------#
    # Method name: _set_step_status
    # ---------------------------------------------------#

    def _set_step_status(self, step_id: str, status: str):
        """Set step status: 'pending', 'running', 'done', 'error'."""
        if step_id not in self.step_labels:
            return

        def _update():
            labels = self.step_labels[step_id]

            if status == "pending":
                labels["status"].config(text="○", foreground=NavyTheme.TEXT_DIM)
                labels["name"].config(foreground=NavyTheme.TEXT_DIM)
                labels["time"].config(text="")
                labels["start_time"] = None

            elif status == "running":
                labels["status"].config(text="◉", foreground=NavyTheme.WARNING)
                labels["name"].config(foreground=NavyTheme.TEXT_PRIMARY)
                labels["time"].config(text="...")
                labels["start_time"] = time.time()

            elif status == "done":
                labels["status"].config(text="✓", foreground=NavyTheme.SUCCESS)
                labels["name"].config(foreground=NavyTheme.SUCCESS)
                if labels["start_time"]:
                    elapsed = time.time() - labels["start_time"]
                    if elapsed < 60:
                        labels["time"].config(text=f"{elapsed:.1f}s")
                    else:
                        labels["time"].config(text=f"{elapsed / 60:.1f}m")

            elif status == "error":
                labels["status"].config(text="✗", foreground=NavyTheme.ERROR)
                labels["name"].config(foreground=NavyTheme.ERROR)

        self.gui_queue.put(_update)

    def _reset_all_steps(self):
        """Reset all steps to pending."""
        for step_id in self.step_labels:
            self._set_step_status(step_id, "pending")



    def _build_refinement_page(self):
        """Build the dataset refinement page with full filtering options."""
        p = self._create_page_frame("refine")

        # =====================================================================
        # 1. SCAN OPTIONS (Dataset Type & Recursive)
        # =====================================================================
        type_frame = ttk.LabelFrame(p, text=f"{Icons.SCAN} Scan Options", padding=10)
        type_frame.pack(fill="x", pady=5)

        self.dataset_type_var = tk.StringVar(value="code")
        self.recursive_scan_var = tk.BooleanVar(value=True)

        row = ttk.Frame(type_frame)
        row.pack(fill="x")

        ttk.Radiobutton(row, text="Source Code", variable=self.dataset_type_var, value="code").pack(side="left", padx=10)
        ttk.Radiobutton(row, text="Text/Books", variable=self.dataset_type_var, value="text").pack(side="left", padx=10)
        ttk.Checkbutton(row, text="Recursive Scan (Subfolders)", variable=self.recursive_scan_var).pack(side="right", padx=10)

        # =====================================================================
        # 2. FILE SIZE FILTER
        # =====================================================================
        size_frame = ttk.LabelFrame(p, text="📏 File Size Filter", padding=10)
        size_frame.pack(fill="x", pady=5)

        size_row = ttk.Frame(size_frame)
        size_row.pack(fill="x")

        ttk.Label(size_row, text="Min Size:").pack(side="left", padx=(0, 5))
        ttk.Entry(size_row, textvariable=self.refine_min_size_var, width=8).pack(side="left")
        
        ttk.Label(size_row, text="Max Size:").pack(side="left", padx=(20, 5))
        ttk.Entry(size_row, textvariable=self.refine_max_size_var, width=8).pack(side="left")

        unit_combo = ttk.Combobox(size_row, textvariable=self.refine_size_unit_var, values=["B", "KB", "MB"], width=4, state="readonly")
        unit_combo.pack(side="left", padx=5)

        ttk.Label(size_row, text="(0 = no limit)", style="Dim.TLabel").pack(side="left", padx=10)

        # Quick size presets
        preset_row = ttk.Frame(size_frame)
        preset_row.pack(fill="x", pady=(5, 0))
        ttk.Label(preset_row, text="Quick:").pack(side="left", padx=(0, 5))
        
        size_presets = [("<1KB", "0", "1", "KB"), ("1-10KB", "1", "10", "KB"), 
                        ("10-100KB", "10", "100", "KB"), ("100KB-1MB", "100", "1000", "KB"), 
                        (">1MB", "1000", "0", "KB"), ("Any", "0", "0", "KB")]
        for label, min_v, max_v, unit in size_presets:
            ttk.Button(preset_row, text=label, width=8,
                       command=lambda m=min_v, x=max_v, u=unit: (
                           self.refine_min_size_var.set(m),
                           self.refine_max_size_var.set(x),
                           self.refine_size_unit_var.set(u)
                       )).pack(side="left", padx=2)

        # =====================================================================
        # 3. QUALITY FILTERING
        # =====================================================================
        quality_frame = ttk.LabelFrame(p, text="🧹 Quality Filtering", padding=10)
        quality_frame.pack(fill="x", pady=5)

        qual_row = ttk.Frame(quality_frame)
        qual_row.pack(fill="x")

        ttk.Label(qual_row, text="Min Quality Score (0-100):").pack(side="left", padx=(0, 5))
        qual_spin = ttk.Spinbox(qual_row, from_=0, to=100, textvariable=self.refine_min_quality_var, width=5)
        qual_spin.pack(side="left")
        ttk.Label(qual_row, text="Files below this score are flagged", style="Dim.TLabel").pack(side="left", padx=15)

        # =====================================================================
        # 4. DEDUPLICATION
        # =====================================================================
        dedup_frame = ttk.LabelFrame(p, text="🔄 Deduplication", padding=10)
        dedup_frame.pack(fill="x", pady=5)

        dedup_row1 = ttk.Frame(dedup_frame)
        dedup_row1.pack(fill="x")

        ttk.Checkbutton(dedup_row1, text="Exact Dedup (SHA256)", variable=self.refine_exact_dedup_var).pack(side="left", padx=10)
        ttk.Checkbutton(dedup_row1, text="Near-Duplicate Detection", variable=self.refine_near_dedup_var).pack(side="left", padx=10)

        dedup_row2 = ttk.Frame(dedup_frame)
        dedup_row2.pack(fill="x", pady=(5, 0))

        ttk.Label(dedup_row2, text="Near-Dup Threshold:").pack(side="left", padx=(0, 5))
        ttk.Entry(dedup_row2, textvariable=self.refine_near_threshold_var, width=6).pack(side="left")
        ttk.Label(dedup_row2, text="(0.50-0.99, higher=stricter)", style="Dim.TLabel").pack(side="left", padx=5)

        ttk.Label(dedup_row2, text="Fingerprint Size:").pack(side="left", padx=(20, 5))
        ttk.Entry(dedup_row2, textvariable=self.refine_fingerprint_size_var, width=6).pack(side="left")

        # =====================================================================
        # 5. TEXT CLEANING MODE
        # =====================================================================
        clean_frame = ttk.LabelFrame(p, text="🧽 Text Cleaning Mode", padding=10)
        clean_frame.pack(fill="x", pady=5)

        clean_row = ttk.Frame(clean_frame)
        clean_row.pack(fill="x")

        cleaning_modes = [("Auto", "auto"), ("Fiction/Books", "fiction"), ("Code", "code"), ("Docs/Markdown", "docs"), ("None", "none")]
        for label, value in cleaning_modes:
            ttk.Radiobutton(clean_row, text=label, variable=self.refine_cleaning_mode_var, value=value).pack(side="left", padx=10)

        # =====================================================================
        # 6. ACTIONS & PROGRESS
        # =====================================================================
        action_frame = ttk.LabelFrame(p, text="▶ Actions", padding=10)
        action_frame.pack(fill="x", pady=5)

        btn_row = ttk.Frame(action_frame)
        btn_row.pack(fill="x", pady=5)

        ttk.Button(btn_row, text=f"{Icons.SCAN} Analyze Dataset", style="Accent.TButton",
                   command=self._run_refinement_analysis).pack(side="left", padx=5)
        
        self.refine_stop_btn = ttk.Button(btn_row, text=f"{Icons.STOP} Stop", command=self._stop_refinement)
        self.refine_stop_btn.pack(side="left", padx=5)

        self.refine_status = ttk.Label(btn_row, text="Ready", style="Dim.TLabel")
        self.refine_status.pack(side="left", padx=10)

        # Progress bar
        # Progress bar
        self.refine_progress = ColoredProgressBar(action_frame, width=400, height=20)
        self.refine_progress.pack(fill="x", pady=5)

        # Stats row
        self.refine_stats = ttk.Label(action_frame, text="Files: 0 scanned | 0 issues found", style="Dim.TLabel")
        self.refine_stats.pack(anchor="w")

        # =====================================================================
        # 7. RESULTS TREEVIEW
        # =====================================================================
        results_frame = ttk.LabelFrame(p, text="📋 Analysis Results", padding=10)
        results_frame.pack(fill="both", expand=True, pady=5)

        tree_container = ttk.Frame(results_frame)
        tree_container.pack(fill="both", expand=True)

        columns = ("Type", "File", "Score", "Reason")
        self.refine_tree = ttk.Treeview(tree_container, columns=columns, show="headings", selectmode="extended")
        self.refine_tree.heading("Type", text="Issue")
        self.refine_tree.heading("File", text="File Path")
        self.refine_tree.heading("Score", text="Quality")
        self.refine_tree.heading("Reason", text="Details")

        self.refine_tree.column("Type", width=80, stretch=False)
        self.refine_tree.column("File", width=350)
        self.refine_tree.column("Score", width=60, stretch=False)
        self.refine_tree.column("Reason", width=200)

        vsb = ttk.Scrollbar(tree_container, orient="vertical", command=self.refine_tree.yview)
        hsb = ttk.Scrollbar(tree_container, orient="horizontal", command=self.refine_tree.xview)
        self.refine_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.refine_tree.pack(fill="both", expand=True)

        # Bottom action buttons
        bottom_row = ttk.Frame(results_frame)
        bottom_row.pack(fill="x", pady=(5, 0))

        ttk.Button(bottom_row, text="Move Selected to '_trash'", command=self._move_garbage_files).pack(side="right")
        ttk.Button(bottom_row, text="Select All", 
                   command=lambda: self.refine_tree.selection_set(self.refine_tree.get_children())).pack(side="right", padx=5)

    def _stop_refinement(self):
        """Stop the refinement analysis."""
        self.refine_stop_flag = True
        self.refine_status.config(text="Stopping...")
        self._log(f"{Icons.STOP} Refinement analysis stopped by user")

    def _run_refinement_analysis(self):
        """Start the refinement analysis in a background thread."""
        folder = self.folder_var.get()
        if not folder:
            messagebox.showwarning("Warning", "Select a Source Data folder first.")
            return

        # Reset stop flag
        self.refine_stop_flag = False

        # Clear tree
        for item in self.refine_tree.get_children():
            self.refine_tree.delete(item)

        # Reset progress
        self.refine_progress.set(0)
        self.refine_stats.config(text="Files: 0 scanned | 0 issues found")
        self.refine_status.config(text="Analyzing...")
        self.root.update()

        # Gather settings
        settings = {
            "dtype": self.dataset_type_var.get(),
            "recursive": self.recursive_scan_var.get(),
            "min_size": self._parse_size(self.refine_min_size_var.get(), self.refine_size_unit_var.get()),
            "max_size": self._parse_size(self.refine_max_size_var.get(), self.refine_size_unit_var.get()),
            "min_quality": int(self.refine_min_quality_var.get() or 0),
            "exact_dedup": self.refine_exact_dedup_var.get(),
            "near_dedup": self.refine_near_dedup_var.get(),
            "near_threshold": float(self.refine_near_threshold_var.get() or 0.90),
            "fingerprint_size": int(self.refine_fingerprint_size_var.get() or 200),
            "cleaning_mode": self.refine_cleaning_mode_var.get(),
        }

        # Run analysis in thread
        threading.Thread(target=self._refinement_worker, args=(folder, settings), daemon=True).start()

    def _parse_size(self, value_str: str, unit: str) -> int:
        """Parse size string to bytes. Returns 0 for no limit."""
        try:
            val = float(value_str or 0)
            if val <= 0:
                return 0
            multipliers = {"B": 1, "KB": 1024, "MB": 1024 * 1024}
            return int(val * multipliers.get(unit, 1024))
        except ValueError:
            return 0

    def _refinement_worker(self, folder: str, settings: dict):
        """Worker thread for refinement analysis."""
        issues = []  # List of (type, path, score, reason)
        path = Path(folder)

        # Get file list
        if settings["recursive"]:
            all_files = list(path.rglob("*"))
        else:
            all_files = list(path.glob("*"))
        
        # Filter to files only
        all_files = [f for f in all_files if f.is_file()]
        total = len(all_files)
        
        if total == 0:
            self.gui_queue.put(lambda: self.refine_status.config(text="No files found"))
            return

        # File extensions filter based on type
        valid_extensions = set()
        if settings["dtype"] == "code":
            valid_extensions = {".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".rb", ".php"}
        else:  # text
            valid_extensions = {".txt", ".md", ".rst", ".html", ".htm", ".xml", ".json"}

        # Data structures for dedup
        exact_hashes = {}  # hash -> first file path
        fingerprints = []  # List of (path, fingerprint_set, quality_score)

        # Garbage patterns
        garbage_patterns = []
        if settings["dtype"] == "code":
            garbage_patterns = [
                (b"generated by", "Auto-generated"),
                (b"DO NOT EDIT", "Auto-generated"),
                (b".min.js", "Minified"),
                (b".min.css", "Minified"),
                (b"node_modules", "Library"),
                (b"site-packages", "Library"),
                (b"__pycache__", "Cache"),
                (b"\x00", "Binary"),
            ]
        else:  # text
            garbage_patterns = [
                (b"Project Gutenberg", "Gutenberg boilerplate"),
                (b"End of the Project Gutenberg", "Gutenberg footer"),
            ]

        scanned = 0
        kept = 0

        for i, fpath in enumerate(all_files):
            # Check stop flag
            if self.refine_stop_flag:
                def update_stopped():
                    self.refine_status.config(text=f"Stopped at {i}/{total} files")
                    self.refine_progress.set((i / total) * 100)
                self.gui_queue.put(update_stopped)
                return

            # Progress update (every 50 files or at boundaries)
            if i % 50 == 0 or i == total - 1:
                progress_pct = ((i + 1) / total) * 100
                def update_progress(pct=progress_pct, idx=i, tot=total, iss=len(issues)):
                    self.refine_progress.set(pct)
                    self.refine_status.config(text=f"Scanning {idx + 1}/{tot}...")
                    self.refine_stats.config(text=f"Files: {idx + 1} scanned | {iss} issues found")
                self.gui_queue.put(update_progress)

            try:
                # 1. Extension filter
                ext = fpath.suffix.lower()
                if valid_extensions and ext not in valid_extensions:
                    continue  # Skip silently, not an issue

                scanned += 1

                # 2. Size filter
                file_size = fpath.stat().st_size
                min_size = settings["min_size"]
                max_size = settings["max_size"]

                if file_size < 100:  # Always flag tiny files
                    issues.append(("Junk", str(fpath), "-", f"Too small ({file_size}B)"))
                    continue

                if min_size > 0 and file_size < min_size:
                    issues.append(("Size", str(fpath), "-", f"Below min ({file_size}B < {min_size}B)"))
                    continue

                if max_size > 0 and file_size > max_size:
                    issues.append(("Size", str(fpath), "-", f"Above max ({file_size}B > {max_size}B)"))
                    continue

                # 3. Read file content
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                except Exception:
                    # Binary or unreadable
                    issues.append(("Binary", str(fpath), "-", "Cannot read as text"))
                    continue

                # 4. Garbage pattern check (on raw content)
                raw_bytes = content[:2048].encode('utf-8', errors='replace')
                path_bytes = str(fpath).encode('utf-8', errors='replace')
                is_garbage = False
                for pattern, reason in garbage_patterns:
                    if pattern in raw_bytes or pattern in path_bytes:
                        issues.append(("Garbage", str(fpath), "-", reason))
                        is_garbage = True
                        break
                if is_garbage:
                    continue

                # 5. Apply cleaning (if not 'none')
                cleaning_mode = settings["cleaning_mode"]
                if cleaning_mode != "none":
                    content = clean_by_mode(content, cleaning_mode, ext)

                # 6. Quality scoring
                is_code = settings["dtype"] == "code"
                metrics = compute_quality_metrics(content, is_code)
                score, reasons = compute_quality_score(metrics, min_chars=100)

                if score < settings["min_quality"]:
                    reason_str = "; ".join(reasons[:2]) if reasons else "Low quality"
                    issues.append(("Quality", str(fpath), str(score), reason_str))
                    continue

                # 7. Exact dedup (SHA256)
                if settings["exact_dedup"]:
                    content_hash = sha256_hash(content)
                    if content_hash in exact_hashes:
                        issues.append(("Duplicate", str(fpath), str(score), f"Exact match: {exact_hashes[content_hash]}"))
                        continue
                    exact_hashes[content_hash] = fpath.name

                # 8. Near-dedup (fingerprint)
                if settings["near_dedup"]:
                    fp = build_fingerprint(content, settings["fingerprint_size"], use_lines=is_code)
                    
                    # Check against existing fingerprints
                    is_near_dup = False
                    for other_path, other_fp, other_score in fingerprints:
                        similarity = jaccard_similarity(fp, other_fp)
                        if similarity >= settings["near_threshold"]:
                            # Keep the one with higher quality
                            if score <= other_score:
                                issues.append(("NearDup", str(fpath), str(score), 
                                             f"{similarity:.0%} similar to {other_path}"))
                                is_near_dup = True
                                break
                    
                    if is_near_dup:
                        continue
                    
                    # Add to fingerprints list
                    fingerprints.append((fpath.name, fp, score))

                # File passed all checks
                kept += 1

            except Exception as e:
                issues.append(("Error", str(fpath), "-", str(e)[:50]))

        # Final UI update
        def show_results():
            for type_, fpath, score, reason in issues:
                self.refine_tree.insert("", "end", values=(type_, fpath, score, reason))
            self.refine_progress.set(100)
            self.refine_status.config(text=f"Complete: {len(issues)} issues in {total} files")
            self.refine_stats.config(text=f"Files: {scanned} scanned | {len(issues)} issues | {kept} OK")

        self.gui_queue.put(show_results)

    def _move_garbage_files(self):
        """Move selected files to _trash folder with unique naming."""
        selected = self.refine_tree.selection()
        if not selected:
            return

        folder = self.folder_var.get()
        trash_dir = os.path.join(folder, "_trash")
        os.makedirs(trash_dir, exist_ok=True)

        count = 0
        import shutil
        
        for item in selected:
            vals = self.refine_tree.item(item)['values']
            fpath = vals[1]
            try:
                fname = os.path.basename(fpath)
                dest = os.path.join(trash_dir, fname)
                
                # Handle duplicate names in trash
                if os.path.exists(dest):
                    base, ext = os.path.splitext(fname)
                    counter = 1
                    while os.path.exists(dest):
                        dest = os.path.join(trash_dir, f"{base}_{counter}{ext}")
                        counter += 1
                
                shutil.move(fpath, dest)
                self.refine_tree.delete(item)
                count += 1
            except Exception as e:
                self._log(f"{Icons.WARNING} Failed to move {fpath}: {e}")

        self._show_toast(f"Moved {count} files to '_trash' folder", "success")


    # ---------------------------------------------------#
    # Method name: _build_model_page
    # ---------------------------------------------------#
    def _build_model_page(self):
        """Build the model architecture page."""
        page = self._create_page_frame("model")

        # Presets section
        preset_frame = ttk.LabelFrame(page, text=f"{Icons.SPARKLE} Presets", padding=15)
        preset_frame.pack(fill="x", pady=(0, 15))

        preset_desc = ttk.Label(
            preset_frame,
            text="Select a preset to auto-configure all settings, or use Auto-Select for optimal configuration.",
            style="Dim.TLabel"
        )
        preset_desc.pack(anchor="w", pady=(0, 10))

        # Preset buttons row 1 - GPU based
        gpu_row = ttk.Frame(preset_frame)
        gpu_row.pack(fill="x", pady=(0, 10))

        ttk.Label(gpu_row, text="By GPU:", style="Dim.TLabel").pack(side="left", padx=(0, 10))

        for preset_id, label in [("gpu_8gb", "8GB"), ("gpu_12gb", "12GB"),
                                 ("gpu_16gb", "16GB"), ("gpu_24gb", "24GB+")]:
            btn = ttk.Button(gpu_row, text=label, width=8,
                             command=lambda p=preset_id: self._apply_preset(p))
            btn.pack(side="left", padx=(0, 5))
            ToolTip.create(btn, self._get_preset_config(preset_id)["description"])

        # Preset buttons row - Target Size (NEW)
        size_row = ttk.Frame(preset_frame)
        size_row.pack(fill="x", pady=(0, 10))

        ttk.Label(size_row, text="By Size:", style="Dim.TLabel").pack(side="left", padx=(0, 10))

        size_combo = ttk.Combobox(
            size_row, 
            values=["10M (Nano)", "50M (Micro)", "124M (Small)", "350M (Medium)", "774M (Large)", "1.5B (XL)"],
            state="readonly",
            width=15
        )
        size_combo.set("Select Size...")
        size_combo.pack(side="left", padx=(0, 10))
        
        ttk.Button(
            size_row, 
            text=f"{Icons.SPARKLE} Apply", 
            width=8,
            command=lambda: self._apply_target_size(size_combo.get())
        ).pack(side="left")

        # Preset buttons row 2 - Dataset based
        data_row = ttk.Frame(preset_frame)
        data_row.pack(fill="x", pady=(0, 10))

        ttk.Label(data_row, text="By Dataset:", style="Dim.TLabel").pack(side="left", padx=(0, 10))

        for preset_id, label in [("dataset_small", "<50k"), ("dataset_medium", "50-200k"),
                                 ("dataset_large", "200-500k"), ("dataset_xlarge", "500k+")]:
            btn = ttk.Button(data_row, text=label, width=10,
                             command=lambda p=preset_id: self._apply_preset(p))
            btn.pack(side="left", padx=(0, 5))
            ToolTip.create(btn, self._get_preset_config(preset_id)["description"])

        # Preset buttons row 3 - Quality based
        quality_row = ttk.Frame(preset_frame)
        quality_row.pack(fill="x")

        ttk.Label(quality_row, text="By Goal:", style="Dim.TLabel").pack(side="left", padx=(0, 10))

        for preset_id, label in [("fast_experiment", "Fast Test"), ("balanced", "Balanced"),
                                 ("high_quality", "High Quality")]:
            btn = ttk.Button(quality_row, text=label, width=12,
                             command=lambda p=preset_id: self._apply_preset(p))
            btn.pack(side="left", padx=(0, 5))
            ToolTip.create(btn, self._get_preset_config(preset_id)["description"])

        # Auto-select button
        auto_btn = ttk.Button(quality_row, text=f"{Icons.BOLT} Auto-Select", style="Accent.TButton",
                              command=self._auto_select_preset)
        auto_btn.pack(side="right", padx=(20, 0))
        ToolTip.create(auto_btn, "Automatically select optimal preset based on your GPU and scanned dataset size")

        # Architecture section
        arch_frame = ttk.LabelFrame(page, text=f"{Icons.MODEL} Architecture", padding=20)
        arch_frame.pack(fill="x", pady=(0, 15))

        # Grid of settings
        grid = ttk.Frame(arch_frame)
        grid.pack(fill="x")

        settings = [
            ("Embedding Dim", self.d_model_var, "d_model", "256, 512, 768, 1024"),
            ("Attention Heads", self.n_heads_var, "n_heads", "4, 8, 12, 16"),
            ("Layers", self.n_layers_var, "n_layers", "4, 6, 8, 12, 24"),
            ("FFN Dim", self.d_ff_var, "d_ff", "1024, 2048, 3072, 4096"),
            ("Vocab Size", self.vocab_size_var, "vocab_size", "16000, 32000, 50000"),
            ("Max Sequence", self.max_seq_var, "max_seq", "256, 512, 1024, 2048"),
            ("Dropout", self.dropout_var, "dropout", "0.0 - 0.3"),
        ]

        for i, (label, var, key, hint) in enumerate(settings):
            row, col = i // 2, (i % 2) * 3

            lbl = ttk.Label(grid, text=f"{label}:")
            lbl.grid(row=row, column=col, sticky="e", padx=(0, 10), pady=8)

            entry = ttk.Entry(grid, textvariable=var, width=15)
            entry.grid(row=row, column=col + 1, sticky="w", pady=8)
            ToolTip.create(entry, TOOLTIPS.get(key, f"Set {label.lower()}"))

            hint_lbl = ttk.Label(grid, text=hint, style="Dim.TLabel")
            hint_lbl.grid(row=row, column=col + 2, sticky="w", padx=(10, 30), pady=8)

        # Gradient Checkpointing Checkbox (ADDED)
        ttk.Checkbutton(arch_frame, text="Enable Gradient Checkpointing (Save VRAM)",
                        variable=self.grad_ckpt_var).pack(anchor="w", padx=5, pady=(10, 0))

        ttk.Label(arch_frame, text="⚠️ Allows training larger models but slightly slower.",
                  style="Dim.TLabel", font=("Segoe UI", 8)).pack(anchor="w", padx=25)

        # Gradient Accumulation
        ttk.Label(grid, text="Grad Accum:").grid(row=5, column=0, sticky="e", padx=(0, 10), pady=8)
        accum_combo = ttk.Combobox(grid, textvariable=self.grad_accum_var,
                                   values=["1", "2", "4", "8", "16", "32"], width=10)
        accum_combo.grid(row=5, column=1, sticky="w", pady=8)
        ToolTip.create(accum_combo,
                       "Gradient accumulation steps.\nEffective batch = batch_size × accum.\nHigher = less VRAM, same quality.")
        ttk.Label(grid, text="Effective batch = batch × accum", style="Dim.TLabel").grid(
            row=2, column=2, sticky="w", padx=(10, 30), pady=8)

        # Precision section
        prec_frame = ttk.LabelFrame(page, text=f"{Icons.BOLT} Precision", padding=20)
        prec_frame.pack(fill="x", pady=(0, 15))

        prec_desc = ttk.Label(
            prec_frame,
            text="Select training precision. BF16 recommended for modern GPUs (RTX 30xx+).",
            style="Dim.TLabel"
        )
        prec_desc.pack(anchor="w", pady=(0, 10))

        prec_row = ttk.Frame(prec_frame)
        prec_row.pack(fill="x")

        for val, text, desc in [
            ("bf16", "BF16", "Best for RTX 30xx+, A100"),
            ("fp16", "FP16", "Good for older GPUs"),
            ("fp32", "FP32", "Full precision, slowest")
        ]:
            rb = ttk.Radiobutton(prec_row, text=f"{text} - {desc}", variable=self.precision_var, value=val)
            rb.pack(side="left", padx=(0, 30))
            ToolTip.create(rb, TOOLTIPS['precision'])

        # Model size estimate
        size_frame = ttk.LabelFrame(page, text=f"{Icons.INFO} Model Size Estimate", padding=20)
        size_frame.pack(fill="x")

        self.model_size_label = ttk.Label(
            size_frame,
            text="Configure model settings to see size estimate",
            style="Accent.TLabel"
        )
        self.model_size_label.pack(anchor="w")

        # Bind updates
        for var in [self.d_model_var, self.n_heads_var, self.n_layers_var,
                    self.d_ff_var, self.vocab_size_var, self.max_seq_var]:
            var.trace_add("write", lambda *args: self._update_model_size_estimate())

    # ---------------------------------------------------#
    # Method name: _build_training_page
    # ---------------------------------------------------#
    # ---------------------------------------------------#
    # Method name: _build_training_page
    # ---------------------------------------------------#
    def _build_training_page(self):
        """Build the training configuration page."""
        page = self._create_page_frame("training")

        # Hyperparameters section
        hyper_frame = ttk.LabelFrame(page, text=f"{Icons.SETTINGS} Hyperparameters", padding=20)
        hyper_frame.pack(fill="x", pady=(0, 15))

        grid = ttk.Frame(hyper_frame)
        grid.pack(fill="x")

        # Learning rate with combobox
        ttk.Label(grid, text="Learning Rate:").grid(row=0, column=0, sticky="e", padx=(0, 10), pady=8)
        lr_combo = ttk.Combobox(grid, textvariable=self.lr_var, values=self.lr_options, width=12)
        lr_combo.grid(row=0, column=1, sticky="w", pady=8)
        ToolTip.create(lr_combo, TOOLTIPS['lr'])
        ttk.Label(grid, text="Start with 3e-4", style="Dim.TLabel").grid(row=0, column=2, sticky="w", padx=(10, 30),
                                                                         pady=8)

        settings = [
            ("Batch Size", self.batch_var, "batch_size", "16, 32, 64, 128"),
            ("Epochs", self.epochs_var, "epochs", "5, 10, 20, 50"),
            ("Stride", self.stride_var, "stride", "128, 256, 512"),
            ("Warmup Ratio", self.warmup_var, "warmup", "0.05 - 0.2"),
            ("Gradient Clip", self.grad_clip_var, "grad_clip", "0.5 - 2.0"),
            ("Val Split", self.val_split_var, "val_split", "0.05 - 0.2"),
        ]

        for i, (label, var, key, hint) in enumerate(settings):
            row = (i + 1) // 2
            col = ((i + 1) % 2) * 3

            ttk.Label(grid, text=f"{label}:").grid(row=row, column=col, sticky="e", padx=(0, 10), pady=8)
            entry = ttk.Entry(grid, textvariable=var, width=12)
            entry.grid(row=row, column=col + 1, sticky="w", pady=8)
            ToolTip.create(entry, TOOLTIPS.get(key, f"Set {label.lower()}"))
            ttk.Label(grid, text=hint, style="Dim.TLabel").grid(row=row, column=col + 2, sticky="w", padx=(10, 30),
                                                                pady=8)

        # Validation Settings
        val_frame = ttk.LabelFrame(page, text=f"{Icons.CHECK} Validation", padding=20)
        val_frame.pack(fill="x", pady=(0, 15))
        
        val_row = ttk.Frame(val_frame)
        val_row.pack(fill="x")
        
        val_check = ttk.Checkbutton(val_row, text="Enable Validation", variable=self.validation_enabled_var)
        val_check.pack(side="left", padx=(0, 20))
        ToolTip.create(val_check, "Calculate loss on validation set each epoch")
        
        ttk.Label(val_row, text="Limit batches:").pack(side="left", padx=(0, 10))
        
        val_options = ["0", "10", "20", "30", "50", "75", "80", "100"] 
        val_combo = ttk.Combobox(val_row, textvariable=self.validation_percent_var, values=val_options, width=5, state="readonly")
        val_combo.pack(side="left", padx=(0, 5))
        ToolTip.create(val_combo, "Percentage of validation data to check (Speed vs Accuracy)")
        
        ttk.Label(val_row, text="%", style="Dim.TLabel").pack(side="left")
        
        # Enable/Disable combo based on check
        def toggle_val_combo(*args):
            state = "readonly" if self.validation_enabled_var.get() else "disabled"
            val_combo.config(state=state)
            
        self.validation_enabled_var.trace_add("write", toggle_val_combo)
        toggle_val_combo() # Init state

        # Early stopping section
        es_frame = ttk.LabelFrame(page, text=f"{Icons.STOP} Early Stopping", padding=20)
        es_frame.pack(fill="x", pady=(0, 15))

        es_row = ttk.Frame(es_frame)
        es_row.pack(fill="x")

        es_check = ttk.Checkbutton(es_row, text="Enable early stopping", variable=self.early_stop_var)
        es_check.pack(side="left")
        ToolTip.create(es_check, TOOLTIPS['early_stopping'])

        ttk.Label(es_row, text="Patience:").pack(side="left", padx=(30, 10))
        patience_entry = ttk.Entry(es_row, textvariable=self.patience_var, width=6)
        patience_entry.pack(side="left")
        ToolTip.create(patience_entry, TOOLTIPS['patience'])

        ttk.Label(es_row, text="epochs without improvement", style="Dim.TLabel").pack(side="left", padx=(10, 0))

        # Training estimate
        estimate_frame = ttk.LabelFrame(page, text=f"{Icons.INFO} Training Estimate", padding=20)
        estimate_frame.pack(fill="x")

        self.training_estimate_label = ttk.Label(
            estimate_frame,
            text="Scan data and configure settings to see training estimate",
            style="Accent.TLabel"
        )
        self.training_estimate_label.pack(anchor="w")

    # ---------------------------------------------------#
    # Method name: _build_context_extend_page
    # ---------------------------------------------------#
    def _build_context_extend_page(self):
        """Build the context extension page for increasing model context length."""
        page = self._create_page_frame("context_extend")
        
        # Info section
        info_frame = ttk.LabelFrame(page, text=f"{Icons.INFO} Context Extension", padding=15)
        info_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(
            info_frame,
            text="Extend your trained model's context window. This loads a checkpoint,\n"
                 "expands position embeddings to the new size, and fine-tunes on longer sequences.\n"
                 "Original positions (0-N) keep learned weights; new positions are randomly initialized.",
            style="Dim.TLabel",
            justify="left"
        ).pack(anchor="w")
        
        # Source checkpoint section
        source_frame = ttk.LabelFrame(page, text=f"{Icons.LOAD} Source Checkpoint", padding=15)
        source_frame.pack(fill="x", pady=(0, 15))
        
        ckpt_row = ttk.Frame(source_frame)
        ckpt_row.pack(fill="x", pady=(0, 10))
        
        ttk.Label(ckpt_row, text="Checkpoint:").pack(side="left", padx=(0, 10))
        ttk.Entry(ckpt_row, textvariable=self.ctx_ext_checkpoint_var, width=50).pack(side="left", fill="x", expand=True, padx=(0, 10))
        ttk.Button(ckpt_row, text=f"{Icons.BROWSE}", width=4, command=self._browse_ctx_ext_checkpoint).pack(side="left")
        
        # Detected info row
        detect_row = ttk.Frame(source_frame)
        detect_row.pack(fill="x")
        
        ttk.Label(detect_row, text="Detected Context:").pack(side="left", padx=(0, 10))
        self.ctx_ext_detected_label = ttk.Label(detect_row, textvariable=self.ctx_ext_detected_context, style="Accent.TLabel")
        self.ctx_ext_detected_label.pack(side="left")
        
        # Target settings section
        target_frame = ttk.LabelFrame(page, text=f"{Icons.BOLT} Extension Settings", padding=15)
        target_frame.pack(fill="x", pady=(0, 15))
        
        # Grid of settings
        grid = ttk.Frame(target_frame)
        grid.pack(fill="x")
        
        # Row 1: Target context
        ttk.Label(grid, text="Target Context:").grid(row=0, column=0, sticky="e", padx=(0, 10), pady=8)
        target_combo = ttk.Combobox(grid, textvariable=self.ctx_ext_target_var, 
                                     values=["1024", "2048", "4096"], width=10, state="readonly")
        target_combo.grid(row=0, column=1, sticky="w", pady=8)
        ttk.Label(grid, text="tokens (new max sequence length)", style="Dim.TLabel").grid(row=0, column=2, sticky="w", padx=10)
        
        # Row 2: Batch size
        ttk.Label(grid, text="Batch Size:").grid(row=1, column=0, sticky="e", padx=(0, 10), pady=8)
        batch_combo = ttk.Combobox(grid, textvariable=self.ctx_ext_batch_var,
                                    values=["1", "2", "4", "8", "16"], width=10, state="readonly")
        batch_combo.grid(row=1, column=1, sticky="w", pady=8)
        ttk.Label(grid, text="(reduced for larger context)", style="Dim.TLabel").grid(row=1, column=2, sticky="w", padx=10)
        
        # Row 3: Gradient accumulation
        ttk.Label(grid, text="Grad Accumulation:").grid(row=2, column=0, sticky="e", padx=(0, 10), pady=8)
        accum_combo = ttk.Combobox(grid, textvariable=self.ctx_ext_accum_var,
                                    values=["4", "8", "16", "32", "64"], width=10, state="readonly")
        accum_combo.grid(row=2, column=1, sticky="w", pady=8)
        eff_batch = int(self.ctx_ext_batch_var.get()) * int(self.ctx_ext_accum_var.get())
        self.ctx_ext_eff_batch_label = ttk.Label(grid, text=f"Effective batch: {eff_batch}", style="Dim.TLabel")
        self.ctx_ext_eff_batch_label.grid(row=2, column=2, sticky="w", padx=10)
        
        # Row 4: Stride
        ttk.Label(grid, text="Stride:").grid(row=3, column=0, sticky="e", padx=(0, 10), pady=8)
        stride_combo = ttk.Combobox(grid, textvariable=self.ctx_ext_stride_var,
                                     values=["512", "1024", "2048"], width=10, state="readonly")
        stride_combo.grid(row=3, column=1, sticky="w", pady=8)
        ttk.Label(grid, text="(50% of target = good overlap)", style="Dim.TLabel").grid(row=3, column=2, sticky="w", padx=10)
        
        # Row 5: Epochs
        ttk.Label(grid, text="Fine-tune Epochs:").grid(row=4, column=0, sticky="e", padx=(0, 10), pady=8)
        epochs_entry = ttk.Entry(grid, textvariable=self.ctx_ext_epochs_var, width=10)
        epochs_entry.grid(row=4, column=1, sticky="w", pady=8)
        ttk.Label(grid, text="(2-5 recommended for context extension)", style="Dim.TLabel").grid(row=4, column=2, sticky="w", padx=10)
        
        # Row 6: Learning rate
        ttk.Label(grid, text="Learning Rate:").grid(row=5, column=0, sticky="e", padx=(0, 10), pady=8)
        lr_combo = ttk.Combobox(grid, textvariable=self.ctx_ext_lr_var,
                                 values=["0.0001", "0.00005", "0.00002", "0.00001"], width=10)
        lr_combo.grid(row=5, column=1, sticky="w", pady=8)
        ttk.Label(grid, text="(lower LR for fine-tuning)", style="Dim.TLabel").grid(row=5, column=2, sticky="w", padx=10)
        
        # Update effective batch on change
        def update_eff_batch(*args):
            try:
                eff = int(self.ctx_ext_batch_var.get()) * int(self.ctx_ext_accum_var.get())
                self.ctx_ext_eff_batch_label.configure(text=f"Effective batch: {eff}")
            except:
                pass
        self.ctx_ext_batch_var.trace_add("write", update_eff_batch)
        self.ctx_ext_accum_var.trace_add("write", update_eff_batch)
        
        # VRAM warning
        warn_frame = ttk.Frame(page)
        warn_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(
            warn_frame,
            text=f"{Icons.WARNING} VRAM Impact: 2048 context uses ~4x more VRAM than 512. "
                 "Gradient checkpointing is automatically enabled.",
            style="Warning.TLabel"
        ).pack(anchor="w")
        
        # Action buttons
        btn_frame = ttk.Frame(page)
        btn_frame.pack(fill="x")
        
        ttk.Button(
            btn_frame,
            text=f"{Icons.ROCKET} Start Context Extension Training",
            style="Accent.TButton",
            command=self._start_context_extension
        ).pack(side="left", padx=(0, 10))
        
        ttk.Button(
            btn_frame,
            text=f"{Icons.STOP} Stop",
            style="Danger.TButton",
            command=self._stop_training
        ).pack(side="left")

    # ---------------------------------------------------#
    # Method name: _browse_ctx_ext_checkpoint
    # ---------------------------------------------------#
    def _browse_ctx_ext_checkpoint(self):
        """Browse for source checkpoint for context extension."""
        path = filedialog.askopenfilename(
            title="Select Source Checkpoint",
            filetypes=[("PyTorch Checkpoint", "*.pt"), ("All Files", "*.*")]
        )
        if path:
            self.ctx_ext_checkpoint_var.set(path)
            # Detect context length from checkpoint
            try:
                ckpt = torch.load(path, map_location='cpu', weights_only=False)
                sd = ckpt.get('model_state_dict', ckpt.get('model', {}))
                pos_emb = sd.get('pos_emb.weight', None)
                if pos_emb is not None:
                    detected = pos_emb.shape[0]
                    self.ctx_ext_detected_context.set(f"{detected} tokens")
                    self._log(f"{Icons.SUCCESS} Detected context: {detected}")
                    
                    # Auto-suggest target
                    if detected <= 512:
                        self.ctx_ext_target_var.set("2048")
                        self.ctx_ext_stride_var.set("1024")
                    elif detected <= 1024:
                        self.ctx_ext_target_var.set("2048")
                        self.ctx_ext_stride_var.set("1024")
                    else:
                        self.ctx_ext_target_var.set("4096")
                        self.ctx_ext_stride_var.set("2048")
                else:
                    self.ctx_ext_detected_context.set("Unknown (no pos_emb found)")
                del ckpt
            except Exception as e:
                self.ctx_ext_detected_context.set(f"Error: {e}")
                self._log(f"{Icons.ERROR} Failed to analyze checkpoint: {e}")

    # ---------------------------------------------------#
    # Method name: _start_context_extension
    # ---------------------------------------------------#
    def _start_context_extension(self):
        """Start context extension training with position embedding surgery."""
        # Validate
        ckpt_path = self.ctx_ext_checkpoint_var.get()
        if not ckpt_path or not os.path.exists(ckpt_path):
            messagebox.showwarning("Warning", "Please select a valid source checkpoint.")
            return
        
        if not self.processor.files:
            messagebox.showwarning("Warning", "Please scan a data folder first (Data tab).")
            return
        
        # Apply context extension settings to training configs
        target_ctx = int(self.ctx_ext_target_var.get())
        
        # Override model config
        self.max_seq_var.set(str(target_ctx))
        
        # Override training config
        self.batch_var.set(self.ctx_ext_batch_var.get())
        self.stride_var.set(self.ctx_ext_stride_var.get())
        self.grad_accum_var.set(self.ctx_ext_accum_var.get())
        self.epochs_var.set(self.ctx_ext_epochs_var.get())
        self.lr_var.set(self.ctx_ext_lr_var.get())
        
        # Force gradient checkpointing
        self.grad_ckpt_var.set(True)
        
        # Set resume checkpoint
        self.resume_var.set(ckpt_path)
        
        # Flag for position embedding surgery
        self._context_extension_mode = True
        self._ctx_ext_reset_counters = True
        self._ctx_ext_source_checkpoint = ckpt_path
        
        self._log(f"{Icons.BOLT} Context Extension Mode: {self.ctx_ext_detected_context.get()} → {target_ctx}")
        self._log(f"   Batch: {self.ctx_ext_batch_var.get()}, Accum: {self.ctx_ext_accum_var.get()}, LR: {self.ctx_ext_lr_var.get()}")
        
        # Start training with context extension
        self._show_page("progress")
        self._start_training()

    # ---------------------------------------------------#
    # Method name: _build_checkpoint_page
    # ---------------------------------------------------#
    def _build_checkpoint_page(self):
        """Build the checkpoint settings page."""
        page = self._create_page_frame("checkpoint")

        # Output section
        out_frame = ttk.LabelFrame(page, text=f"{Icons.SAVE} Output Settings", padding=20)
        out_frame.pack(fill="x", pady=(0, 15))

        # Directory
        dir_row = ttk.Frame(out_frame)
        dir_row.pack(fill="x", pady=(0, 15))

        ttk.Label(dir_row, text="Directory:").pack(side="left", padx=(0, 10))
        dir_entry = ttk.Entry(dir_row, textvariable=self.output_var)
        dir_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ToolTip.create(dir_entry, TOOLTIPS['output_dir'])

        ttk.Button(dir_row, text=f"{Icons.BROWSE} Browse", command=self._browse_output).pack(side="left")

        # Name and save steps
        name_row = ttk.Frame(out_frame)
        name_row.pack(fill="x")

        ttk.Label(name_row, text="Checkpoint Name:").pack(side="left", padx=(0, 10))
        name_entry = ttk.Entry(name_row, textvariable=self.ckpt_name_var, width=20)
        name_entry.pack(side="left", padx=(0, 30))
        ToolTip.create(name_entry, TOOLTIPS['ckpt_name'])

        ttk.Label(name_row, text="Save every:").pack(side="left", padx=(0, 10))
        steps_entry = ttk.Entry(name_row, textvariable=self.save_steps_var, width=8)
        steps_entry.pack(side="left", padx=(0, 5))
        ToolTip.create(steps_entry, TOOLTIPS['save_steps'])
        ttk.Label(name_row, text="steps").pack(side="left")

        # Resume section
        resume_frame = ttk.LabelFrame(page, text=f"{Icons.LOAD} Resume Training", padding=20)
        resume_frame.pack(fill="x", pady=(0, 15))

        resume_row = ttk.Frame(resume_frame)
        resume_row.pack(fill="x", pady=(0, 15))

        resume_entry = ttk.Entry(resume_row, textvariable=self.resume_var)
        resume_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        ToolTip.create(resume_entry, TOOLTIPS['resume'])

        ttk.Button(resume_row, text=f"{Icons.BROWSE} Select", command=self._browse_resume).pack(side="left",
                                                                                                padx=(0, 5))
        ttk.Button(resume_row, text=f"{Icons.CLEAR} Clear", command=lambda: self.resume_var.set("")).pack(side="left")

        # Incremental training
        incr_check = ttk.Checkbutton(
            resume_frame,
            text=f"{Icons.REFRESH} Incremental Training (reuse tokenizer, add new data)",
            variable=self.incremental_var
        )
        incr_check.pack(anchor="w")
        ToolTip.create(incr_check, TOOLTIPS['incremental'])

        # Quick actions
        actions_frame = ttk.LabelFrame(page, text=f"{Icons.BOLT} Quick Actions", padding=20)
        actions_frame.pack(fill="x")

        btn_row = ttk.Frame(actions_frame)
        btn_row.pack(fill="x")

        # Note: Save/Load Settings are in the sidebar under "Project" section
        clear_vram_btn = ttk.Button(btn_row, text=f"{Icons.CLEAR} Clear VRAM", command=self._clear_vram)
        clear_vram_btn.pack(side="left",padx=(0, 10))
        ToolTip.create(clear_vram_btn, "Clear GPU memory cache")

        clear_cache_btn = ttk.Button(btn_row, text=f"{Icons.REFRESH} Clear Cache", command=self._clear_cache)
        clear_cache_btn.pack(side="left", padx=(0, 10))
        ToolTip.create(clear_cache_btn, "Clear cached texts and tokenizer.\nNext run will reprocess all files.")

    # ---------------------------------------------------#
    # Method name: _build_progress_page
    # ---------------------------------------------------#
    def _build_progress_page(self):
        """Build the training progress page."""
        page = self._create_page_frame("progress")

        # Top section - split into two columns
        top_frame = ttk.Frame(page)
        top_frame.pack(fill="x", pady=(0, 15))

        # Left column - Steps checklist
        steps_frame = ttk.LabelFrame(top_frame, text=f"{Icons.DATA} Pipeline Steps", padding=15)
        steps_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.step_labels = {}
        steps = [
            ("setup", "Setup & Configuration"),
            ("reading", "Reading Files"),
            ("tokenizer", "Training Tokenizer"),
            ("encoding", "Encoding Texts"),
            ("samples", "Creating Samples"),
            ("loaders", "Creating DataLoaders"),
            ("model", "Initializing Model"),
            ("training", "Training Model"),
            ("saving", "Saving Checkpoint"),
        ]

        for step_id, step_name in steps:
            frame = ttk.Frame(steps_frame)
            frame.pack(fill="x", pady=2)

            status_label = ttk.Label(frame, text="○", width=3, font=(NavyTheme.FONT_FAMILY, 12))
            status_label.pack(side="left")

            name_label = ttk.Label(frame, text=step_name, style="Dim.TLabel")
            name_label.pack(side="left")

            time_label = ttk.Label(frame, text="", style="Dim.TLabel")
            time_label.pack(side="right")

            self.step_labels[step_id] = {
                "status": status_label,
                "name": name_label,
                "time": time_label,
                "start_time": None
            }

        # Right column - Metrics Grid (2x2)
        metrics_frame = ttk.LabelFrame(top_frame, text=f"{Icons.PLOT} Live Metrics", padding=10)
        metrics_frame.pack(side="right", fill="both", expand=True)
        
        # Grid container
        m_grid = ttk.Frame(metrics_frame)
        m_grid.pack(fill="both", expand=True)
        
        # --- Quadrant 1: Status (Top-Left) ---
        q1 = ttk.LabelFrame(m_grid, text=f"{Icons.TARGET} STATUS", padding=5)
        q1.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.metric_lbl_epoch = ttk.Label(q1, text="Epoch:    - / -", font=(NavyTheme.FONT_MONO, 9))
        self.metric_lbl_epoch.pack(anchor="w")
        self.metric_lbl_step = ttk.Label(q1, text="Step:     - / -", font=(NavyTheme.FONT_MONO, 9))
        self.metric_lbl_step.pack(anchor="w")
        self.metric_lbl_prog_text = ttk.Label(q1, text="Progress: 0.0%", font=(NavyTheme.FONT_MONO, 9))
        self.metric_lbl_prog_text.pack(anchor="w")
        # Visual Progress Bar
        self.metric_prog_bar = ttk.Progressbar(q1, orient="horizontal", length=140, mode="determinate")
        self.metric_prog_bar.pack(anchor="w", pady=(2,0))
        
        # --- Quadrant 2: Losses (Top-Right) ---
        q2 = ttk.LabelFrame(m_grid, text=f"{Icons.CHART} LOSSES", padding=5)
        q2.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.metric_lbl_train_loss = ttk.Label(q2, text="Train:    -", font=(NavyTheme.FONT_MONO, 9))
        self.metric_lbl_train_loss.pack(anchor="w")
        self.metric_lbl_val_loss = ttk.Label(q2, text="Valid:    -", font=(NavyTheme.FONT_MONO, 9))
        self.metric_lbl_val_loss.pack(anchor="w")
        self.metric_lbl_best_loss = ttk.Label(q2, text="Best:     -", font=(NavyTheme.FONT_MONO, 9))
        self.metric_lbl_best_loss.pack(anchor="w")
        self.metric_lbl_ppl = ttk.Label(q2, text="Perplex:  -", font=(NavyTheme.FONT_MONO, 9))
        self.metric_lbl_ppl.pack(anchor="w")

        # --- Quadrant 3: Performance (Bottom-Left) ---
        q3 = ttk.LabelFrame(m_grid, text=f"{Icons.BOLT} PERFORMANCE", padding=5)
        q3.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.metric_lbl_speed = ttk.Label(q3, text="Speed:    -", font=(NavyTheme.FONT_MONO, 9))
        self.metric_lbl_speed.pack(anchor="w")
        self.metric_lbl_tokens = ttk.Label(q3, text="Tokens:   -", font=(NavyTheme.FONT_MONO, 9))
        self.metric_lbl_tokens.pack(anchor="w")
        self.metric_lbl_elapsed = ttk.Label(q3, text="Elapsed:  -", font=(NavyTheme.FONT_MONO, 9))
        self.metric_lbl_elapsed.pack(anchor="w")
        self.metric_lbl_eta = ttk.Label(q3, text="ETA:      -", font=(NavyTheme.FONT_MONO, 9))
        self.metric_lbl_eta.pack(anchor="w")

        # --- Quadrant 4: Resources (Bottom-Right) ---
        q4 = ttk.LabelFrame(m_grid, text=f"{Icons.GPU} RESOURCES", padding=5)
        q4.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.metric_lbl_vram = ttk.Label(q4, text="VRAM:     - / -", font=(NavyTheme.FONT_MONO, 9))
        self.metric_lbl_vram.pack(anchor="w")
        self.metric_lbl_usage = ttk.Label(q4, text="Usage:    -", font=(NavyTheme.FONT_MONO, 9))
        self.metric_lbl_usage.pack(anchor="w")
        self.metric_lbl_lr = ttk.Label(q4, text="LR:       -", font=(NavyTheme.FONT_MONO, 9))
        self.metric_lbl_lr.pack(anchor="w")
        
        # Expand weights
        m_grid.columnconfigure(0, weight=1)
        m_grid.columnconfigure(1, weight=1)
        m_grid.rowconfigure(0, weight=1)
        m_grid.rowconfigure(1, weight=1)

        # Progress section
        progress_frame = ttk.LabelFrame(page, text=f"{Icons.FIRE} Training Progress", padding=15)
        progress_frame.pack(fill="x", pady=(0, 15))

        # Status row
        status_row = ttk.Frame(progress_frame)
        status_row.pack(fill="x", pady=(0, 10))

        self.progress_status_label = ttk.Label(
            status_row,
            text=f"{Icons.INFO} Ready to train",
            font=(NavyTheme.FONT_FAMILY, 11, "bold"),
            foreground=NavyTheme.ACCENT_CYAN
        )
        self.progress_status_label.pack(side="left")

        self.progress_eta_label = ttk.Label(
            status_row,
            text="",
            font=(NavyTheme.FONT_FAMILY, 10),
            foreground=NavyTheme.TEXT_SECONDARY
        )
        self.progress_eta_label.pack(side="right")

        # Overall progress
        ttk.Label(progress_frame, text="Overall", style="Dim.TLabel").pack(anchor="w")
        self.overall_progress_bar = ColoredProgressBar(progress_frame, width=400, height=20)
        self.overall_progress_bar.pack(fill="x", pady=(5, 10))

        # Epoch progress
        ttk.Label(progress_frame, text="Current Epoch", style="Dim.TLabel").pack(anchor="w")
        self.epoch_progress_bar = ColoredProgressBar(progress_frame, width=400, height=20)
        self.epoch_progress_bar.pack(fill="x", pady=(5, 0))

        # Log section
        log_frame = ttk.LabelFrame(page, text=f"{Icons.FILE} Training Log", padding=15)
        log_frame.pack(fill="both", expand=True)

        log_container = ttk.Frame(log_frame)
        log_container.pack(fill="both", expand=True)

        log_scrollbar = ttk.Scrollbar(log_container)
        log_scrollbar.pack(side="right", fill="y")

        self.log_text = tk.Text(
            log_container,
            height=8,  # Reduced height as requested
            bg=NavyTheme.BG_INPUT,
            fg=NavyTheme.TEXT_SECONDARY,
            font=(NavyTheme.FONT_MONO, 9),
            state="disabled",
            relief="flat",
            padx=15,
            pady=10,
            yscrollcommand=log_scrollbar.set,
            insertbackground=NavyTheme.TEXT_PRIMARY
        )
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scrollbar.config(command=self.log_text.yview)

        self.log_text.tag_config("success", foreground=NavyTheme.SUCCESS)
        self.log_text.tag_config("error", foreground=NavyTheme.ERROR)
        self.log_text.tag_config("warning", foreground=NavyTheme.WARNING)
        self.log_text.tag_config("info", foreground=NavyTheme.INFO)
        self.log_text.tag_config("accent", foreground=NavyTheme.ACCENT_CYAN)

    # ---------------------------------------------------#
    # Method name: _build_dpo_page
    # ---------------------------------------------------#
    def _show_dpo_page(self):
        """Show DPO page."""
        self._show_page("dpo")

    def _build_dpo_page(self):
        """Build the DPO/RLHF training page with Tabs."""
        page = self._create_page_frame("dpo")
        
        notebook = ttk.Notebook(page)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        train_tab = ttk.Frame(notebook)
        notebook.add(train_tab, text=f"{Icons.SPARKLE} Train DPO")
        self._build_dpo_train_tab(train_tab)
        
        create_tab = ttk.Frame(notebook)
        notebook.add(create_tab, text=f"{Icons.ADD} Create Dataset")
        self._build_dpo_create_tab(create_tab)

    def _build_dpo_train_tab(self, page):
        """Build the Training Tab content."""
        
        # Instructions
        info_frame = ttk.LabelFrame(page, text=f"{Icons.INFO} DPO Training", padding=15)
        info_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(
            info_frame,
            text="Direct Preference Optimization (DPO) fine-tunes your model using preference pairs.\n"
                 "You provide examples of good vs bad outputs, and the model learns to prefer good ones.\n\n"
                 "Use dop_new.py to create preference pairs, then load the .dpo file here.",
            style="Dim.TLabel",
            wraplength=700
        ).pack(anchor="w")
        
        # Preference data section
        data_frame = ttk.LabelFrame(page, text=f"{Icons.DATA} Preference Data", padding=15)
        data_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        # File loading row
        file_row = ttk.Frame(data_frame)
        file_row.pack(fill="x", pady=(0, 10))
        
        ttk.Label(file_row, text="Preference File:").pack(side="left", padx=(0, 10))
        
        self.dpo_file_var = tk.StringVar()
        dpo_file_entry = ttk.Entry(file_row, textvariable=self.dpo_file_var, width=60)
        dpo_file_entry.pack(side="left", padx=(0, 10))
        ToolTip.create(dpo_file_entry, "Path to .dpo or .json file with preference pairs")
        
        ttk.Button(file_row, text=f"{Icons.BROWSE} Browse", command=self._browse_dpo_file).pack(side="left", padx=(0, 10))
        ttk.Button(file_row, text=f"{Icons.LOAD} Load", command=self._load_dpo_file).pack(side="left")
        
        # Stats row
        stats_row = ttk.Frame(data_frame)
        stats_row.pack(fill="x", pady=(0, 10))
        
        self.dpo_stats_label = ttk.Label(stats_row, text="No data loaded", style="Dim.TLabel")
        self.dpo_stats_label.pack(side="left")
        
        # Preview area
        ttk.Label(data_frame, text="Preview (first 5 pairs):").pack(anchor="w", pady=(10, 5))
        
        self.dpo_preview_text = scrolledtext.ScrolledText(
            data_frame,
            height=10,
            bg=NavyTheme.BG_INPUT,
            fg=NavyTheme.TEXT_PRIMARY,
            font=(NavyTheme.FONT_MONO, 9),
            insertbackground=NavyTheme.TEXT_PRIMARY,
            wrap="word"
        )
        self.dpo_preview_text.pack(fill="both", expand=True)
        
        # Settings section
        settings_frame = ttk.LabelFrame(page, text=f"{Icons.SETTINGS} DPO Settings", padding=15)
        settings_frame.pack(fill="x", pady=(0, 15))
        
        settings_row = ttk.Frame(settings_frame)
        settings_row.pack(fill="x")
        
        # Beta
        ttk.Label(settings_row, text="Beta (KL penalty):").pack(side="left", padx=(0, 10))
        self.dpo_beta_var = tk.StringVar(value="0.1")
        beta_entry = ttk.Entry(settings_row, textvariable=self.dpo_beta_var, width=8)
        beta_entry.pack(side="left", padx=(0, 30))
        ToolTip.create(beta_entry, "Higher = more conservative, stays closer to original model\nLower = more aggressive updates\nTypical: 0.1-0.5")
        
        # Epochs
        ttk.Label(settings_row, text="Epochs:").pack(side="left", padx=(0, 10))
        self.dpo_epochs_var = tk.StringVar(value="3")
        epochs_entry = ttk.Entry(settings_row, textvariable=self.dpo_epochs_var, width=8)
        epochs_entry.pack(side="left", padx=(0, 30))
        ToolTip.create(epochs_entry, "Number of passes through the preference data\nTypical: 1-5 epochs")
        
        # LR
        ttk.Label(settings_row, text="Learning Rate:").pack(side="left", padx=(0, 10))
        self.dpo_lr_var = tk.StringVar(value="0.00001")
        lr_entry = ttk.Entry(settings_row, textvariable=self.dpo_lr_var, width=12)
        lr_entry.pack(side="left")
        ToolTip.create(lr_entry, "Use very low LR for DPO\nTypical: 1e-5 to 5e-5")
        
        # Base model section
        model_frame = ttk.LabelFrame(page, text=f"{Icons.MODEL} Base Model", padding=15)
        model_frame.pack(fill="x", pady=(0, 15))
        
        model_row = ttk.Frame(model_frame)
        model_row.pack(fill="x")
        
        ttk.Label(model_row, text="Base Checkpoint:").pack(side="left", padx=(0, 10))
        
        self.dpo_model_var = tk.StringVar()
        model_entry = ttk.Entry(model_row, textvariable=self.dpo_model_var, width=60)
        model_entry.pack(side="left", padx=(0, 10))
        ToolTip.create(model_entry, "Path to the model checkpoint to fine-tune with DPO")
        
        ttk.Button(model_row, text=f"{Icons.BROWSE} Browse", command=self._browse_dpo_model).pack(side="left")
        
        self.dpo_model_status = ttk.Label(model_row, text="", style="Dim.TLabel")
        self.dpo_model_status.pack(side="left", padx=(20, 0))
        
        # Buttons
        btn_frame = ttk.Frame(page)
        btn_frame.pack(fill="x")
        
        ttk.Button(
            btn_frame,
            text=f"{Icons.PLAY} Start DPO Training",
            style="Accent.TButton",
            command=self._start_dpo_training
        ).pack(side="left", padx=(0, 10))
        
        ttk.Button(
            btn_frame,
            text=f"{Icons.STOP} Stop",
            style="Danger.TButton",
            command=self._stop_training
        ).pack(side="left")
        
        # Initialize DPO data storage
        self._dpo_preference_data = []
        
    def _build_dpo_create_tab(self, page):
        """Build the DPO Dataset Creation Tab."""
        # 1. Setup Bar
        setup = ttk.LabelFrame(page, text="Setup", padding=10)
        setup.pack(fill="x", padx=10, pady=5)
        
        # Session Restore & Export
        sess_row = ttk.Frame(setup)
        sess_row.pack(fill="x", pady=(0, 5))
        ttk.Button(sess_row, text=f"{Icons.REFRESH} Restore Last Session", command=self._dpo_restore_session, style="Accent.TButton").pack(side="left")
        ttk.Button(sess_row, text=f"{Icons.SAVE} Export Pairs", command=self._export_dpo_pairs).pack(side="right", padx=5)
        ttk.Button(sess_row, text=f"{Icons.SETTINGS} Save Settings", command=self._dpo_save_settings_manual).pack(side="right")
        
        # Grid Container for precise alignment
        grid_frame = ttk.Frame(setup)
        grid_frame.pack(fill="x", pady=5)
        
        # --- Row 0: Tokenizer & Model ---
        # Col 0: Label (pady increased)
        ttk.Label(grid_frame, text="Tokenizer:").grid(row=0, column=0, sticky="w", padx=5, pady=(5, 15))
        
        # Col 1: Status
        self.dpo_tok_status = ttk.Label(grid_frame, text="Not loaded", foreground=NavyTheme.ERROR, width=15)
        self.dpo_tok_status.grid(row=0, column=1, sticky="w", padx=5, pady=(5, 15))
        
        # Col 2: Button
        btn_tok = ttk.Button(grid_frame, text="Load Tokenizer", command=self._dpo_load_tokenizer)
        btn_tok.grid(row=0, column=2, sticky="w", padx=5, pady=(5, 15))
        ToolTip.create(btn_tok, "Load tokenizer configuration file")
        
        # Col 3: Separator (rowspan 2, but we need to respect the gap)
        ttk.Separator(grid_frame, orient="vertical").grid(row=0, column=3, rowspan=2, sticky="ns", padx=20, pady=5)
        
        # Col 4: Model Label
        ttk.Label(grid_frame, text="Model:").grid(row=0, column=4, sticky="w", padx=5, pady=(5, 15))
        
        # Col 5: Model Status
        self.dpo_model_status_lbl = ttk.Label(grid_frame, text="Not loaded", foreground=NavyTheme.ERROR, width=15)
        self.dpo_model_status_lbl.grid(row=0, column=5, sticky="w", padx=5, pady=(5, 15))
        
        # Col 6: Model Button
        btn_mod = ttk.Button(grid_frame, text="Load Model", command=self._dpo_load_creator_model)
        btn_mod.grid(row=0, column=6, sticky="w", padx=5, pady=(5, 15))
        ToolTip.create(btn_mod, "Load PyTorch model checkpoint (.pt)")
        
        # --- Row 1: Grader ---
        # Col 0: Label
        ttk.Label(grid_frame, text="Auto-Grader:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        
        # Col 1: Status
        self.dpo_grader_status = ttk.Label(grid_frame, text="Not loaded", foreground="gray", width=15)
        self.dpo_grader_status.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Col 2: Button (Aligned with Tokenizer Button)
        btn_grad_disk = ttk.Button(grid_frame, text="Load (Disk)", command=self._dpo_load_grader_local)
        btn_grad_disk.grid(row=1, column=2, sticky="ew", padx=5, pady=5)
        ToolTip.create(btn_grad_disk, "Load grader model from local disk")
        
        # Col 6: Button (Aligned with Model Button)
        btn_grad_hf = ttk.Button(grid_frame, text="Load (HF)", command=self._dpo_load_grader_hf)
        
        # Add weights to push col 8 to right
        grid_frame.columnconfigure(7, weight=1)
        btn_grad_hf.grid(row=1, column=8, sticky="e", padx=5, pady=5)
        ToolTip.create(btn_grad_hf, "Download/Load grader from HuggingFace")
        
        # 2a. Mode Selection
        mode_frame = ttk.LabelFrame(page, text="Training Mode", padding=5)
        mode_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Radiobutton(mode_frame, text="DPO (Best & Worst)", variable=self.dpo_mode_var, value="DPO", command=self._dpo_on_mode_change).pack(side="left", padx=10)
        ttk.Radiobutton(mode_frame, text="SFT (Best Only)", variable=self.dpo_mode_var, value="SFT", command=self._dpo_on_mode_change).pack(side="left", padx=10)
        
        ttk.Separator(mode_frame, orient="vertical").pack(side="left", fill="y", padx=15)
        
        self.sft_count_lbl = ttk.Label(mode_frame, text="SFT: 0")
        self.sft_count_lbl.pack(side="left", padx=10)
        
        # 3. Browse
        browse = ttk.LabelFrame(page, text="Select Documents", padding=10)
        browse.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(browse, text="Folder:").pack(side="left")
        ttk.Entry(browse, textvariable=self.dpo_folder_var, width=40).pack(side="left", padx=5)
        ttk.Button(browse, text="Browse", command=self._dpo_browse_folder).pack(side="left")
        
        ttk.Label(browse, text="Type:").pack(side="left", padx=(15, 5))
        ttk.Combobox(browse, textvariable=self.dpo_type_var, values=list(DOC_TYPES.keys()), state="readonly", width=12).pack(side="left")
        
        # Filters
        ttk.Label(browse, text="Min KB:").pack(side="left", padx=(15, 5))
        ttk.Entry(browse, textvariable=self.dpo_min_kb_var, width=5).pack(side="left")
        ttk.Label(browse, text="Max KB:").pack(side="left", padx=(5, 5))
        ttk.Entry(browse, textvariable=self.dpo_max_kb_var, width=6).pack(side="left")
        ttk.Label(browse, text="Max Files:").pack(side="left", padx=(5, 5))
        ttk.Entry(browse, textvariable=self.dpo_max_files_var, width=5).pack(side="left")
        
        ttk.Button(browse, text="Scan", command=self._dpo_scan_files, style="Accent.TButton").pack(side="left", padx=15)
        self.dpo_file_label = ttk.Label(browse, text="0 files")
        self.dpo_file_label.pack(side="left", padx=10)

        # 3. Main Area (PanedWindow)
        main = ttk.PanedWindow(page, orient="horizontal")
        main.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Left: Files & Samples
        left = ttk.LabelFrame(main, text="Pick Sample", padding=5, width=400)
        main.add(left, weight=1)
        
        # Nav
        nav = ttk.Frame(left)
        nav.pack(fill="x")
        ttk.Button(nav, text="<", width=3, command=self._dpo_prev_file).pack(side="left")
        self.dpo_file_info = ttk.Label(nav, text="File 0/0")
        self.dpo_file_info.pack(side="left", fill="x", expand=True) # Center?
        ttk.Button(nav, text=">", width=3, command=self._dpo_next_file).pack(side="right")
        
        self.dpo_file_preview = scrolledtext.ScrolledText(left, height=10, font=(NavyTheme.FONT_MONO, 8), 
                                                        bg=NavyTheme.BG_INPUT, fg=NavyTheme.TEXT_PRIMARY, insertbackground="white")
        self.dpo_file_preview.pack(fill="both", expand=True, pady=5)
        
        ttk.Label(left, text="Extracted Samples:").pack(anchor="w")
        self.dpo_samples_list = tk.Listbox(left, height=10, font=("Segoe UI", 9), 
                                           bg=NavyTheme.BG_INPUT, fg=NavyTheme.TEXT_PRIMARY, selectbackground=NavyTheme.BG_SELECTED)
        self.dpo_samples_list.pack(fill="both", expand=True)
        self.dpo_samples_list.bind("<<ListboxSelect>>", self._dpo_on_sample_select)
        
        # Right: Generate & Rate
        right = ttk.LabelFrame(main, text="Generate & Rate", padding=5)
        main.add(right, weight=2)
        
        # Prompt
        p_frame = ttk.Frame(right)
        p_frame.pack(fill="x")
        ttk.Label(p_frame, text="Prompt:").pack(anchor="w")
        self.dpo_prompt_text = tk.Text(p_frame, height=3, font=(NavyTheme.FONT_MONO, 9),
                                     bg=NavyTheme.BG_INPUT, fg=NavyTheme.TEXT_PRIMARY, insertbackground="white")
        self.dpo_prompt_text.pack(fill="x")
        
        # Responses (3)
        self.dpo_res_texts = []
        self.dpo_score_labels = []
        res_frame = ttk.Frame(right)
        res_frame.pack(fill="both", expand=True, pady=5)
        
        for i, char in enumerate(['A', 'B', 'C']):
            f = ttk.LabelFrame(res_frame, text=f"Response {char}")
            f.pack(fill="both", expand=True, pady=2)
            
            # Header for score
            score_lbl = ttk.Label(f, text="Score: -", style="Dim.TLabel")
            score_lbl.pack(anchor="ne")
            self.dpo_score_labels.append(score_lbl)
            
            t = scrolledtext.ScrolledText(f, height=4, font=(NavyTheme.FONT_MONO, 9),
                                        bg=NavyTheme.BG_INPUT, fg=NavyTheme.TEXT_PRIMARY, insertbackground="white")
            t.pack(fill="both", expand=True)
            self.dpo_res_texts.append(t)
            
        # Actions
        act_frame = ttk.Frame(right)
        act_frame.pack(fill="x", pady=5)
        ttk.Button(act_frame, text=f"{Icons.BOLT} Generate All", command=self._dpo_generate, style="Accent.TButton").pack(side="left")
        
        ttk.Label(act_frame, text="Seed:").pack(side="left", padx=(15, 5))
        self.dpo_seed_var = tk.StringVar(value="42")
        ttk.Entry(act_frame, textvariable=self.dpo_seed_var, width=6).pack(side="left")
        
        ttk.Label(act_frame, text="Max Tok:").pack(side="left", padx=(10, 5))
        self.dpo_max_tok_var = tk.StringVar(value="150")
        ttk.Entry(act_frame, textvariable=self.dpo_max_tok_var, width=6).pack(side="left")

        ttk.Button(act_frame, text=f"{Icons.TEST} Grade", command=self._dpo_run_grading).pack(side="left", padx=15)
        
        # Selection
        sel_frame = ttk.Frame(right)
        sel_frame.pack(fill="x", pady=5)
        ttk.Label(sel_frame, text="Chosen:").pack(side="left")
        self.dpo_chosen_var = tk.StringVar(value="A")
        ttk.Combobox(sel_frame, textvariable=self.dpo_chosen_var, values=["A", "B", "C"], width=3).pack(side="left")
        
        self.dpo_rejected_lbl = ttk.Label(sel_frame, text="Rejected:")
        self.dpo_rejected_lbl.pack(side="left", padx=(10, 0))
        self.dpo_rejected_var = tk.StringVar(value="B")
        self.dpo_rejected_combo = ttk.Combobox(sel_frame, textvariable=self.dpo_rejected_var, values=["A", "B", "C"], width=3)
        self.dpo_rejected_combo.pack(side="left")
        
        self.dpo_save_btn = ttk.Button(sel_frame, text=f"{Icons.SAVE} Save Pair", command=self._dpo_save_entry)
        self.dpo_save_btn.pack(side="right")
        self.dpo_pair_count_lbl = ttk.Label(sel_frame, text="DPO: 0")
        self.dpo_pair_count_lbl.pack(side="right", padx=10)
        
        # Initialize UI state
        self._dpo_on_mode_change()
        self._dpo_update_counts()

    # ---------------------------------------------------#
    # DPO LOGIC METHODS
    # ---------------------------------------------------#
    def _dpo_browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.dpo_folder_var.set(folder)
            self.dpo_settings.set("folder", folder)
            self.dpo_settings.save()

    def _dpo_load_tokenizer(self):
        path = filedialog.askopenfilename(filetypes=[("Tokenizer Config", "*.json")])
        if path:
            self.dpo_settings.set("tokenizer_path", path)
            self.dpo_settings.save()
            try:
                from tokenizers import Tokenizer as HFTokenizer
                self.dpo_generator_tokenizer = HFTokenizer.from_file(path)
                self.dpo_tok_status.config(text="Loaded", foreground=NavyTheme.SUCCESS)
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _dpo_load_creator_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt")])
        if path:
            def loader():
                try:
                    # Reuse existing loading logic if possible, or simple load
                    # Assuming standard model architecture from this trainer
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    ckpt = torch.load(path, map_location=device, weights_only=False)
                    # Extract config
                    cfg_dict = ckpt.get("model_cfg", {})
                    allowed = {f.name for f in fields(ModelConfig)}
                    cfg_dict = {k: v for k, v in cfg_dict.items() if k in allowed}
                    cfg = ModelConfig(**cfg_dict)
                    
                    self.dpo_creator_model = GPTModel(cfg).to(device)
                    state = ckpt.get("model_state_dict", ckpt.get("model", {}))
                    # Handle module prefix
                    if all(k.startswith("module.") for k in state):
                         state = {k[7:]: v for k, v in state.items()}
                         
                    self.dpo_creator_model.load_state_dict(state, strict=False)
                    self.dpo_creator_model.eval()
                    
                    self.gui_queue.put(lambda: self.dpo_model_status_lbl.config(text="Loaded", foreground=NavyTheme.SUCCESS))
                except Exception as e:
                    err = str(e)
                    self.gui_queue.put(lambda: messagebox.showerror("Error", err))
            
            self.dpo_settings.set("model_path", path)
            self.dpo_settings.save()
            threading.Thread(target=loader, daemon=True).start()
            self.dpo_model_status_lbl.config(text="Loading...", foreground=NavyTheme.ACCENT_CYAN)

    def _dpo_load_grader_local(self):
        path = filedialog.askdirectory(title="Select Local Grader Model")
        if path:
            if not self.dpo_grader: self.dpo_grader = AutoGrader()
            
            def load_worker():
                success = self.dpo_grader.load(path)
                color = NavyTheme.SUCCESS if success else NavyTheme.ERROR
                text = "Loaded (Disk)" if success else "Failed"
                self.gui_queue.put(lambda: self.dpo_grader_status.config(text=text, foreground=color))
            
            self.dpo_settings.set("grader_path", path)
            self.dpo_settings.save()
            threading.Thread(target=load_worker, daemon=True).start()
            self.dpo_grader_status.config(text="Loading...", foreground=NavyTheme.ACCENT_CYAN)

    def _dpo_load_grader_hf(self):
        from tkinter import simpledialog
        model_name = simpledialog.askstring("Load Grader", "Enter HuggingFace Model (e.g. microsoft/phi-2):")
        if model_name:
            if not self.dpo_grader: self.dpo_grader = AutoGrader()
            
            def load_worker():
                success = self.dpo_grader.load(model_name)
                color = NavyTheme.SUCCESS if success else NavyTheme.ERROR
                text = "Loaded" if success else "Failed"
                self.gui_queue.put(lambda: self.dpo_grader_status.config(text=text, foreground=color))
                
            self.dpo_settings.set("grader_path", model_name)
            self.dpo_settings.save()
            threading.Thread(target=load_worker, daemon=True).start()
            self.dpo_grader_status.config(text="Loading...", foreground=NavyTheme.ACCENT_CYAN)

    def _dpo_scan_files(self):
        folder = self.dpo_folder_var.get()
        doc_type = self.dpo_type_var.get()
        if not folder or not os.path.exists(folder): return
        
        info = DOC_TYPES.get(doc_type, DOC_TYPES["Story/Prose"])
        exts = info["extensions"]
        try:
            max_f = int(self.dpo_max_files_var.get())
            min_kb = int(self.dpo_min_kb_var.get())
            max_kb = int(self.dpo_max_kb_var.get())
            
            self.dpo_files = find_files_recursive(folder, exts, max_f, min_kb, max_kb)
            self.dpo_file_label.config(text=f"{len(self.dpo_files)} files")
            self.dpo_current_file_idx = 0
            self._dpo_update_file_nav()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _dpo_update_file_nav(self):
        if not self.dpo_files:
            self.dpo_file_preview.delete("1.0", "end")
            self.dpo_file_info.config(text="No files")
            return
        
        idx = self.dpo_current_file_idx
        path = self.dpo_files[idx]
        self.dpo_file_info.config(text=f"File {idx+1}/{len(self.dpo_files)}: {os.path.basename(path)}")
        
        content = read_file(path)
        self.dpo_file_preview.delete("1.0", "end")
        self.dpo_file_preview.insert("1.0", content)
        
        # Extract samples
        doc_type = self.dpo_type_var.get()
        mode = DOC_TYPES.get(doc_type, {}).get("mode", "prose")
        self.dpo_samples = extract_samples(content, mode)
        
        self.dpo_samples_list.delete(0, "end")
        for s in self.dpo_samples:
            preview = s[:50].replace("\n", " ") + "..."
            self.dpo_samples_list.insert("end", preview)

    def _dpo_prev_file(self):
        if self.dpo_files:
            self.dpo_current_file_idx = (self.dpo_current_file_idx - 1) % len(self.dpo_files)
            self._dpo_update_file_nav()

    def _dpo_next_file(self):
        if self.dpo_files:
            self.dpo_current_file_idx = (self.dpo_current_file_idx + 1) % len(self.dpo_files)
            self._dpo_update_file_nav()

    def _dpo_on_sample_select(self, event):
        sel = self.dpo_samples_list.curselection()
        if sel:
            idx = sel[0]
            if 0 <= idx < len(self.dpo_samples):
                prompt = self.dpo_samples[idx]
                self.dpo_prompt_text.delete("1.0", "end")
                self.dpo_prompt_text.insert("1.0", prompt)

    def _dpo_restore_session(self):
        """Load paths from settings and attempt to restore."""
        self._log(f"{Icons.REFRESH} Restoring DPO Session...")
        restored = 0
        
        # Tokenizer
        tok_path = self.dpo_settings.get("tokenizer_path")
        if tok_path and os.path.exists(tok_path):
            try:
                from tokenizers import Tokenizer as HFTokenizer
                self.dpo_generator_tokenizer = HFTokenizer.from_file(tok_path)
                self.dpo_tok_status.config(text="Loaded", foreground=NavyTheme.SUCCESS)
                self._log(f"   Tokenizer restored: {os.path.basename(tok_path)}")
                restored += 1
            except Exception as e:
                self._log(f"   {Icons.ERROR} Tokenizer failed: {e}")
        else:
            # Fallback: Try project folder
            project_dir = self.folder_var.get()
            possible_tok = os.path.join(project_dir, "tokenizer.json")
            if os.path.exists(possible_tok):
                 try:
                    from tokenizers import Tokenizer as HFTokenizer
                    self.dpo_generator_tokenizer = HFTokenizer.from_file(possible_tok)
                    self.dpo_tok_status.config(text="Loaded", foreground=NavyTheme.SUCCESS)
                    self._log(f"   Tokenizer found in project: {os.path.basename(possible_tok)}")
                    self.dpo_settings.set("tokenizer_path", possible_tok)
                    restored += 1
                 except: 
                     self._log("   No saved tokenizer found.")
            else:
                 self._log("   No saved tokenizer found.")
            
        # Model
        mod_path = self.dpo_settings.get("model_path")
        
        # Fallback: Check project checkpoint folder for latest pt
        if not mod_path or not os.path.exists(mod_path):
             ckpt_dir = self.output_var.get()
             if os.path.exists(ckpt_dir):
                 try:
                     files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
                     if files:
                         # Sort by mod time
                         files.sort(key=os.path.getmtime, reverse=True)
                         mod_path = files[0]
                         self._log(f"   Auto-detected latest model: {os.path.basename(mod_path)}")
                 except: pass

        if mod_path and os.path.exists(mod_path):
             self.dpo_settings.set("model_path", mod_path) # Update settings
             def loader():
                try:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    ckpt = torch.load(mod_path, map_location=device, weights_only=False)
                    cfg_dict = ckpt.get("model_cfg", {})
                    # 'fields' is now imported globally
                    allowed = {f.name for f in fields(ModelConfig)}
                    cfg_dict = {k: v for k, v in cfg_dict.items() if k in allowed}
                    cfg = ModelConfig(**cfg_dict)
                    self.dpo_creator_model = GPTModel(cfg).to(device)
                    state = ckpt.get("model_state_dict", ckpt.get("model", {}))
                    if all(k.startswith("module.") for k in state): state = {k[7:]: v for k, v in state.items()}
                    self.dpo_creator_model.load_state_dict(state, strict=False)
                    self.dpo_creator_model.eval()
                    self.gui_queue.put(lambda: self.dpo_model_status_lbl.config(text="Loaded", foreground=NavyTheme.SUCCESS))
                    self.gui_queue.put(lambda: self._log(f"   Model restored: {os.path.basename(mod_path)}"))
                except Exception as e:
                    err = str(e)
                    self.gui_queue.put(lambda: self.dpo_model_status_lbl.config(text="Error", foreground=NavyTheme.ERROR))
                    self.gui_queue.put(lambda: self._log(f"   {Icons.ERROR} Model restore failed: {err}"))
                    
             threading.Thread(target=loader, daemon=True).start()
             self.dpo_model_status_lbl.config(text="Loading...", foreground=NavyTheme.ACCENT_CYAN)
             restored += 1
        else:
            self._log("   No saved model found.")

        # Grader
        grad_path = self.dpo_settings.get("grader_path")
        if grad_path:
             if not self.dpo_grader: self.dpo_grader = AutoGrader()
             def grade_loader():
                 success = self.dpo_grader.load(grad_path)
                 color = NavyTheme.SUCCESS if success else NavyTheme.ERROR
                 text = "Loaded" if success else "Failed"
                 self.gui_queue.put(lambda: self.dpo_grader_status.config(text=text, foreground=color))
                 if success:
                     self.gui_queue.put(lambda: self._log(f"   Grader restored: {grad_path}"))
                 else:
                     self.gui_queue.put(lambda: self._log(f"   {Icons.ERROR} Grader load failed."))
                     
             threading.Thread(target=grade_loader, daemon=True).start()
             self.dpo_grader_status.config(text="Loading...", foreground=NavyTheme.ACCENT_CYAN)
             restored += 1
        else:
            self._log("   No saved grader found.")
            
        if restored == 0:
            self._show_toast("Nothing to restore", "info")
        else:
            self._show_toast(f"Restoring {restored} items...", "info")

    def _dpo_generate(self):
        if not self.dpo_creator_model or not self.dpo_generator_tokenizer:
            messagebox.showwarning("Warning", "Load Model and Tokenizer first")
            return
        
        prompt = self.dpo_prompt_text.get("1.0", "end").strip()
        if not prompt: return
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        def gen_worker():
            try:
                responses = []
                ids = self.dpo_generator_tokenizer.encode(prompt).ids
                
                for _ in range(3):
                    # Basic generation
                    x = torch.tensor([ids], dtype=torch.long, device=device)
                    # Limit to ~100 tokens
                    with torch.no_grad():
                        for _ in range(100):
                            logits = self.dpo_creator_model(x)[:, -1, :]
                            probs = F.softmax(logits / 0.9, dim=-1)
                            next_id = torch.multinomial(probs, 1)
                            x = torch.cat([x, next_id], dim=1)
                            # Stop cases (2=BOS, 3=EOS usually) - 
                            # Warning: token ids depend on tokenizer. Assuming defaults or checking common
                            # Often 0=unknown, 1=bos, 2=eos?
                            # Using arbitrary ID or checking tokenizer?
                            # Usually 0 or 2. We'll rely on string decoding for safety?
                            # Or just run for fixed length.
                            if next_id.item() == 2 or next_id.item() == 0: break
                            
                    out_text = self.dpo_generator_tokenizer.decode(x[0].tolist())
                    responses.append(out_text[len(prompt):]) # strip prompt
                
                self.gui_queue.put(lambda: self._dpo_show_responses(responses, prompt))
            except Exception as e:
                err = str(e)
                self.gui_queue.put(lambda: messagebox.showerror("Error", err))
        
        threading.Thread(target=gen_worker, daemon=True).start()

    def _dpo_show_responses(self, responses, prompt):
        for i, text in enumerate(responses):
            if i < len(self.dpo_res_texts):
                self.dpo_res_texts[i].delete("1.0", "end")
                self.dpo_res_texts[i].insert("1.0", text)
        
        self._show_toast("Generation Complete", "success")
        
        # Auto grade if loaded
        if self.dpo_grader and self.dpo_grader.loaded:
            self._dpo_run_grading(prompt)

    def _dpo_run_grading(self, prompt=None):
        if not self.dpo_grader or not self.dpo_grader.loaded:
            messagebox.showinfo("Info", "Load Auto-Grader first")
            return
            
        if not prompt:
            prompt = self.dpo_prompt_text.get("1.0", "end").strip()
            
        responses = [t.get("1.0", "end").strip() for t in self.dpo_res_texts]
        
        def grade_worker():
            try:
                # Get scores (0-10)
                results = self.dpo_grader.score_all(prompt, responses)
                
                def update_ui():
                    # Find best and worst
                    best_idx = 0
                    worst_idx = 0
                    best_score = -1
                    worst_score = 999
                    
                    for i, (label, score) in enumerate(results):
                        # Update label
                        lbl = self.dpo_score_labels[i]
                        # Calc perplexity-like metric (just inverse score for display?)
                        # Or display score 0-10
                        lbl.config(text=f"Score: {score:.1f}/10")
                        
                        # Color coding
                        if score >= 8.0: lbl.config(foreground=NavyTheme.SUCCESS)
                        elif score <= 4.0: lbl.config(foreground=NavyTheme.ERROR)
                        else: lbl.config(foreground=NavyTheme.WARNING)
                        
                        if score > best_score:
                            best_score = score
                            best_idx = i
                        if score < worst_score:
                            worst_score = score
                            worst_idx = i
                            
                    # Auto select dropdowns
                    labels = ["A", "B", "C"]
                    if best_idx != worst_idx:
                        self.dpo_chosen_var.set(labels[best_idx])
                        self.dpo_rejected_var.set(labels[worst_idx])
                        
                    self._show_toast("Grading Complete", "success")

                self.gui_queue.put(update_ui)
            except Exception as e:
                self.gui_queue.put(lambda: print(f"Grading error: {e}"))
                
        threading.Thread(target=grade_worker, daemon=True).start()


    def _dpo_on_mode_change(self):
        """Handle DPO/SFT mode switch."""
        mode = self.dpo_mode_var.get()
        if mode == "SFT":
            self.dpo_rejected_lbl.pack_forget()
            self.dpo_rejected_combo.pack_forget()
            self.dpo_save_btn.config(text=f"{Icons.SAVE} Save SFT")
        else:
            self.dpo_rejected_lbl.pack(side="left", padx=(10, 0))
            self.dpo_rejected_combo.pack(side="left")
            self.dpo_save_btn.config(text=f"{Icons.SAVE} Save Pair")
            
        self.dpo_settings.set("mode", mode)
        self.dpo_settings.save()

    def _dpo_update_counts(self):
        """Update counts for DPO and SFT."""
        try:
            dpo_count = 0
            if os.path.exists(DPO_PAIRS_FILE):
                 with open(DPO_PAIRS_FILE, 'r') as f: dpo_count = len(json.load(f))
            self.dpo_pair_count_lbl.config(text=f"DPO: {dpo_count}")
            
            sft_count = 0
            if os.path.exists(SFT_FILE):
                 with open(SFT_FILE, 'r') as f: sft_count = len(json.load(f))
            # Just verify sft_count_lbl exists before config (it should)
            if hasattr(self, 'sft_count_lbl'):
                self.sft_count_lbl.config(text=f"SFT: {sft_count}")
        except: pass

    def _dpo_save_entry(self):
        mode = self.dpo_mode_var.get()
        prompt = self.dpo_prompt_text.get("1.0", "end").strip()
        if not prompt: return

        chosen_char = self.dpo_chosen_var.get()
        labels = ["A", "B", "C"]
        if chosen_char not in labels: return
        c_idx = labels.index(chosen_char)
        chosen = self.dpo_res_texts[c_idx].get("1.0", "end").strip()
        
        ts = time.time()
        
        if mode == "SFT":
            # Save SFT entry
            entry = {"prompt": prompt, "response": chosen, "model": "gpt", "ts": ts}
            
            existing = []
            if os.path.exists(SFT_FILE):
                 try:
                     with open(SFT_FILE, 'r') as f: existing = json.load(f)
                 except: pass
                 
            existing.append(entry)
            with open(SFT_FILE, 'w') as f: json.dump(existing, f, indent=2)
            self._show_toast("SFT Entry Saved", "success")
            
        else:
            # Save DPO Pair
            rejected_char = self.dpo_rejected_var.get()
            if rejected_char not in labels: return
            r_idx = labels.index(rejected_char)
            
            if c_idx == r_idx:
                messagebox.showwarning("Warning", "Chosen and Rejected cannot be same")
                return
            
            rejected = self.dpo_res_texts[r_idx].get("1.0", "end").strip()
            
            pair = {"prompt": prompt, "chosen": chosen, "rejected": rejected, "model": "gpt", "ts": ts}
            
            existing = []
            if os.path.exists(DPO_PAIRS_FILE):
                 try:
                     with open(DPO_PAIRS_FILE, 'r') as f: existing = json.load(f)
                 except: pass
                 
            existing.append(pair)
            with open(DPO_PAIRS_FILE, 'w') as f: json.dump(existing, f, indent=2)
            self._show_toast("DPO Pair Saved", "success")
            
        self._dpo_update_counts()

    def _export_dpo_pairs(self):
        """Export DPO or SFT data."""
        # Simple dialog to ask what to export
        dialog = tk.Toplevel(self.root)
        dialog.title("Export Data")
        dialog.geometry("300x150")
        
        ttk.Label(dialog, text="Select data to export:", font=("", 11, "bold")).pack(pady=10)
        
        def do_export(dtype):
            dialog.destroy()
            src = DPO_PAIRS_FILE if dtype == "DPO" else SFT_FILE
            
            if not os.path.exists(src):
                 messagebox.showinfo("Info", f"No {dtype} data to export.")
                 return
                 
            dest = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")], title=f"Export {dtype} Data")
            if dest:
                import shutil
                try:
                    shutil.copy2(src, dest)
                    self._show_toast(f"Exported {dtype}", "success")
                except Exception as e:
                    messagebox.showerror("Error", str(e))
        
        ttk.Button(dialog, text="Export DPO Pairs", command=lambda: do_export("DPO")).pack(fill="x", padx=20, pady=5)
        ttk.Button(dialog, text="Export SFT Data", command=lambda: do_export("SFT")).pack(fill="x", padx=20, pady=5)

    def _dpo_save_settings_manual(self):
        if self.dpo_settings.save():
            self._show_toast("Settings Saved", "success")
        else:
            messagebox.showerror("Error", "Failed to save settings")

    # ---------------------------------------------------#
    # Method name: _browse_dpo_file
    # ---------------------------------------------------#
    def _browse_dpo_file(self):
        """Browse for DPO preference file."""
        path = filedialog.askopenfilename(
            filetypes=[("DPO Files", "*.dpo"), ("JSON", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.dpo_file_var.set(path)

    # ---------------------------------------------------#
    # Method name: _browse_dpo_model
    # ---------------------------------------------------#
    def _browse_dpo_model(self):
        """Browse for base model checkpoint."""
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self.dpo_model_var.set(path)
            self.dpo_model_status.config(text=f"Selected: {os.path.basename(path)}")

    # ---------------------------------------------------#
    # Method name: _load_dpo_file
    # ---------------------------------------------------#
    def _load_dpo_file(self):
        """Load preference data from file."""
        path = self.dpo_file_var.get()
        if not path or not os.path.exists(path):
            messagebox.showwarning("Warning", "Select a valid preference file first!")
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("Data must be a JSON array")
            
            # Validate pairs
            valid_pairs = []
            for i, pair in enumerate(data):
                if all(k in pair for k in ['prompt', 'chosen', 'rejected']):
                    valid_pairs.append(pair)
                else:
                    self._log(f"{Icons.WARNING} Pair {i} missing required keys, skipping")
            
            self._dpo_preference_data = valid_pairs
            
            # Update stats
            self.dpo_stats_label.config(
                text=f"{Icons.CHECK} Loaded {len(valid_pairs)} preference pairs from {os.path.basename(path)}"
            )
            
            # Preview first 5
            self.dpo_preview_text.delete("1.0", "end")
            for i, pair in enumerate(valid_pairs[:5]):
                prompt_preview = pair['prompt'][:100].replace('\n', ' ')
                chosen_preview = pair['chosen'][:80].replace('\n', ' ')
                rejected_preview = pair['rejected'][:80].replace('\n', ' ')
                
                self.dpo_preview_text.insert("end", f"[{i+1}] Prompt: {prompt_preview}...\n")
                self.dpo_preview_text.insert("end", f"    ✓ Chosen: {chosen_preview}...\n")
                self.dpo_preview_text.insert("end", f"    ✗ Rejected: {rejected_preview}...\n\n")
            
            if len(valid_pairs) > 5:
                self.dpo_preview_text.insert("end", f"... and {len(valid_pairs) - 5} more pairs\n")
            
            self._log(f"{Icons.SUCCESS} Loaded {len(valid_pairs)} preference pairs")
            
        except json.JSONDecodeError as e:
            messagebox.showerror("Error", f"Invalid JSON:\n{e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")

    # ---------------------------------------------------#
    # Method name: _start_dpo_training
    # ---------------------------------------------------#
    def _start_dpo_training(self):
        """Start DPO training."""
        if not TORCH_AVAILABLE:
            messagebox.showerror("Error", "PyTorch not available!")
            return
        
        # Validate preference data
        if not self._dpo_preference_data:
            messagebox.showwarning("Warning", "Load preference data first!")
            return
        
        # Validate model
        model_path = self.dpo_model_var.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showwarning("Warning", "Select a base model checkpoint first!")
            return
        
        # Validate tokenizer - look for it
        model_dir = os.path.dirname(model_path)
        tokenizer_path = None
        
        # Search locations
        search_paths = [
            os.path.join(model_dir, "tokenizer.json"),
            os.path.join(model_dir, f"{self.checkpoint_name_var.get()}_tokenizer.json"),
        ]
        
        for p in search_paths:
            if os.path.exists(p):
                tokenizer_path = p
                break
        
        if not tokenizer_path:
            tokenizer_path = filedialog.askopenfilename(
                title="Select Tokenizer",
                filetypes=[("JSON", "*.json"), ("All files", "*.*")]
            )
            if not tokenizer_path:
                messagebox.showwarning("Warning", "Tokenizer required!")
                return
        
        # Parse settings
        try:
            dpo_epochs = int(self.dpo_epochs_var.get())
            dpo_lr = float(self.dpo_lr_var.get())
            dpo_beta = float(self.dpo_beta_var.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid settings: {e}")
            return
        
        self._log(f"{Icons.ROCKET} Starting DPO Training")
        self._log(f"   Base model: {os.path.basename(model_path)}")
        self._log(f"   Tokenizer: {os.path.basename(tokenizer_path)}")
        self._log(f"   Pairs: {len(self._dpo_preference_data)}")
        
        def dpo_worker():
            try:
                # Load tokenizer
                self.gui_queue.put(lambda: self._log(f"{Icons.LOADING} Loading tokenizer..."))
                tokenizer = BytePairTokenizer()
                tokenizer.load(tokenizer_path)
                
                # Load model
                self.gui_queue.put(lambda: self._log(f"{Icons.LOADING} Loading base model..."))
                ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
                
                # Get model config
                cfg_dict = ckpt.get("model_cfg", {})
                if "max_seq" in cfg_dict and "max_seq_len" not in cfg_dict:
                    cfg_dict["max_seq_len"] = cfg_dict["max_seq"]
                
                from dataclasses import fields
                allowed = {f.name for f in fields(ModelConfig)}
                cfg_dict = {k: v for k, v in cfg_dict.items() if k in allowed}
                model_cfg = ModelConfig(**cfg_dict)
                
                # Create policy model
                model = GPTModel(model_cfg)
                state = ckpt.get("model_state_dict", ckpt.get("model", {}))
                if all(k.startswith("module.") for k in state):
                    state = {k[7:]: v for k, v in state.items()}
                model.load_state_dict(state, strict=False)
                
                # Create reference model (frozen copy)
                self.gui_queue.put(lambda: self._log(f"{Icons.LOADING} Creating reference model..."))
                ref_model = GPTModel(model_cfg)
                ref_model.load_state_dict(model.state_dict())
                
                # Create training config
                train_cfg = TrainingConfig(
                    learning_rate=dpo_lr,
                    epochs=dpo_epochs,
                    precision=self.precision_var.get(),
                    gradient_clip=float(self.grad_clip_var.get()),
                    weight_decay=0.01
                )
                
                # Create checkpoint config
                ckpt_cfg = CheckpointConfig(
                    output_dir=self._get_project_dir(),
                    checkpoint_name=self.checkpoint_name_var.get() or "dpo_model"
                )
                
                # Create DPO trainer
                dpo_trainer = DPOTrainer(
                    model=model,
                    ref_model=ref_model,
                    tokenizer=tokenizer,
                    model_cfg=model_cfg,
                    train_cfg=train_cfg,
                    ckpt_cfg=ckpt_cfg,
                    app_state=self.app_state
                )
                dpo_trainer.beta = dpo_beta
                dpo_trainer.dpo_lr = dpo_lr
                dpo_trainer.log_cb = self._log
                
                # Run training
                dpo_trainer.train(self._dpo_preference_data)
                
                self.gui_queue.put(lambda: self._log(f"{Icons.SUCCESS} DPO Training complete!"))
                self.gui_queue.put(lambda: messagebox.showinfo("Complete", "DPO Training finished!"))
                
            except Exception as e:
                import traceback
                self.gui_queue.put(lambda: self._log(f"{Icons.ERROR} DPO failed: {e}"))
                self.gui_queue.put(lambda: self._log(traceback.format_exc()))
                self.app_state.state = AppState.ERROR
        
        # Switch to progress page and start
        self._show_page("progress")
        threading.Thread(target=dpo_worker, daemon=True).start()

    # ---------------------------------------------------#
    # Method name: _build_help_page
    # ---------------------------------------------------#
    def _open_help_window(self):
        """Open the Help Window (Modal-less)."""
        if hasattr(self, 'help_window') and self.help_window.winfo_exists():
            self.help_window.lift()
            return

        self.help_window = tk.Toplevel(self.root)
        self.help_window.title("Help & Documentation")
        self.help_window.geometry("600x700")
        self.help_window.configure(bg=NavyTheme.NAVY_DARK)
        
        # Header
        head = ttk.Frame(self.help_window)
        head.pack(fill="x", padx=10, pady=10)
        ttk.Label(head, text=f"{Icons.HELP} User Guide", style="Title.TLabel").pack(side="left")
        
        # Content
        container = ttk.Frame(self.help_window)
        container.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        sb = ttk.Scrollbar(container)
        sb.pack(side="right", fill="y")
        
        text = tk.Text(container, bg=NavyTheme.BG_INPUT, fg=NavyTheme.TEXT_PRIMARY,
                       font=(NavyTheme.FONT_FAMILY, 10), relief="flat", wrap="word",
                       yscrollcommand=sb.set, padx=15, pady=15)
        text.pack(fill="both", expand=True)
        sb.config(command=text.yview)
        
        # Initialize Renderer
        self.help_renderer = RichTextRenderer(text, NavyTheme)
        
        # Load content
        try:
            with open("e:\\ai\\my_gpt\\HELP.LLM", "r", encoding="utf-8") as f:
                content = f.read()
            self.help_renderer.render(content)
        except Exception as e:
            text.insert("end", f"Error loading help file: {e}")
            
        text.config(state="disabled")


    # ---------------------------------------------------#
    # Method name: _on_state_change
    # ---------------------------------------------------#
    def _on_state_change(self, old_state: str, new_state: str):
        """Handle state changes."""
        self.gui_queue.put(self._update_state_ui)

    # ---------------------------------------------------#
    # Method name: _update_state_ui
    # ---------------------------------------------------#
    def _update_state_ui(self):
        """Update UI based on current state."""
        state = self.app_state.state

        # Status indicator text and color
        status_map = {
            AppState.IDLE: (f"{Icons.CHECK} Ready", NavyTheme.SUCCESS),
            AppState.SCANNING: (f"{Icons.LOADING} Scanning...", NavyTheme.INFO),
            AppState.PROCESSING: (f"{Icons.LOADING} Processing...", NavyTheme.INFO),
            AppState.TOKENIZING: (f"{Icons.LOADING} Tokenizing...", NavyTheme.INFO),
            AppState.ENCODING: (f"{Icons.LOADING} Encoding...", NavyTheme.INFO),
            AppState.TRAINING: (f"{Icons.FIRE} Training", NavyTheme.WARNING),
            AppState.PAUSED: (f"{Icons.PAUSE} Paused", NavyTheme.WARNING),
            AppState.SAVING: (f"{Icons.SAVE} Saving...", NavyTheme.INFO),
            AppState.LOADING: (f"{Icons.LOAD} Loading...", NavyTheme.INFO),
            AppState.ERROR: (f"{Icons.ERROR} Error", NavyTheme.ERROR),
        }

        text, color = status_map.get(state, (f"{Icons.INFO} Unknown", NavyTheme.TEXT_DIM))
        # self.status_indicator.config(text=text)

        # Determine button states
        is_idle = state == AppState.IDLE
        is_error = state == AppState.ERROR
        is_paused = state == AppState.PAUSED
        is_training = state == AppState.TRAINING
        is_busy = state not in (AppState.IDLE, AppState.ERROR)

        # Start button: only when idle or error
        self.start_btn.config(state="normal" if (is_idle or is_error) else "disabled")

        # Stop button: enabled during ANY busy state
        self.stop_btn.config(state="normal" if is_busy else "disabled")

        # Pause button: only during training or paused
        self.pause_btn.config(state="normal" if (is_training or is_paused) else "disabled")

        # Update pause button text
        if is_paused:
            self.pause_btn.config(text=f"{Icons.PLAY} Resume")
        else:
            self.pause_btn.config(text=f"{Icons.PAUSE} Pause")
            
        # Update utility buttons
        if hasattr(self, 'save_btn'):
            # Save: Only during training/paused
            self.save_btn.config(state="normal" if (is_training or is_paused) else "disabled")
            
        if hasattr(self, 'test_btn'):
             # Test: Always enable, unless busy processing non-interruptible
             # But let's disable during training to avoid VRAM conflict
             self.test_btn.config(state="disabled" if is_training else "normal")
             
        if hasattr(self, 'plot_btn'):
             self.plot_btn.config(state="normal")

        # Update status bar
        if hasattr(self, 'status_bar'):
            self.status_bar.set_state(state, text.split()[-1] if len(text.split()) > 1 else text)
            
        # Update window title with dirty indicator
        self._update_title_dirty()
        
        # Lock/Unlock interface based on busy state
        self._lock_inputs(is_busy)

    def _lock_inputs(self, locked: bool):
        """Lock input fields on settings pages during training."""
        target_pages = ["data", "model", "training", "refine", "checkpoint", "dpo", "context_extend"]
        state = "disabled" if locked else "normal"
        
        def _recursive_state(widget, state):
            try:
                if isinstance(widget, (ttk.Entry, ttk.Button, ttk.Checkbutton, ttk.Radiobutton, ttk.Combobox, tk.Listbox, tk.Text)):
                     widget.configure(state=state)
            except:
                pass
            for child in widget.winfo_children():
                _recursive_state(child, state)

        for pid in target_pages:
            if pid not in self.page_frames: continue
            page = self.page_frames[pid]
            # Don't disable the start/stop buttons in header, only page content
            # Page content is inside 'page' frame.
            _recursive_state(page, state)
            
            # Re-enable Scan Stop button if needed? No, refiner has stop.
            # But the MAIN stop button is in Header, which is NOT in target_pages. 
            # So main Stop button remains active (handled by _update_state_ui).


    # ---------------------------------------------------#
    # Method name: _start_monitors
    # ---------------------------------------------------#
    def _start_monitors(self):
        """Start background monitors."""
        self._vram_monitor_running = True
        # self._update_vram()
        self._update_progress_display()

    # ---------------------------------------------------#
    # Method name: _update_vram
    # ---------------------------------------------------#
    def _update_vram(self):
        """Update VRAM display."""
        if not self._vram_monitor_running:
            return

        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            try:
                used = torch.cuda.memory_allocated() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.vram_label.config(text=f"{Icons.MEMORY} VRAM: {used:.1f}/{total:.1f} GB")
            except Exception:
                pass

        self.root.after(1000, self._update_vram)



    # ---------------------------------------------------#
    # Method name: _update_progress_display
    # ---------------------------------------------------#
    def _update_progress_display(self):
        """Update progress bars and labels."""
        if not hasattr(self, '_vram_monitor_running') or not self._vram_monitor_running:
            return

        if self._stop_requested:
            self.root.after(100, self._update_progress_display)
            return

        state = self.app_state.state
        progress, msg = self.app_state.progress
        sub_progress, sub_msg = self.app_state.sub_progress

        self.overall_progress_bar.set(progress * 100)
        self.epoch_progress_bar.set(sub_progress * 100)

        # State-specific icons
        icon_map = {
            AppState.PROCESSING: Icons.LOADING,
            AppState.TOKENIZING: Icons.LOADING,
            AppState.ENCODING: Icons.BOLT,
            AppState.TRAINING: Icons.FIRE,
            AppState.PAUSED: Icons.PAUSE,
            AppState.SAVING: Icons.SAVE,
            AppState.LOADING: Icons.LOAD,
        }
        icon = icon_map.get(state, Icons.INFO)

        # Update status label with current message
        status_text = f"{icon} {sub_msg}" if sub_msg else f"{icon} {msg}"
        self.progress_status_label.config(text=status_text)

        # Try to extract ETA from message if it exists there (for non-training stages)
        if state != AppState.TRAINING:
            eta_text = ""
            if "ETA:" in status_text:
                eta_text = status_text.split("ETA:")[-1].strip()
            self.progress_eta_label.config(text=eta_text)

        self.root.after(50, self._update_progress_display)

    # ---------------------------------------------------#
    # Method name: _update_model_size_estimate
    # ---------------------------------------------------#
    def _update_model_size_estimate(self):
        """Update model size estimate based on current settings."""
        try:
            d_model = int(self.d_model_var.get())
            n_layers = int(self.n_layers_var.get())
            d_ff = int(self.d_ff_var.get())
            vocab_size = int(self.vocab_size_var.get())

            # Rough parameter count estimate
            embed_params = vocab_size * d_model * 2  # token + position
            attn_params = n_layers * (4 * d_model * d_model)  # qkv + out
            ff_params = n_layers * (2 * d_model * d_ff)  # up + down
            other_params = n_layers * 2 * d_model + d_model  # layer norms

            total_params = embed_params + attn_params + ff_params + other_params
            size_mb = total_params * 4 / 1024 / 1024  # float32 size

            self.model_size_label.config(
                text=f"{Icons.MODEL} Estimated: {total_params / 1e6:.1f}M parameters | "
                     f"~{size_mb:.0f} MB (FP32) | ~{size_mb / 2:.0f} MB (FP16/BF16)"
            )
        except (ValueError, ZeroDivisionError):
            self.model_size_label.config(text="Enter valid values to see estimate")

    # ========================================================================
    # ACTIONS
    # ========================================================================

    # ---------------------------------------------------#
    # Method name: _browse_folder
    # ---------------------------------------------------#
    def _browse_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.folder_var.set(path)

    # ---------------------------------------------------#
    # Method name: _browse_output
    # ---------------------------------------------------#
    def _browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_var.set(path)

    # ---------------------------------------------------#
    # Method name: _browse_resume
    # ---------------------------------------------------#
    def _browse_resume(self):
        path = filedialog.askopenfilename(filetypes=[("Checkpoint", "*.pt")])
        if path:
            self.resume_var.set(path)

    # ---------------------------------------------------#
    # Method name: _scan_folder
    # ---------------------------------------------------#
    def _scan_folder(self):
        """Scan folder for Python files with progress."""
        folder = self.folder_var.get()
        if not folder:
            messagebox.showwarning("Warning", "Select a folder first")
            return

        if not os.path.isdir(folder):
            messagebox.showerror("Error", "Invalid folder path")
            return

        self.app_state.state = AppState.SCANNING

        # Show progress
        if hasattr(self, 'scan_progress_frame'):
            self.scan_progress_frame.pack(fill="x", pady=(15, 0))

        def scan_worker():
            def progress_cb(progress: float, message: str):
                if hasattr(self, 'scan_progress_bar'):
                    self.gui_queue.put(lambda: self.scan_progress_bar.set(progress * 100))
                if hasattr(self, 'scan_progress_label'):
                    self.gui_queue.put(lambda: self.scan_progress_label.configure(text=message))

            files = self.processor.scan(folder, progress_cb)
            total_size = sum(f['size'] for f in files)

            def update_ui():
                if hasattr(self, 'scan_progress_frame'):
                    self.scan_progress_frame.pack_forget()

                self._log(f"{Icons.SUCCESS} Found {len(files):,} Matching files ({total_size / 1024 / 1024:.2f} MB)")
                self.app_state.state = AppState.IDLE

                # Update estimates
                self._update_effective_files()
                self._update_dataset_estimate()

            self.gui_queue.put(update_ui)

        threading.Thread(target=scan_worker, daemon=True).start()

    # ---------------------------------------------------#
    # Method name: _clear_vram
    # ---------------------------------------------------#
    def _clear_vram(self):
        """Clear GPU memory."""
        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            torch.cuda.empty_cache()
            gc.collect()
            self._log(f"{Icons.SUCCESS} VRAM cache cleared")

    # ---------------------------------------------------#
    # Method name: _log
    # ---------------------------------------------------#
    def _log(self, msg: str, tag: str = None):
        """Add message to log with optional tag for coloring."""

        def _update():
            self.log_text.config(state="normal")
            timestamp = datetime.now().strftime("%H:%M:%S")

            if tag:
                self.log_text.insert("end", f"[{timestamp}] ", "")
                self.log_text.insert("end", f"{msg}\n", tag)
            else:
                # Auto-detect tag from icons
                if Icons.SUCCESS in msg or Icons.CHECK in msg:
                    detected_tag = "success"
                elif Icons.ERROR in msg or Icons.CROSS in msg:
                    detected_tag = "error"
                elif Icons.WARNING in msg:
                    detected_tag = "warning"
                elif Icons.SPARKLE in msg or Icons.ROCKET in msg:
                    detected_tag = "accent"
                else:
                    detected_tag = None

                self.log_text.insert("end", f"[{timestamp}] {msg}\n", detected_tag)

            self.log_text.see("end")
            self.log_text.config(state="disabled")

        self.gui_queue.put(_update)

    # ---------------------------------------------------#
    # Method name: _show_toast
    # ---------------------------------------------------#
    def _show_toast(self, message: str, level: str = "info", duration: int = 3000):
        """Show a toast notification.
        
        Args:
            message: Text to display
            level: "info", "success", "warning", or "error"
            duration: Time in ms before auto-dismiss
        """
        if hasattr(self, 'toast'):
            self.toast.show(message, level, duration)

    # ---------------------------------------------------#
    # Method name: _update_metrics_display
    # ---------------------------------------------------#
    # ---------------------------------------------------#
    # Method name: _update_metrics_display
    # ---------------------------------------------------#
    def _update_metrics_display(self, metrics: Dict):
        """Update metrics UI (Labels and Progress Bar)."""

        def _update():
            # 1. Update Labels (Grid)
            # Quadrant 1: Status
            self.metric_lbl_epoch.config(text=f"Epoch:    {metrics.get('epoch', 0)} / {metrics.get('total_epochs', 0)}")
            self.metric_lbl_step.config(text=f"Step:     {metrics.get('global_step', 0)/1000:.1f}k / {metrics.get('total_global_steps', 0)/1000:.1f}k")
            
            prog_pct = metrics.get('global_step', 0) / max(1, metrics.get('total_global_steps', 1)) * 100
            self.metric_lbl_prog_text.config(text=f"Progress: {prog_pct:.1f}%")
            self.metric_prog_bar['value'] = prog_pct

            # Quadrant 2: Losses
            self.metric_lbl_train_loss.config(text=f"Train:    {metrics.get('train_loss', 0):.4f}")
            self.metric_lbl_val_loss.config(text=f"Valid:    {metrics.get('val_loss', 0):.4f}")
            self.metric_lbl_best_loss.config(text=f"Best:     {metrics.get('best_loss', 0):.4f}")
            self.metric_lbl_ppl.config(text=f"Perplex:  {metrics.get('perplexity', 0):.1f}")
            
            # Quadrant 3: Performance
            self.metric_lbl_speed.config(text=f"Speed:    {metrics.get('steps_per_sec', 0):.2f} steps/s")
            self.metric_lbl_tokens.config(text=f"Tokens:   {metrics.get('tokens_per_sec', 0)/1000:.1f}k /s")
            
            elapsed = timedelta(seconds=int(metrics.get('elapsed', 0)))
            self.metric_lbl_elapsed.config(text=f"Elapsed:  {elapsed}")
            
            eta_seconds = int(metrics.get('eta', 0))
            if eta_seconds > 86400:
                eta_str = f"{eta_seconds // 86400}d {eta_seconds % 86400 // 3600}h"
            elif eta_seconds > 3600:
                eta_str = f"{eta_seconds // 3600}h {eta_seconds % 3600 // 60}m"
            else:
                eta_str = f"{eta_seconds // 60}m {eta_seconds % 60}s"
            self.metric_lbl_eta.config(text=f"ETA:      {eta_str}")

            # Quadrant 4: Resources
            vram = metrics.get('gpu_mem', 0)
            vram_total = metrics.get('gpu_total', 1)
            vram_pct = (vram / vram_total) * 100
            
            self.metric_lbl_vram.config(text=f"VRAM:     {vram:.1f} / {vram_total:.0f} GB")
            self.metric_lbl_usage.config(text=f"Usage:    {vram_pct:.0f}%")
            self.metric_lbl_lr.config(text=f"LR:       {metrics.get('lr', 0):.2e}")

            # 2. Update Main Progress Bars (Bottom)
            self.overall_progress_bar.set(prog_pct)
            
            # Epoch progress might need its own calc if not provided, assuming roughly linear within epoch
            # If metrics doesn't provide epoch_progress, we can approximate or ignore
            # self.epoch_progress_bar.set(...) 

            # Update main ETA label
            self.progress_eta_label.config(text=f"ETA: {eta_str}")

        self.gui_queue.put(_update)

    def _colorize_metrics(self):
        """Apply colors to metrics text."""
        # Simple highlighting for numbers and headers
        text = self.metrics_text.get("1.0", "end")

        # Headers (bold + accent)
        for line_idx, line in enumerate(text.split('\n')):
            if any(x in line for x in ["TRAINING STATUS", "LOSSES", "SPEED", "RESOURCES"]):
                self.metrics_text.tag_add("header", f"{line_idx + 1}.0", f"{line_idx + 1}.end")

        self.metrics_text.tag_config("header", foreground=NavyTheme.ACCENT_CYAN,
                                     font=(NavyTheme.FONT_MONO, 10, "bold"))

    # ---------------------------------------------------#
    # Method name: _get_configs
    # ---------------------------------------------------#
    def _get_configs(self) -> List[str]:
        """Parse UI values into config objects with validation."""
        errors = []
        warnings = []

        # Detect GPU VRAM for validation
        gpu_vram_gb = 0
        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            try:
                gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            except Exception:
                gpu_vram_gb = 8

        # =================================================================
        # Model Config Validation
        # =================================================================
        try:
            d_model = int(self.d_model_var.get())
            n_heads = int(self.n_heads_var.get())
            n_layers = int(self.n_layers_var.get())
            d_ff = int(self.d_ff_var.get())
            vocab_size = int(self.vocab_size_var.get())
            max_seq_len = int(self.max_seq_var.get())
            dropout = float(self.dropout_var.get())

            if d_model < 64:
                errors.append(f"Embedding dim ({d_model}) too small. Minimum: 64")
            elif d_model > 2048:
                errors.append(f"Embedding dim ({d_model}) too large for personal GPU. Maximum: 2048")
            elif d_model % 64 != 0:
                warnings.append(f"Embedding dim ({d_model}) not divisible by 64. May be suboptimal.")

            if n_heads < 1:
                errors.append(f"Attention heads must be at least 1")
            elif n_heads > 32:
                errors.append(f"Attention heads ({n_heads}) excessive. Maximum: 32")
            elif d_model % n_heads != 0:
                errors.append(f"Embedding dim ({d_model}) must be divisible by heads ({n_heads})")
            else:
                head_dim = d_model // n_heads
                if head_dim not in [32, 64, 128]:
                    warnings.append(f"Head dim ({head_dim}) not optimal. Prefer 64 or 128.")




            if n_layers < 1:
                errors.append(f"Layers must be at least 1")
            elif n_layers > 48:
                errors.append(f"Layers ({n_layers}) too many. Maximum: 48")

            if d_ff < d_model:
                errors.append(f"FFN dim ({d_ff}) should be >= embedding dim ({d_model})")
            elif d_ff > 16384:
                errors.append(f"FFN dim ({d_ff}) too large. Maximum: 16384")
            elif d_ff != d_model * 4:
                warnings.append(f"FFN dim typically 4x embedding ({d_model * 4})")

            if vocab_size < 1000:
                errors.append(f"Vocab size ({vocab_size}) too small. Minimum: 1000")
            elif vocab_size > 100000:
                errors.append(f"Vocab size ({vocab_size}) too large. Maximum: 100000")

            if max_seq_len < 64:
                errors.append(f"Context length ({max_seq_len}) too small. Minimum: 64")
            elif max_seq_len > 4096:
                errors.append(f"Context length ({max_seq_len}) too large. Maximum: 4096")

            if dropout < 0 or dropout >= 1:
                errors.append(f"Dropout ({dropout}) must be between 0 and 1")
            elif dropout > 0.5:
                warnings.append(f"Dropout ({dropout}) very high. Typical: 0.1-0.3")

            self.model_cfg = ModelConfig(
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_ff,
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )

        except ValueError as e:
            errors.append(f"Model config: Invalid number format - {e}")

        # =================================================================
        # Training Config Validation
        # =================================================================
        try:
            learning_rate = float(self.lr_var.get())
            batch_size = int(self.batch_var.get())
            epochs = int(self.epochs_var.get())
            context_length = int(self.max_seq_var.get())
            stride = int(self.stride_var.get())
            warmup_ratio = float(self.warmup_var.get())
            gradient_clip = float(self.grad_clip_var.get())
            val_split = float(self.val_split_var.get())
            precision = self.precision_var.get()
            early_stopping = self.early_stop_var.get()
            patience = int(self.patience_var.get())

            if learning_rate <= 0:
                errors.append(f"Learning rate must be positive")
            elif learning_rate > 0.01:
                errors.append(f"Learning rate ({learning_rate}) too high. Maximum: 0.01")
            elif learning_rate > 0.001:
                warnings.append(f"Learning rate ({learning_rate}) is high. Typical: 1e-4 to 5e-4")
            elif learning_rate < 1e-6:
                warnings.append(f"Learning rate ({learning_rate}) very low.")

            if batch_size < 1:
                errors.append(f"Batch size must be at least 1")
            elif batch_size > 512:
                errors.append(f"Batch size ({batch_size}) too large. Maximum: 512")
            elif batch_size % 8 != 0:
                suggested = ((batch_size + 4) // 8) * 8
                warnings.append(f"Batch size not divisible by 8. Suggest: {suggested}")

            # =================================================================
            # VRAM Estimation - Empirically calibrated
            # =================================================================
            # =================================================================
            # VRAM Estimation - Conservative
            # =================================================================
            d_model = int(self.d_model_var.get())
            n_heads = int(self.n_heads_var.get())
            n_layers = int(self.n_layers_var.get())
            d_ff = int(self.d_ff_var.get())
            vocab_size = int(self.vocab_size_var.get())
            ctx = int(self.max_seq_var.get())
            bs = batch_size

            dtype_bytes = 2 if precision in ('bf16', 'fp16') else 4

            # Model parameters
            embed_params = vocab_size * d_model + ctx * d_model
            layer_params = 4 * d_model * d_model + 2 * d_model * d_ff + 4 * d_model
            total_params = embed_params + n_layers * layer_params

            # Memory components (in GB)
            model_gb = total_params * dtype_bytes / 1e9
            grad_gb = model_gb
            optimizer_gb = total_params * 8 / 1e9  # AdamW: 2x fp32 states

            # Activation memory - empirical formula
            # ~12 bytes per (batch * seq * d_model) per layer for bf16
            activation_gb = (bs * ctx * d_model * n_layers * 12) / 1e9

            # KV cache and attention intermediate
            kv_gb = (bs * ctx * d_model * n_layers * 4) / 1e9

            # Total with 20% buffer
            total_gb = (model_gb + grad_gb + optimizer_gb + activation_gb + kv_gb) * 1.2
            available_gb = gpu_vram_gb - 1.0  # Reserve 1GB for OS

            if total_gb > available_gb:
                # Calculate max batch
                fixed_cost = (model_gb + grad_gb + optimizer_gb) * 1.2
                per_batch = ((activation_gb + kv_gb) / bs) * 1.2
                max_batch_raw = (available_gb - fixed_cost) / per_batch
                max_batch = max(8, (int(max_batch_raw) // 8) * 8)

                errors.append(
                    f"VRAM needed: ~{total_gb:.1f}GB, available: ~{available_gb:.1f}GB\n"
                    f"   Model: {model_gb:.2f}GB | Grad: {grad_gb:.2f}GB | Optim: {optimizer_gb:.2f}GB\n"
                    f"   Activations: {activation_gb:.2f}GB | KV: {kv_gb:.2f}GB\n"
                    f"   → Reduce batch to {max_batch} or lower"
                )



            if epochs < 1:
                errors.append(f"Epochs must be at least 1")
            elif epochs > 100:
                warnings.append(f"Epochs ({epochs}) very high.")

            if stride < 1:
                errors.append(f"Stride must be at least 1")
            elif stride > context_length:
                errors.append(f"Stride ({stride}) > context ({context_length})")
            elif stride < context_length // 4:
                warnings.append(f"Stride ({stride}) very small. Many overlapping samples.")

            if warmup_ratio < 0 or warmup_ratio > 0.5:
                errors.append(f"Warmup ratio must be 0-0.5")

            if gradient_clip <= 0:
                errors.append(f"Gradient clip must be positive")
            elif gradient_clip > 10:
                warnings.append(f"Gradient clip ({gradient_clip}) high. Typical: 0.5-2.0")

            if val_split <= 0 or val_split >= 0.5:
                errors.append(f"Validation split must be 0-0.5")

            if early_stopping and patience < 1:
                errors.append(f"Patience must be at least 1")
            elif patience > epochs:
                warnings.append(f"Patience >= epochs. Early stopping won't trigger.")

            # Inside the training config try block, add:
            gradient_accumulation = int(self.grad_accum_var.get())

            if gradient_accumulation < 1:
                errors.append("Gradient accumulation must be at least 1")
            elif gradient_accumulation > 64:
                errors.append("Gradient accumulation too high (max 64)")

            # Then add to TrainingConfig:
            self.train_cfg = TrainingConfig(
                learning_rate=learning_rate,
                batch_size=batch_size,
                gradient_accumulation=gradient_accumulation,  # ADD THIS
                epochs=epochs,
                context_length=context_length,
                stride=stride,
                warmup_ratio=warmup_ratio,
                gradient_clip=gradient_clip,
                val_split=val_split,
                precision=precision,
                early_stopping=early_stopping,
                early_stopping_patience=patience,
            )

        except ValueError as e:
            errors.append(f"Training config: Invalid number format - {e}")

        # =================================================================
        # Checkpoint Config Validation
        # =================================================================
        try:
            output_dir = self.output_var.get()
            checkpoint_name = self.ckpt_name_var.get()
            save_steps = int(self.save_steps_var.get())
            resume_from = self.resume_var.get()

            if not output_dir:
                errors.append(f"Output directory cannot be empty")

            if not checkpoint_name:
                errors.append(f"Checkpoint name cannot be empty")
            elif not checkpoint_name.replace('_', '').replace('-', '').isalnum():
                errors.append(f"Checkpoint name: only letters, numbers, _, -")

            if save_steps < 0:
                errors.append(f"Save steps cannot be negative")

            if resume_from and not os.path.exists(resume_from):
                errors.append(f"Resume checkpoint not found: {resume_from}")

            self.ckpt_cfg = CheckpointConfig(
                output_dir=output_dir,
                checkpoint_name=checkpoint_name,
                save_every_steps=save_steps,
                resume_from=resume_from,
            )

        except ValueError as e:
            errors.append(f"Checkpoint config: Invalid number format - {e}")

        # =================================================================
        # Show warnings
        # =================================================================
        if warnings and not errors:
            warning_msg = "Warnings:\n\n• " + "\n• ".join(warnings)
            self._log(f"{Icons.WARNING} {len(warnings)} warning(s)")
            for w in warnings:
                self._log(f"   {Icons.BULLET} {w}")

            if not messagebox.askyesno("Warnings", f"{warning_msg}\n\nContinue?"):
                errors.append("Cancelled by user")

        return errors



    # ---------------------------------------------------#
    # Method name: _get_safe_checkpoint_path
    # ---------------------------------------------------#
    def _get_safe_checkpoint_path(self, directory: str, base_name: str, suffix: str = ".pt") -> str:
        """
        Generate a unique filename to prevent overwriting.
        Format: {base_name}.pt -> {base_name}_001.pt, etc.
        """
        filename = f"{base_name}{suffix}"
        path = os.path.join(directory, filename)
        
        if not os.path.exists(path):
            return path
            
        # Try finding a unique suffix
        for i in range(1, 1000):
            new_filename = f"{base_name}_{i:03d}{suffix}"
            new_path = os.path.join(directory, new_filename)
            if not os.path.exists(new_path):
                return new_path
                
        # If all taken (unlikely), overwrite the last one or return original
        return path

    # ---------------------------------------------------#
    # Method name: _start_training
    # ---------------------------------------------------#
    def _start_training(self):
        """Start the training process."""
        if not TORCH_AVAILABLE:
            messagebox.showerror("Error", "PyTorch is not available!")
            return

        if not self.processor.files:
            messagebox.showwarning("Warning", "Please scan a folder first!")
            return

        errors = self._get_configs()
        if errors:
            messagebox.showerror("Configuration Error", "\n".join(errors))
            return

        # Reset stop flag
        self._stop_requested = False

        # Switch to progress page
        self._show_page("progress")

        # Force GUI to update before starting thread
        self.root.update_idletasks()
        self.root.update()

        # Start training in background
        thread = threading.Thread(target=self._training_worker, daemon=True)
        thread.start()

        print(f"DEBUG: Training thread started: {thread.is_alive()}")

    def _training_worker(self):
        """Background training worker - Optimized & Resumable."""

        import numpy as np
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from numpy.lib.stride_tricks import as_strided

        eta_calc = ETACalculator()
        self._reset_all_steps()

        def should_stop() -> bool:
            return self._stop_requested

        def check_stop(stage: str) -> bool:
            if self._stop_requested:
                self._log(f"{Icons.STOP} Stopped during {stage}")
                self._set_step_status(stage, "error")
                self.app_state.state = AppState.IDLE
                return True
            return False

        def log(msg):
            self._log(msg)

        def update_progress(main_progress: float, main_msg: str, sub_progress: float = 0, sub_msg: str = ""):
            if self._stop_requested:
                return
            self.app_state.set_progress(main_progress, main_msg)
            self.app_state.set_sub_progress(sub_progress, sub_msg)

        try:
            # =================================================================
            # STAGE 1: Project Setup & Binary Cache Check
            # =================================================================
            self._set_step_status("setup", "running")
            self.app_state.state = AppState.PROCESSING

            proj_dir = self._get_project_dir()
            ckpt_dir = os.path.join(proj_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            self.ckpt_cfg.output_dir = ckpt_dir
            train_bin_path = os.path.join(proj_dir, "train.bin")
            val_bin_path = os.path.join(proj_dir, "val.bin")
            tok_path = os.path.join(proj_dir, "tokenizer.json")

            # Validate Resume Checkpoint consistency
            # If resuming, we MUST match the checkpoint's architecture, otherwise load_state_dict fails.
            if self.ckpt_cfg.resume_from and os.path.exists(self.ckpt_cfg.resume_from):
                try:
                    log(f"{Icons.INFO} Validating resume checkpoint...")
                    ckpt = torch.load(self.ckpt_cfg.resume_from, map_location='cpu')
                    
                    ckpt_config = ckpt.get('model_config', {})
                    ckpt_vocab = ckpt_config.get('vocab_size', 0)
                    
                    # Infer from weights if config missing
                    if not ckpt_vocab and 'model_state_dict' in ckpt:
                        w_emb = ckpt['model_state_dict'].get('tok_emb.weight')
                        if w_emb is not None:
                            ckpt_vocab = w_emb.shape[0]

                    if ckpt_vocab and ckpt_vocab != self.model_cfg.vocab_size:
                        log(f"{Icons.WARNING} Resume Checkpoint vocab ({ckpt_vocab}) != GUI Config ({self.model_cfg.vocab_size})")
                        log(f"{Icons.BOLT} Forcing config to match checkpoint to allow resume.")
                        self.model_cfg.vocab_size = ckpt_vocab
                    
                    del ckpt
                except Exception as e:
                    log(f"{Icons.WARNING} Failed to validate checkpoint: {e}")

            # Check if binary data exists
            data_ready = os.path.exists(train_bin_path) and os.path.exists(val_bin_path) and os.path.exists(tok_path)
            
            # Logic: Use cache ONLY if it exists AND matches the GUI vocab size.
            # "Vocab size in GUI should rule"
            reuse_cache = False

            if data_ready:
                try:
                    # Peek at existing tokenizer size
                    check_tok = BytePairTokenizer()
                    check_tok.load(tok_path)
                    cached_vocab = check_tok.get_vocab_size()

                    if cached_vocab == self.model_cfg.vocab_size:
                        # Sizes match. 
                        # Only reuse if NOT in incremental mode (incremental implies processing/adding files)
                        if not self.incremental_var.get():
                            reuse_cache = True
                        else:
                            log(f"{Icons.INFO} Incremental scan requested. Examining files...")
                    else:
                        # Mismatch
                        log(f"{Icons.WARNING} Cached data vocab ({cached_vocab}) != GUI Setting ({self.model_cfg.vocab_size})")
                        log(f"{Icons.BOLT} GUI setting rules. Will REDO tokenization...")
                        reuse_cache = False
                        
                except Exception as e:
                    log(f"{Icons.WARNING} Could not validate cache: {e}. Rebuilding...")
                    reuse_cache = False

            if reuse_cache:
                log(f"{Icons.BOLT} Found valid binary cache in project folder")
                log(f"{Icons.INFO} Skipping Read/Encode stages (Load from Disk)")

                # Load Tokenizer
                self.tokenizer = BytePairTokenizer(self.model_cfg.vocab_size)
                self.tokenizer.load(tok_path)

                # Verify binary file integrity roughly
                bin_dtype = np.uint16 if self.model_cfg.vocab_size < 65536 else np.int32
                if os.path.getsize(train_bin_path) < 100:
                     log(f"{Icons.WARNING} Cache file too small, rebuilding...")
                     goto_stage_5 = False
                else:
                    log(f"{Icons.SUCCESS} Vocab verified: {self.model_cfg.vocab_size}")
                    
                    # Mark skipped steps as done
                    self._set_step_status("setup", "done")
                    self._set_step_status("reading", "done") 
                    self._set_step_status("tokenizer", "done")
                    self._set_step_status("encoding", "done")
                    goto_stage_5 = True
            else:
                goto_stage_5 = False
                # Standard Setup for Fresh Run / Incremental
                selected_files = self._get_filtered_files()
                if not selected_files:
                    log(f"{Icons.ERROR} No files match filters!")
                    self._set_step_status("setup", "error")
                    self.app_state.state = AppState.ERROR
                    return

                total_files = len(selected_files)
                log(f"{Icons.INFO} Selected {total_files:,} files for processing")
                self._set_step_status("setup", "done")

            if check_stop("setup"):
                return

            if not goto_stage_5:
                # =================================================================
                # STAGE 2: Read Files
                # =================================================================
                self._set_step_status("reading", "running")
                log(f"{Icons.LOADING} Reading {total_files:,} files...")
                texts = []
                last_update = time.time()

                for i, f in enumerate(selected_files):
                    if self._stop_requested:
                        self.app_state.state = AppState.IDLE
                        return
                    try:
                        raw = Path(f['path']).read_bytes()
                        for enc in ['utf-8', 'latin-1', 'cp1252']:
                            try:
                                content = raw.decode(enc).replace('\r\n', '\n').replace('\r', '\n')
                                if content.strip():
                                    texts.append(content.strip())
                                break
                            except:
                                continue
                    except:
                        pass

                    now = time.time()
                    if now - last_update >= 0.25:
                        progress = (i + 1) / total_files
                        update_progress(progress * 0.10, "Reading", progress, f"{i + 1:,}/{total_files:,}")
                        last_update = now

                if not texts:
                    log(f"{Icons.ERROR} No valid text found")
                    self._set_step_status("reading", "error")
                    self.app_state.state = AppState.ERROR
                    return

                log(f"{Icons.SUCCESS} Read {len(texts):,} files")
                self._set_step_status("reading", "done")
                if check_stop("reading"):
                    return

                # =================================================================
                # STAGE 3: Train/Load Tokenizer
                # =================================================================
                self._set_step_status("tokenizer", "running")

                # If incremental, try load from resume checkpoint or project
                if self.incremental_var.get() and self.ckpt_cfg.resume_from:
                    resume_tok = os.path.join(os.path.dirname(self.ckpt_cfg.resume_from),
                                              f"{self.ckpt_name_var.get()}_tokenizer.json")
                    if os.path.exists(resume_tok):
                        tok_path = resume_tok  # Use resume tokenizer

                if os.path.exists(tok_path) and self.incremental_var.get():
                    log(f"{Icons.BOLT} Loading tokenizer...")
                    self.tokenizer = BytePairTokenizer(self.model_cfg.vocab_size)
                    self.tokenizer.load(tok_path)
                else:
                    log(f"{Icons.LOADING} Training new tokenizer...")
                    self.tokenizer = BytePairTokenizer(self.model_cfg.vocab_size)
                    self.tokenizer.train(texts)
                    self.tokenizer.save(tok_path)  # Save to project

                # Sync vocab after training/loading
                actual_vocab = self.tokenizer.get_vocab_size()
                if actual_vocab != self.model_cfg.vocab_size:
                    log(f"{Icons.WARNING} Vocab adjusted: {self.model_cfg.vocab_size} → {actual_vocab}")
                    self.model_cfg.vocab_size = actual_vocab

                log(f"{Icons.SUCCESS} Tokenizer ready: {self.model_cfg.vocab_size} tokens")
                self._set_step_status("tokenizer", "done")

                # =================================================================
                # STAGE 4: Encode & Cache Binary
                # =================================================================
                self._set_step_status("encoding", "running")
                log(f"{Icons.BOLT} Encoding {len(texts):,} files...")

                chunk_size = 5000
                total_texts = len(texts)
                token_arrays = []

                # Save tokenizer temp for workers
                temp_tok = "temp_tok.json"
                self.tokenizer.save(temp_tok)

                chunks = []
                for i in range(0, total_texts, chunk_size):
                    chunks.append((texts[i:i + chunk_size], temp_tok))

                max_workers = min(8, multiprocessing.cpu_count())
                encode_start = time.time()
                last_update = time.time()
                eta_calc.reset()

                try:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        for i, arr in enumerate(executor.map(_global_process_encode, chunks)):
                            if self._stop_requested:
                                self.app_state.state = AppState.IDLE
                                return
                            # Always use uint16 for vocab < 65535
                            if self.model_cfg.vocab_size < 65535:
                                token_arrays.append(arr.astype(np.uint16))
                            else:
                                token_arrays.append(arr)

                            now = time.time()
                            if now - last_update >= 0.5:
                                processed = min((i + 1) * chunk_size, total_texts)
                                progress = processed / total_texts
                                eta = eta_calc.get_eta(progress)
                                update_progress(0.2 + progress * 0.2, "Encoding", progress, f"ETA: {eta}")
                                last_update = now
                finally:
                    if os.path.exists(temp_tok):
                        try:
                            os.remove(temp_tok)
                        except:
                            pass

                log(f"{Icons.INFO} Concatenating & Caching Binary...")
                full_tokens = np.concatenate(token_arrays)
                del token_arrays, texts
                gc.collect()

                # Verify max token before saving
                max_token = int(full_tokens.max())
                if max_token >= self.model_cfg.vocab_size:
                    log(f"{Icons.WARNING} Max token {max_token} >= vocab {self.model_cfg.vocab_size}")
                    self.model_cfg.vocab_size = max_token + 1
                    log(f"{Icons.INFO} Vocab adjusted to {self.model_cfg.vocab_size}")

                # Split and Save Binary
                val_len = int(len(full_tokens) * self.train_cfg.val_split)
                train_len = len(full_tokens) - val_len

                train_data = full_tokens[:train_len]
                val_data = full_tokens[train_len:]

                train_data.tofile(train_bin_path)
                val_data.tofile(val_bin_path)

                log(f"{Icons.SUCCESS} Saved binary dataset: {len(full_tokens) / 1e6:.1f}M tokens")
                self._set_step_status("encoding", "done")

                # Cleanup RAM
                del full_tokens, train_data, val_data
                gc.collect()

            # =================================================================
            # STAGE 5: Load Data (Memory Mapped)
            # =================================================================
            self._set_step_status("samples", "running")
            log(f"{Icons.BOLT} Loading Memmap Dataset...")

            # Log current vocab for debug
            log(f"{Icons.INFO} Using vocab_size={self.model_cfg.vocab_size}")

            # Define Memmap Dataset Class
            class MemmapDataset(Dataset):
                def __init__(self, path, ctx, stride, vocab_size=65536):
                    dtype = np.uint16 if vocab_size < 65536 else np.int32
                    self.data = np.memmap(path, dtype=dtype, mode='r')
                    self.ctx = ctx
                    self.stride = stride
                    self.vocab_size = vocab_size
                    self.len = (len(self.data) - ctx) // stride

                def __len__(self):
                    return max(0, self.len)

                def __getitem__(self, idx):
                    start = idx * self.stride
                    x = torch.from_numpy(self.data[start: start + self.ctx].astype(np.int64))
                    y = torch.from_numpy(self.data[start + 1: start + self.ctx + 1].astype(np.int64))
                    return x, y

            ctx = self.train_cfg.context_length
            stride = self.train_cfg.stride

            # Initialize datasets pointing to binary files on disk
            try:
                train_ds = MemmapDataset(train_bin_path, ctx, stride, self.model_cfg.vocab_size)
                val_ds = MemmapDataset(val_bin_path, ctx, stride, self.model_cfg.vocab_size)
                log(f"{Icons.INFO} Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}")
            except Exception as e:
                log(f"{Icons.ERROR} Failed to load binary cache: {e}")
                self.app_state.state = AppState.ERROR
                return

            self._set_step_status("samples", "done")
            if check_stop("samples"):
                return

            # =================================================================
            # STAGE 6: DataLoaders
            # =================================================================
            self._set_step_status("loaders", "running")
            update_progress(0.60, "Creating loaders", 0, "")

            num_workers = 0

            train_loader = DataLoader(train_ds, batch_size=self.train_cfg.batch_size,
                                      shuffle=True, pin_memory=True, num_workers=num_workers)
            val_loader = DataLoader(val_ds, batch_size=self.train_cfg.batch_size,
                                    shuffle=False, pin_memory=True, num_workers=num_workers)

            log(f"{Icons.SUCCESS} DataLoaders ready")
            self._set_step_status("loaders", "done")
            if check_stop("loaders"):
                return

            # =================================================================
            # STAGE 7: Model Init
            # =================================================================
            self._set_step_status("model", "running")
            log(f"{Icons.LOADING} Initializing model...")
            log(f"{Icons.INFO} Model vocab_size={self.model_cfg.vocab_size}")

            # Configure Gradient Checkpointing if requested
            self.model_cfg.gradient_checkpointing = self.grad_ckpt_var.get()
            if self.model_cfg.gradient_checkpointing:
                log(f"{Icons.INFO} Gradient Checkpointing ENABLED (VRAM Saver)")

            self.model = GPTModel(self.model_cfg)
            n_params = self.model.count_parameters()
            log(f"{Icons.INFO} Model: {n_params / 1e6:.2f}M parameters")

            if CUDA_AVAILABLE:
                device = torch.device('cuda')
                log(f"{Icons.LOADING} Moving model to GPU...")
                torch.cuda.empty_cache()
                gc.collect()

                self.model = self.model.to(device)
                log(f"{Icons.SUCCESS} Model on GPU")
            else:
                device = torch.device('cpu')
                log(f"{Icons.WARNING} Using CPU")

            self._set_step_status("model", "done")
            if check_stop("model"):
                return

            # =================================================================
            # STAGE 8: Training Loop
            # =================================================================
            self._set_step_status("training", "running")
            self.app_state.state = AppState.TRAINING

            log(f"{Icons.LOADING} Setting up optimizer...")

            try:
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.train_cfg.learning_rate,
                    weight_decay=self.train_cfg.weight_decay,
                    fused=(device.type == 'cuda')
                )
                log(f"{Icons.INFO} Using fused AdamW")
            except TypeError:
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.train_cfg.learning_rate,
                    weight_decay=self.train_cfg.weight_decay
                )
                log(f"{Icons.INFO} Using standard AdamW")

            use_amp = self.train_cfg.precision != 'fp32' and device.type == 'cuda'
            amp_dtype = torch.bfloat16 if self.train_cfg.precision == 'bf16' else torch.float16
            scaler = GradScaler() if self.train_cfg.precision == 'fp16' else None

            # Accumulation Steps
            accum_steps = getattr(self.train_cfg, 'gradient_accumulation', 1)
            effective_batch = self.train_cfg.batch_size * accum_steps

            log(f"{Icons.INFO} AMP: {use_amp}, dtype: {amp_dtype}")
            log(f"{Icons.INFO} Accumulation: {accum_steps} (Effective Batch: {effective_batch})")

            update_progress(0.0, "Training...", 0, "")

            total_steps = (len(train_loader) // accum_steps) * self.train_cfg.epochs
            warmup_steps = int(total_steps * self.train_cfg.warmup_ratio)

            # --- Initialize ALL variables before resume ---
            step = 0
            start_epoch = 0
            epoch = 0
            best_loss = float('inf')
            train_losses = []
            val_losses = []
            patience = 0
            avg_loss = 0.0
            lr = self.train_cfg.learning_rate

            # --- RESUME LOGIC ---
            if self.ckpt_cfg.resume_from and os.path.exists(self.ckpt_cfg.resume_from):
                log(f"{Icons.LOAD} Loading checkpoint: {self.ckpt_cfg.resume_from}")
                try:
                    ckpt = torch.load(self.ckpt_cfg.resume_from, map_location=device)

                    sd = ckpt.get('model_state_dict', ckpt.get('model', None))
                    if sd:
                        # --- POSITION EMBEDDING SURGERY ---
                        # Check if context extension mode is active
                        do_surgery = getattr(self, '_context_extension_mode', False)
                        
                        if do_surgery:
                            ckpt_pos_emb = sd.get('pos_emb.weight', None)
                            if ckpt_pos_emb is not None:
                                ckpt_seq_len = ckpt_pos_emb.shape[0]
                                new_seq_len = self.model_cfg.max_seq_len
                                
                                if ckpt_seq_len != new_seq_len:
                                    if ckpt_seq_len < new_seq_len:
                                        # EXTENSION: Copy old positions, new ones stay random
                                        log(f"{Icons.BOLT} Position Embedding Surgery: {ckpt_seq_len} → {new_seq_len}")
                                        
                                        # Get dtype and device from checkpoint to ensure compatibility
                                        ckpt_dtype = ckpt_pos_emb.dtype
                                        ckpt_device = ckpt_pos_emb.device
                                        d_model = ckpt_pos_emb.shape[1]
                                        
                                        # Create new embeddings with matching dtype/device
                                        new_pos_emb = torch.randn(new_seq_len, d_model, dtype=ckpt_dtype, device=ckpt_device) * 0.02
                                        new_pos_emb[:ckpt_seq_len, :] = ckpt_pos_emb
                                        sd['pos_emb.weight'] = new_pos_emb
                                        log(f"   Copied positions 0-{ckpt_seq_len-1}, new positions {ckpt_seq_len}-{new_seq_len-1} randomly initialized")
                                    else:
                                        sd['pos_emb.weight'] = ckpt_pos_emb[:new_seq_len, :]
                            
                            self.model.load_state_dict(sd)

                    od = ckpt.get('optimizer_state_dict', ckpt.get('optimizer', None))
                    
                    # CRITICAL: Do NOT load optimizer if context extension was just performed
                    is_ctx_ext = getattr(self, '_context_extension_mode', False)
                    
                    if od and not is_ctx_ext:
                        # Only load optimizer if NOT doing context extension/surgery
                        try:
                            optimizer.load_state_dict(od)
                            log(f"{Icons.LOAD} Optimizer state loaded")
                        except Exception as e:
                             log(f"{Icons.WARNING} Could not load optimizer (ignore if fine-tuning): {e}")
                    elif is_ctx_ext:
                        log(f"{Icons.INFO} Context Extension: Optimizer state reset (Fresh Start)")

                    step = ckpt.get('step', 0)
                    saved_epoch = ckpt.get('epoch', 0)
                    best_loss = ckpt.get('best_loss', float('inf'))
                    train_losses = ckpt.get('train_losses', [])
                    val_losses = ckpt.get('val_losses', [])

                    # Check if this is a context extension (fresh start with model weights)
                    if getattr(self, '_ctx_ext_reset_counters', False):
                        log(f"{Icons.BOLT} Context extension: starting fresh training on extended context")
                        step = 0
                        start_epoch = 0
                        best_loss = float('inf')
                        train_losses = []
                        val_losses = []
                        self._ctx_ext_reset_counters = False
                        # Clear the mode flag explicitly now that we are done with setup
                        self._context_extension_mode = False 
                    else:
                        # Smart Resume Logic
                        steps_per_epoch = len(train_loader) // accum_steps
                        if steps_per_epoch > 0:
                            calculated_epoch = step // steps_per_epoch
                            if abs(calculated_epoch - saved_epoch) > 5:
                                log(f"{Icons.WARNING} Dataset changed significantly. Resetting epoch counter.")
                                start_epoch = 0
                                step = 0
                            else:
                                start_epoch = calculated_epoch
                                log(f"   Resuming from Epoch {start_epoch + 1}, Step {step}")
                        else:
                            start_epoch = 0

                except Exception as e:
                    log(f"{Icons.ERROR} Resume failed: {e}")

            if start_epoch >= self.train_cfg.epochs:
                log(f"{Icons.WARNING} Training already completed ({start_epoch} >= {self.train_cfg.epochs} epochs)")
                log(f"{Icons.INFO} Please increase 'Epochs' setting.")
                self.app_state.state = AppState.IDLE
                return

            start_time = time.time()
            session_start_step = step
            last_log_time = 0
            last_metrics_time = 0

            # Final vocab safety assertion
            _check_dtype = np.uint16 if self.model_cfg.vocab_size < 65536 else np.int32
            _check_data = np.memmap(train_bin_path, dtype=_check_dtype, mode='r')
            _data_max = int(_check_data.max())
            del _check_data
            if _data_max >= self.model_cfg.vocab_size:
                raise RuntimeError(
                    f"FATAL: Binary cache has token ID {_data_max} but model vocab is {self.model_cfg.vocab_size}. "
                    f"Delete the train.bin and val.bin files in your project folder, then restart training."
                )
            log(f"{Icons.CHECK} Vocab verified: max_token={_data_max} < vocab={self.model_cfg.vocab_size}")

            log(f"{Icons.ROCKET} Training started")
            log(f"   Steps/epoch: {len(train_loader) // accum_steps:,} | Total: {total_steps:,}")

            optimizer.zero_grad(set_to_none=True)

            for epoch in range(start_epoch, self.train_cfg.epochs):
                if self.app_state.state == AppState.IDLE:
                    break
                self.model.train()
                epoch_loss = 0.0
                epoch_steps = 0
                log(f"\n{Icons.ARROW_RIGHT} Epoch {epoch + 1}/{self.train_cfg.epochs}")

                # Calculate start batch
                start_batch = 0
                if epoch == start_epoch:
                    start_batch = (step * accum_steps) % len(train_loader)
                    if start_batch > 0:
                        log(f"   Skipping {start_batch} batches...")

                for batch_idx, (x, y) in enumerate(train_loader):
                    if epoch == start_epoch and batch_idx < start_batch:
                        continue
                    if self.app_state.state == AppState.IDLE:
                        break
                    while self.app_state.state == AppState.PAUSED:
                        time.sleep(0.1)

                    x = x.to(device, non_blocking=True).long()
                    y = y.to(device, non_blocking=True).long()

                    # LR Schedule
                    if (batch_idx + 1) % accum_steps == 0:
                        if step < warmup_steps:
                            lr = self.train_cfg.learning_rate * (step + 1) / warmup_steps
                        else:
                            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                            lr = self.train_cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * prog))
                        for pg in optimizer.param_groups:
                            pg['lr'] = lr

                    # Forward
                    try:
                        if use_amp:
                            with autocast(device_type='cuda', dtype=amp_dtype):
                                logits = self.model(x)
                                loss = F.cross_entropy(logits.view(-1, self.model_cfg.vocab_size), y.view(-1))
                                loss = loss / accum_steps

                            if scaler:
                                scaler.scale(loss).backward()
                            else:
                                loss.backward()
                        else:
                            logits = self.model(x)
                            loss = F.cross_entropy(logits.view(-1, self.model_cfg.vocab_size), y.view(-1))
                            loss = loss / accum_steps
                            loss.backward()
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            torch.cuda.empty_cache()
                            raise RuntimeError(f"OOM - reduce batch size")
                        raise

                    # Step
                    if (batch_idx + 1) % accum_steps == 0:
                        if scaler:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.gradient_clip)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.gradient_clip)
                            optimizer.step()

                        optimizer.zero_grad(set_to_none=True)
                        step += 1

                        loss_val = loss.item() * accum_steps
                        epoch_loss += loss_val
                        epoch_steps += 1

                        self.app_state.set_progress(step / total_steps, f"Epoch {epoch + 1}/{self.train_cfg.epochs}")
                        self.app_state.set_sub_progress((batch_idx + 1) / len(train_loader),
                                                        f"Step {step} | Loss: {loss_val:.4f}")

                        now = time.time()

                        # Metrics
                        if now - last_metrics_time >= 2.0:
                            elapsed = now - start_time
                            sess_steps = step - session_start_step
                            sps = sess_steps / elapsed if elapsed > 10 else 0
                            eta = (total_steps - step) / sps if sps > 0 else 0

                            vram = torch.cuda.memory_reserved(0) / 1e9 if CUDA_AVAILABLE else 0
                            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9 if CUDA_AVAILABLE else 1
                            metrics = {
                                'epoch': epoch + 1,
                                'total_epochs': self.train_cfg.epochs,
                                'step': batch_idx + 1,
                                'total_steps': len(train_loader),
                                'global_step': step,
                                'total_global_steps': total_steps,
                                'train_loss': loss_val,
                                'val_loss': val_losses[-1] if val_losses else 0,
                                'best_loss': best_loss if best_loss != float('inf') else 0,
                                'lr': lr,
                                'steps_per_sec': sps,
                                'tokens_per_sec': sps * effective_batch * ctx,
                                'elapsed': elapsed,
                                'eta': eta,
                                'gpu_mem': vram,
                                'gpu_total': vram_total,
                                'perplexity': math.exp(min(loss_val, 20))
                            }
                            self._update_metrics_display(metrics)
                            last_metrics_time = now

                        # Log
                        if now - last_log_time >= 30.0:
                            elapsed = now - start_time
                            sess_steps = step - session_start_step
                            sps = sess_steps / elapsed if elapsed > 10 else 0
                            eta_str = str(timedelta(seconds=int((total_steps - step) / max(0.1, sps))))
                            log(f"   Step {step:,} | Loss: {loss_val:.4f} | LR: {lr:.2e} | ETA: {eta_str}")
                            last_log_time = now

                        # Manual Save Request
                        if self._save_requested:
                            self._save_requested = False
                            ckpt_path = os.path.join(self.ckpt_cfg.output_dir,
                                                     f"{self.ckpt_cfg.checkpoint_name}_manual_{step}.pt")
                            torch.save({
                                'epoch': epoch,
                                'step': step,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss_val,
                                'best_loss': best_loss,
                                'train_losses': train_losses,
                                'val_losses': val_losses,
                                'model_cfg': asdict(self.model_cfg),
                                'train_cfg': asdict(self.train_cfg)
                            }, ckpt_path)
                            log(f"   {Icons.SAVE} Saved: manual_{step}")

                        # Step Checkpoint
                        if self.ckpt_cfg.save_every_steps > 0 and step % self.ckpt_cfg.save_every_steps == 0:
                            ckpt_path = os.path.join(self.ckpt_cfg.output_dir,
                                                     f"{self.ckpt_cfg.checkpoint_name}_step_{step}.pt")
                            torch.save({
                                'epoch': epoch,
                                'step': step,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss_val,
                                'best_loss': best_loss,
                                'train_losses': train_losses,
                                'val_losses': val_losses,
                                'model_cfg': asdict(self.model_cfg),
                                'train_cfg': asdict(self.train_cfg)
                            }, ckpt_path)
                            log(f"   {Icons.SAVE} Saved: step_{step}")

                # Check stop BEFORE validation - prevents race condition on restart
                if self.app_state.state == AppState.IDLE:
                    log(f"{Icons.INFO} Skipping validation (stopped)")
                    break

                # End of Epoch - Validation
                avg_loss = epoch_loss / max(1, epoch_steps)
                train_losses.append(avg_loss)

                self.model.eval()
                val_loss = 0.0
                val_count = 0
                log(f"{Icons.TEST} Validating...")
                
                # Dynamic validation limit
                val_percent = getattr(self.train_cfg, 'validation_percent', 20.0)
                total_val_batches = len(val_loader)
                
                if val_percent > 0:
                    max_val_batches = max(1, int(total_val_batches * (val_percent / 100.0)))
                    log(f"   Using {max_val_batches}/{total_val_batches} batches ({val_percent}%)")

                    with torch.no_grad():
                        for batch_idx, (vx, vy) in enumerate(val_loader):
                            if batch_idx >= max_val_batches:
                                break
                            # Check stop inside validation loop
                            if self.app_state.state == AppState.IDLE:
                                break
                            vx = vx.to(device).long()
                            vy = vy.to(device).long()
                            if use_amp:
                                with autocast(device_type='cuda', dtype=amp_dtype):
                                    vlogits = self.model(vx)
                                    vloss = F.cross_entropy(vlogits.view(-1, vlogits.size(-1)), vy.view(-1))
                            else:
                                vlogits = self.model(vx)
                                vloss = F.cross_entropy(vlogits.view(-1, vlogits.size(-1)), vy.view(-1))
                            val_loss += vloss.item()
                            val_count += 1
                else:
                    log("   Validation skipped (Disabled)")

                # Process results if validation ran
                if val_count > 0:
                    val_loss /= val_count
                    val_losses.append(val_loss)
                    log(f"   Train: {avg_loss:.4f} | Val: {val_loss:.4f}")

                    # Best Model Check
                    if val_loss < best_loss:
                        best_loss = val_loss
                        patience = 0
                        ckpt_path = os.path.join(self.ckpt_cfg.output_dir,
                                                 f"{self.ckpt_cfg.checkpoint_name}_best.pt")
                        torch.save({
                            'epoch': epoch,
                            'step': step,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': val_loss,
                            'best_loss': best_loss,
                            'train_losses': train_losses,
                            'val_losses': val_losses,
                            'model_cfg': asdict(self.model_cfg),
                            'train_cfg': asdict(self.train_cfg)
                        }, ckpt_path)
                        log(f"   {Icons.SUCCESS} New best: {val_loss:.4f}")
                    else:
                        patience += 1
                else:
                    log(f"   Train: {avg_loss:.4f} | Val: N/A")

                # Epoch Checkpoint
                ckpt_path = os.path.join(self.ckpt_cfg.output_dir,
                                         f"{self.ckpt_cfg.checkpoint_name}_epoch_{epoch + 1}.pt")
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'best_loss': best_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'model_cfg': asdict(self.model_cfg),
                    'train_cfg': asdict(self.train_cfg)
                }, ckpt_path)

                # Early Stopping Check (Only if validation ran)
                if val_count > 0 and self.train_cfg.early_stopping and patience >= self.train_cfg.early_stopping_patience:
                    log(f"{Icons.WARNING} Early stopping triggered")
                    break


            # --- Final Save ---
            # Skip if stopped or model was cleared
            if self.app_state.state == AppState.IDLE or self.model is None:
                log(f"{Icons.INFO} Training stopped (skipping final save)")
                return

            self._set_step_status("saving", "running")
            
            # Versioning: Check if final checkpoint exists
            base_final_name = f"{self.ckpt_cfg.checkpoint_name}_final"
            ckpt_path = self._get_safe_checkpoint_path(self.ckpt_cfg.output_dir, base_final_name, ".pt")
            
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'model_cfg': asdict(self.model_cfg),
                'train_cfg': asdict(self.train_cfg)
            }, ckpt_path)

            # Save tokenizer to checkpoint folder
            self.tokenizer.save(
                os.path.join(self.ckpt_cfg.output_dir, f"{self.ckpt_cfg.checkpoint_name}_tokenizer.json"))

            self._set_step_status("training", "done")
            self._set_step_status("saving", "done")

            total_time = time.time() - start_time
            log(f"\n{Icons.SPARKLE} Training complete!")
            log(f"   Total time: {timedelta(seconds=int(total_time))}")
            log(f"   Best Val Loss: {best_loss:.4f}")
            log(f"   Final checkpoint: {ckpt_path}")

            self.app_state.state = AppState.IDLE

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(tb) # Print to console/stderr for debugging
            
            # Show friendly error to user
            clean_err = str(e)
            if "out of memory" in clean_err.lower():
                msg = "GPU Out of Memory.\nTry creating a smaller model or reducing batch size."
            else:
                msg = f"Assessment: {clean_err}\n\nPlease check your settings and data."

            self.root.after(0, lambda: messagebox.showerror("Training Error", f"Training Failed:\n\n{msg}"))
            
            log(f"{Icons.ERROR} Training failed. See console for details.")
            self.app_state.state = AppState.ERROR
            self._set_step_status("training", "error")



    # ---------------------------------------------------#
    # Method name: _pause_training
    # ---------------------------------------------------#
    def _pause_training(self):
        if self.trainer:
            self.trainer.pause()

    # ---------------------------------------------------#
    # Method name: _stop_training
    # ---------------------------------------------------#
    def _stop_training(self):
        """Stop any running operation."""
        self._stop_requested = True

        if self.trainer:
            self.trainer.stop()

        # Clear CUDA state
        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self.app_state.state = AppState.IDLE
        self.app_state.set_progress(0, "")
        self.app_state.set_sub_progress(0, "")

        # Note: Do NOT set self.model = None here - worker thread may still be using it
        # Model will be replaced on next training run; memory freed by gc
        self.trainer = None
        gc.collect()

        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            torch.cuda.empty_cache()

        def _force_stopped_ui():
            try:
                self.progress_status_label.config(text=f"{Icons.STOP} Stopped")
                self.progress_eta_label.config(text="")
                self.overall_progress_bar['value'] = 0
                self.epoch_progress_bar['value'] = 0
            except tk.TclError:
                pass

        self.root.after_idle(_force_stopped_ui)
        self._log(f"{Icons.STOP} Stopped")

    # ---------------------------------------------------#
    # Method name: _save_now
    # ---------------------------------------------------#
    def _save_now(self):
        """Request a manual save."""
        # Allow save in TRAINING or PAUSED state
        if self.app_state.state in (AppState.TRAINING, AppState.PAUSED):
            self._save_requested = True
            self._log(f"{Icons.WAIT} Save requested... waiting for next step")
        else:
            self._show_toast("Training is not running", "info")

    # ---------------------------------------------------#
    # Method name: _plot_loss
    # ---------------------------------------------------#
    def _plot_loss(self):
        """Plot training curves."""
        if not self.trainer or not self.trainer.train_losses:
            self._show_toast("No training data to plot", "info")
            return

        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6), facecolor=NavyTheme.NAVY_DARK)
            ax.set_facecolor(NavyTheme.NAVY_MEDIUM)

            epochs = range(1, len(self.trainer.train_losses) + 1)

            ax.plot(epochs, self.trainer.train_losses, 'o-',
                    color=NavyTheme.ACCENT_BLUE, label='Train Loss', linewidth=2, markersize=6)
            ax.plot(epochs, self.trainer.val_losses, 's-',
                    color=NavyTheme.SUCCESS, label='Val Loss', linewidth=2, markersize=6)

            ax.set_xlabel('Epoch', color=NavyTheme.TEXT_PRIMARY, fontsize=12)
            ax.set_ylabel('Loss', color=NavyTheme.TEXT_PRIMARY, fontsize=12)
            ax.set_title('Training Progress', color=NavyTheme.TEXT_PRIMARY, fontsize=14, fontweight='bold')

            ax.tick_params(colors=NavyTheme.TEXT_SECONDARY)
            ax.spines['bottom'].set_color(NavyTheme.BORDER_LIGHT)
            ax.spines['top'].set_color(NavyTheme.BORDER_LIGHT)
            ax.spines['left'].set_color(NavyTheme.BORDER_LIGHT)
            ax.spines['right'].set_color(NavyTheme.BORDER_LIGHT)

            ax.legend(facecolor=NavyTheme.NAVY_DARK, edgecolor=NavyTheme.BORDER_LIGHT,
                      labelcolor=NavyTheme.TEXT_PRIMARY)
            ax.grid(True, alpha=0.3, color=NavyTheme.BORDER_LIGHT)

            plt.tight_layout()
            plt.show()

        except ImportError:
            messagebox.showerror("Error", "matplotlib not installed.\nRun: pip install matplotlib")
        except Exception as e:
            messagebox.showerror("Error", f"Plot failed: {e}")



    def _test_model(self):
        """Open improved model testing window with auto-continue and scenario support."""
        test_win = tk.Toplevel(self.root)
        test_win.title(f"{Icons.TEST} Story Generator")
        test_win.geometry("1100x950")
        NavyTheme.apply(test_win)

        # State variables
        stop_auto = [False]
        is_first_generation = [True]
        current_scenario = [""]

        # --- Top Control Bar ---
        top_bar = ttk.Frame(test_win)
        top_bar.pack(fill="x", padx=20, pady=10)

        status_var = tk.StringVar()
        if self.model and self.tokenizer:
            p = sum(x.numel() for x in self.model.parameters()) / 1e6
            status_var.set(f"{Icons.CHECK} Loaded: {self.ckpt_name_var.get()} ({p:.1f}M)")
        else:
            status_var.set(f"{Icons.WARNING} No Model Loaded")

        ttk.Label(top_bar, textvariable=status_var, font=("Segoe UI", 10, "bold")).pack(side="left")

        def find_tokenizer(model_path):
            """Find tokenizer in order: parent folder, checkpoint folder, ask user."""
            model_dir = os.path.dirname(model_path)
            parent_dir = os.path.dirname(model_dir)
            model_name = os.path.basename(model_path).replace(".pt", "")

            # Extract base name (remove _best, _final, _step_X, _epoch_X)
            import re
            base_name = re.sub(r'_(best|final|step_\d+|epoch_\d+)$', '', model_name)

            # Search locations
            search_paths = [
                # Parent folder (where bin files are)
                os.path.join(parent_dir, "tokenizer.json"),
                os.path.join(parent_dir, f"{base_name}_tokenizer.json"),
                # Checkpoint folder
                os.path.join(model_dir, "tokenizer.json"),
                os.path.join(model_dir, f"{base_name}_tokenizer.json"),
                os.path.join(model_dir, f"{model_name}_tokenizer.json"),
            ]

            for path in search_paths:
                if os.path.exists(path):
                    return path

            # Ask user
            messagebox.showinfo("Tokenizer Not Found",
                                f"Could not find tokenizer in:\n• {parent_dir}\n• {model_dir}\n\nPlease select manually.")
            path = filedialog.askopenfilename(
                title="Select Tokenizer",
                filetypes=[("Tokenizer", "*.json")],
                initialdir=parent_dir
            )
            return path if path else None

        def load_file():
            path = filedialog.askopenfilename(filetypes=[("Model", "*.pt")])
            if not path:
                return
            try:
                d = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
                ckpt = torch.load(path, map_location=d)

                cfg = self.model_cfg
                if 'model_cfg' in ckpt:
                    cfg = ModelConfig(**ckpt['model_cfg'])

                self.model = GPTModel(cfg)
                self.model.load_state_dict(ckpt.get('model_state_dict', ckpt.get('model')))
                self.model.to(d).eval()
                self.model_cfg = cfg

                # Find tokenizer
                tok_path = find_tokenizer(path)
                if tok_path and os.path.exists(tok_path):
                    self.tokenizer = BytePairTokenizer(cfg.vocab_size)
                    self.tokenizer.load(tok_path)
                    status_var.set(f"{Icons.CHECK} Loaded: {os.path.basename(path)}")
                else:
                    status_var.set(f"{Icons.WARNING} Model loaded, no tokenizer")

            except Exception as e:
                messagebox.showerror("Error", str(e))

        ttk.Button(top_bar, text=f"{Icons.LOAD} Load Model", command=load_file).pack(side="right")

        # --- Settings ---
        set_frame = ttk.LabelFrame(test_win, text="Generation Settings", padding=10)
        set_frame.pack(fill="x", padx=20, pady=(0, 10))

        row1 = ttk.Frame(set_frame)
        row1.pack(fill="x", pady=(0, 5))

        # Temperature
        ttk.Label(row1, text="Temp:").pack(side="left")
        temp_var = tk.DoubleVar(value=0.85)
        temp_val = ttk.Label(row1, text="0.85", width=4)
        temp_val.pack(side="left")

        def update_temp(v):
            temp_val.config(text=f"{float(v):.2f}")

        ttk.Scale(row1, from_=0.1, to=1.5, variable=temp_var, length=100, command=update_temp).pack(side="left", padx=5)

        # Top-K
        ttk.Label(row1, text="Top-K:").pack(side="left", padx=(15, 0))
        topk_var = tk.IntVar(value=50)
        topk_val = ttk.Label(row1, text="50", width=4)
        topk_val.pack(side="left")

        def update_topk(v):
            topk_val.config(text=f"{int(float(v))}")

        ttk.Scale(row1, from_=1, to=100, variable=topk_var, length=100, command=update_topk).pack(side="left", padx=5)

        # Max Tokens
        ttk.Label(row1, text="Max Tokens:").pack(side="left", padx=(15, 0))
        len_var = tk.StringVar(value="200")
        ttk.Entry(row1, textvariable=len_var, width=6).pack(side="left", padx=5)

        # Repetition Penalty
        ttk.Label(row1, text="Rep Penalty:").pack(side="left", padx=(15, 0))
        rep_var = tk.StringVar(value="1.15")
        ttk.Entry(row1, textvariable=rep_var, width=5).pack(side="left", padx=5)

        # Row 2: Auto-continue settings
        row2 = ttk.Frame(set_frame)
        row2.pack(fill="x", pady=(5, 0))

        ttk.Label(row2, text="Auto-Continue Iterations:").pack(side="left")
        iter_var = tk.StringVar(value="5")
        ttk.Entry(row2, textvariable=iter_var, width=5).pack(side="left", padx=5)

        ttk.Label(row2, text="Delay (sec):").pack(side="left", padx=(15, 0))
        delay_var = tk.StringVar(value="0.5")
        ttk.Entry(row2, textvariable=delay_var, width=5).pack(side="left", padx=5)

        # --- Scenario ---
        scenario_frame = ttk.LabelFrame(test_win, text=f"{Icons.SPARKLE} Scenario (Sets the tone for your story)",
                                        padding=10)
        scenario_frame.pack(fill="x", padx=20, pady=(0, 10))

        scenario_txt = scrolledtext.ScrolledText(
            scenario_frame, height=1, bg=NavyTheme.BG_INPUT, fg=NavyTheme.ACCENT_CYAN,
            font=("Cascadia Code", 10), insertbackground="white"
        )
        scenario_txt.pack(fill="x")
        scenario_txt.insert("1.0", "A dark fantasy tale about a wandering knight haunted by his past.")

        # --- Starting Prompt ---
        prompt_frame = ttk.LabelFrame(test_win, text=f"{Icons.ARROW_RIGHT} Starting Prompt", padding=10)
        prompt_frame.pack(fill="x", padx=20, pady=(0, 10))

        prompt_txt = scrolledtext.ScrolledText(
            prompt_frame, height=1, bg=NavyTheme.BG_INPUT, fg="white",
            font=("Cascadia Code", 11), insertbackground="white"
        )
        prompt_txt.pack(fill="x")
        prompt_txt.insert("1.0", "The knight walked alone through the mist, his armor silent.")

        # --- Generated Story ---
        story_frame = ttk.LabelFrame(test_win, text=f"{Icons.FILE} Generated Story", padding=10)
        story_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        story_txt = scrolledtext.ScrolledText(
            story_frame,  height=10, bg=NavyTheme.BG_INPUT, fg="white",
            font=("Cascadia Code", 11), insertbackground="white", wrap="word"
        )
        story_txt.pack(fill="both", expand=True)

        # Stats bar
        stats_frame = ttk.Frame(story_frame)
        stats_frame.pack(fill="x", pady=(5, 0))

        word_count_var = tk.StringVar(value="Words: 0 | Tokens: 0")
        ttk.Label(stats_frame, textvariable=word_count_var, style="Dim.TLabel").pack(side="left")

        status_lbl = ttk.Label(stats_frame, text="Ready", style="Dim.TLabel")
        status_lbl.pack(side="right")

        def get_last_sentence(text):
            """Extract last complete sentence from text."""
            text = text.strip()
            if not text:
                return ""

            import re
            # Find last sentence ending
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) >= 1:
                # Return last 1-2 sentences for better context
                return ' '.join(sentences[-2:]) if len(sentences) >= 2 else sentences[-1]

            # Fallback: last 150 chars
            return text[-150:]

        def update_stats():
            text = story_txt.get("1.0", "end-1c")
            words = len(text.split())
            tokens = len(self.tokenizer.encode(text)) if self.tokenizer and text else 0
            word_count_var.set(f"Words: {words:,} | Tokens: {tokens:,}")

        def generate_once(prompt_text, use_scenario=False):
            """Single generation. Returns generated text (after prompt) or None."""
            if not self.model or not self.tokenizer:
                status_lbl.config(text="No model loaded!")
                return None

            try:
                # Build full prompt
                if use_scenario:
                    scenario = scenario_txt.get("1.0", "end-1c").strip()
                    if scenario:
                        full_prompt = f"[{scenario}]\n\n{prompt_text}"
                        current_scenario[0] = scenario
                    else:
                        full_prompt = prompt_text
                else:
                    full_prompt = prompt_text

                d = next(self.model.parameters()).device
                ids = torch.tensor([self.tokenizer.encode(full_prompt)], device=d)

                # Truncate if too long
                max_ctx = self.model_cfg.max_seq_len - int(len_var.get()) - 10
                if ids.size(1) > max_ctx:
                    ids = ids[:, -max_ctx:]

                with torch.no_grad():
                    out = self.model.generate(
                        ids,
                        max_tokens=int(len_var.get()),
                        temperature=temp_var.get(),
                        top_k=topk_var.get(),
                        repetition_penalty=float(rep_var.get())
                    )

                full_result = self.tokenizer.decode(out[0].tolist())

                # Extract only new generated part
                new_text = full_result[len(full_prompt):].strip()

                return new_text

            except Exception as e:
                status_lbl.config(text=f"Error: {e}")
                import traceback
                traceback.print_exc()
                return None

        def do_generate():
            """Single generate button click."""
            stop_auto[0] = True

            prompt = prompt_txt.get("1.0", "end-1c").strip()
            if not prompt:
                messagebox.showwarning("Warning", "Enter a prompt first")
                return

            gen_btn.config(state="disabled")
            cont_btn.config(state="disabled")
            auto_btn.config(state="disabled")
            status_lbl.config(text="Generating...")
            test_win.update()

            t0 = time.time()

            # First generation uses scenario
            new_text = generate_once(prompt, use_scenario=True)

            if new_text:
                # Clear and show prompt + new text
                story_txt.delete("1.0", "end")
                story_txt.insert("1.0", prompt + " " + new_text)
                story_txt.see("end")

                is_first_generation[0] = False

                dt = time.time() - t0
                status_lbl.config(text=f"Generated in {dt:.1f}s")
                update_stats()

            gen_btn.config(state="normal")
            cont_btn.config(state="normal")
            auto_btn.config(state="normal")

        def do_continue():
            """Continue from last sentence."""
            current_story = story_txt.get("1.0", "end-1c").strip()
            if not current_story:
                messagebox.showwarning("Warning", "Generate first, then continue")
                return

            cont_btn.config(state="disabled")
            gen_btn.config(state="disabled")
            auto_btn.config(state="disabled")
            status_lbl.config(text="Continuing...")
            test_win.update()

            t0 = time.time()

            # Get last sentence as prompt
            last_sent = get_last_sentence(current_story)

            # Generate without scenario (already established)
            new_text = generate_once(last_sent, use_scenario=False)

            if new_text:
                # Append to existing story
                story_txt.insert("end", " " + new_text)
                story_txt.see("end")

                dt = time.time() - t0
                status_lbl.config(text=f"Continued in {dt:.1f}s")
                update_stats()

            cont_btn.config(state="normal")
            gen_btn.config(state="normal")
            auto_btn.config(state="normal")

        def do_auto_continue():
            """Auto-continue loop."""
            current_story = story_txt.get("1.0", "end-1c").strip()

            # If no story yet, generate first
            if not current_story:
                do_generate()
                current_story = story_txt.get("1.0", "end-1c").strip()
                if not current_story:
                    return

            try:
                iterations = int(iter_var.get())
                delay = float(delay_var.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid iterations or delay value")
                return

            stop_auto[0] = False
            gen_btn.config(state="disabled")
            cont_btn.config(state="disabled")
            auto_btn.config(state="disabled")
            stop_btn.config(state="normal")

            def continue_loop(i):
                if stop_auto[0]:
                    finish_auto(i, "Stopped")
                    return

                if i >= iterations:
                    finish_auto(i, "Completed")
                    return

                status_lbl.config(text=f"Auto-continue {i + 1}/{iterations}...")
                test_win.update()

                # Get last sentence
                current_story = story_txt.get("1.0", "end-1c").strip()
                last_sent = get_last_sentence(current_story)

                # Generate
                new_text = generate_once(last_sent, use_scenario=False)

                if new_text:
                    story_txt.insert("end", " " + new_text)
                    story_txt.see("end")
                    update_stats()
                else:
                    finish_auto(i, "Generation failed")
                    return

                # Check stop flag after generation completes
                if stop_auto[0]:
                    finish_auto(i + 1, "Stopped")
                    return

                # Schedule next
                test_win.after(int(delay * 1000), lambda: continue_loop(i + 1))

            def finish_auto(count, reason):
                gen_btn.config(state="normal")
                cont_btn.config(state="normal")
                auto_btn.config(state="normal")
                stop_btn.config(state="disabled")
                status_lbl.config(text=f"{reason} after {count} iterations")

            continue_loop(0)

        def do_stop():
            """Stop auto-continue after current generation."""
            stop_auto[0] = True
            status_lbl.config(text="Stopping after current...")

        def do_clear():
            """Clear story and reset."""
            story_txt.delete("1.0", "end")
            is_first_generation[0] = True
            update_stats()
            status_lbl.config(text="Cleared")

        def do_copy():
            """Copy story to clipboard."""
            text = story_txt.get("1.0", "end-1c")
            test_win.clipboard_clear()
            test_win.clipboard_append(text)
            status_lbl.config(text="Copied to clipboard!")

        def do_save():
            """Save story to project folder."""
            text = story_txt.get("1.0", "end-1c").strip()
            if not text:
                messagebox.showwarning("Warning", "Nothing to save")
                return

            # Default save location
            proj_dir = self._get_project_dir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"story_{timestamp}.story"
            default_path = os.path.join(proj_dir, default_name)

            path = filedialog.asksaveasfilename(
                initialdir=proj_dir,
                initialfile=default_name,
                defaultextension=".story",
                filetypes=[("Story", "*.story"), ("Text", "*.txt"), ("All", "*.*")]
            )

            if path:
                try:
                    # Save with metadata
                    scenario = scenario_txt.get("1.0", "end-1c").strip()
                    prompt = prompt_txt.get("1.0", "end-1c").strip()

                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(f"[Scenario]\n{scenario}\n\n")
                        f.write(f"[Starting Prompt]\n{prompt}\n\n")
                        f.write(f"[Story]\n{text}\n")

                    status_lbl.config(text=f"Saved: {os.path.basename(path)}")
                except Exception as e:
                    messagebox.showerror("Error", f"Save failed: {e}")

        def do_load_story():
            """Load a saved story."""
            proj_dir = self._get_project_dir()

            path = filedialog.askopenfilename(
                initialdir=proj_dir,
                filetypes=[("Story", "*.story"), ("Text", "*.txt"), ("All", "*.*")]
            )

            if path:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Parse if .story format
                    if "[Scenario]" in content and "[Story]" in content:
                        import re

                        scenario_match = re.search(r'\[Scenario\]\n(.*?)\n\n', content, re.DOTALL)
                        prompt_match = re.search(r'\[Starting Prompt\]\n(.*?)\n\n', content, re.DOTALL)
                        story_match = re.search(r'\[Story\]\n(.*)', content, re.DOTALL)

                        if scenario_match:
                            scenario_txt.delete("1.0", "end")
                            scenario_txt.insert("1.0", scenario_match.group(1).strip())

                        if prompt_match:
                            prompt_txt.delete("1.0", "end")
                            prompt_txt.insert("1.0", prompt_match.group(1).strip())

                        if story_match:
                            story_txt.delete("1.0", "end")
                            story_txt.insert("1.0", story_match.group(1).strip())
                    else:
                        # Plain text
                        story_txt.delete("1.0", "end")
                        story_txt.insert("1.0", content)

                    is_first_generation[0] = False
                    update_stats()
                    status_lbl.config(text=f"Loaded: {os.path.basename(path)}")

                except Exception as e:
                    messagebox.showerror("Error", f"Load failed: {e}")

        # --- Buttons ---
        btn_frame = ttk.Frame(test_win)
        btn_frame.pack(fill="x", padx=20, pady=10)

        # Row 1: Main actions
        row1_btns = ttk.Frame(btn_frame)
        row1_btns.pack(fill="x", pady=(0, 5))

        gen_btn = ttk.Button(row1_btns, text=f"{Icons.ROCKET} Generate", style="Accent.TButton", command=do_generate)
        gen_btn.pack(side="left", padx=(0, 10))

        cont_btn = ttk.Button(row1_btns, text=f"{Icons.ARROW_RIGHT} Continue", command=do_continue)
        cont_btn.pack(side="left", padx=(0, 10))

        auto_btn = ttk.Button(row1_btns, text=f"{Icons.REFRESH} Auto-Continue", command=do_auto_continue)
        auto_btn.pack(side="left", padx=(0, 10))

        stop_btn = ttk.Button(row1_btns, text=f"{Icons.STOP} Stop", command=do_stop, state="disabled")
        stop_btn.pack(side="left", padx=(0, 10))

        # Row 2: File actions
        row2_btns = ttk.Frame(btn_frame)
        row2_btns.pack(fill="x")

        ttk.Button(row2_btns, text=f"{Icons.SAVE} Save", command=do_save).pack(side="left", padx=(0, 10))
        ttk.Button(row2_btns, text=f"{Icons.LOAD} Load", command=do_load_story).pack(side="left", padx=(0, 10))
        ttk.Button(row2_btns, text=f"{Icons.COPY} Copy", command=do_copy).pack(side="left", padx=(0, 10))
        ttk.Button(row2_btns, text=f"{Icons.CLEAR} Clear", command=do_clear).pack(side="left")

    # ========================================================================
    # SETTINGS PERSISTENCE
    # ========================================================================

    # Note: _save_settings removed - use sidebar's _save_project_settings instead

    # ---------------------------------------------------#
    # Method name: _load_settings
    # ---------------------------------------------------#
    def _load_settings(self):
        """Load settings from project folder. Handles both key formats."""
        # Try to load from current project name, else default
        proj_dir = self._get_project_dir()
        path = os.path.join(proj_dir, "settings.json")

        # If not in project dir, try root (legacy support)
        if not os.path.exists(path) and os.path.exists("trainer_settings.json"):
            path = "trainer_settings.json"

        if not os.path.exists(path):
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                s = json.load(f)

            # Helper to set if key exists (handles both formats)
            def set_var(var, key, type_cast=str):
                # Try without _var suffix first, then with _var suffix
                actual_key = key
                if key not in s and f"{key}_var" in s:
                    actual_key = f"{key}_var"
                elif key not in s:
                    return  # Key not in settings
                
                try:
                    var.set(type_cast(s[actual_key]))
                except Exception:
                    pass

            set_var(self.project_name_var, 'project_name')
            set_var(self.folder_var, 'folder')
            set_var(self.file_limit_var, 'file_limit')
            set_var(self.min_file_size_var, 'min_file_size')
            set_var(self.max_file_size_var, 'max_file_size')
            set_var(self.size_unit_var, 'size_unit')
            set_var(self.random_seed_var, 'random_seed', bool)
            set_var(self.seed_value_var, 'seed_value')

            set_var(self.d_model_var, 'd_model')
            set_var(self.n_heads_var, 'n_heads')
            set_var(self.n_layers_var, 'n_layers')
            set_var(self.d_ff_var, 'd_ff')
            set_var(self.vocab_size_var, 'vocab_size')
            set_var(self.max_seq_var, 'max_seq')
            set_var(self.dropout_var, 'dropout')
            set_var(self.grad_ckpt_var, 'grad_checkpointing', bool)

            set_var(self.lr_var, 'lr')
            set_var(self.batch_var, 'batch')
            set_var(self.epochs_var, 'epochs')
            set_var(self.stride_var, 'stride')
            set_var(self.warmup_var, 'warmup')
            set_var(self.grad_clip_var, 'grad_clip')
            set_var(self.val_split_var, 'val_split')
            set_var(self.precision_var, 'precision')
            set_var(self.early_stop_var, 'early_stop', bool)
            set_var(self.patience_var, 'patience')

            set_var(self.output_var, 'output')
            set_var(self.ckpt_name_var, 'ckpt_name')
            set_var(self.save_steps_var, 'save_steps')
            set_var(self.resume_var, 'resume')
            set_var(self.incremental_var, 'incremental', bool)

            self._log(f"{Icons.LOAD} Loaded settings from {path}")

            # Update estimates logic
            self._update_effective_files()
            self._update_model_size_estimate()

        except Exception as e:
            print(f"Failed to load settings: {e}")


    def _load_project_settings(self):
        """Show dialog to load project from sllm_projects.settings."""
        index_file = "sllm_projects.settings"
        if not os.path.exists(index_file):
            messagebox.showinfo("Info", "No projects found.\nSave a project first.")
            return

        # Parse index file
        projects = {}
        with open(index_file, 'r') as f:
            for line in f:
                line = line.strip()
                if " = " in line:
                    n, p = line.split(" = ", 1)
                    projects[n] = p

        if not projects:
            messagebox.showinfo("Info", "No projects found in index.")
            return

        # Dialog
        d = tk.Toplevel(self.root)
        d.title("Load Project")
        d.geometry("450x350")
        d.transient(self.root)
        d.grab_set()
        NavyTheme.apply(d)

        ttk.Label(d, text="Select Project:", font=("Segoe UI", 11, "bold")).pack(pady=(15, 10))

        # Listbox with scrollbar
        list_frame = ttk.Frame(d)
        list_frame.pack(fill="both", expand=True, padx=15, pady=5)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        lb = tk.Listbox(
            list_frame,
            bg=NavyTheme.BG_INPUT,
            fg=NavyTheme.TEXT_PRIMARY,
            font=("Segoe UI", 10),
            selectbackground=NavyTheme.ACCENT_BLUE,
            selectforeground="white",
            yscrollcommand=scrollbar.set
        )
        lb.pack(fill="both", expand=True)
        scrollbar.config(command=lb.yview)

        for p in sorted(projects.keys()):
            lb.insert("end", p)

        if projects:
            lb.selection_set(0)

        # Info label
        info_var = tk.StringVar(value="")
        info_label = ttk.Label(d, textvariable=info_var, style="Dim.TLabel")
        info_label.pack(pady=5)

        def on_select(event=None):
            sel = lb.curselection()
            if sel:
                name = lb.get(sel[0])
                path = projects[name]
                json_path = os.path.join(path, "settings.json")
                if os.path.exists(json_path):
                    info_var.set(f"📁 {path}")
                else:
                    info_var.set(f"⚠️ settings.json not found")

        lb.bind("<<ListboxSelect>>", on_select)
        lb.bind("<Double-Button-1>", lambda e: load_selected())

        def load_selected():
            sel = lb.curselection()
            if not sel:
                messagebox.showwarning("Warning", "Select a project first")
                return

            name = lb.get(sel[0])
            path = projects[name]

            if not os.path.exists(path):
                messagebox.showerror("Error", f"Project folder missing:\n{path}")
                return

            json_path = os.path.join(path, "settings.json")

            if not os.path.exists(json_path):
                messagebox.showerror("Error", f"settings.json not found in:\n{path}")
                return

            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                loaded_count = 0

                # Keys in JSON already have _var suffix, so use directly
                for key, value in data.items():
                    if hasattr(self, key):
                        var = getattr(self, key)

                        # Handle based on variable type
                        if isinstance(var, tk.BooleanVar):
                            var.set(bool(value))
                        elif isinstance(var, tk.StringVar):
                            var.set(str(value))
                        elif isinstance(var, tk.IntVar):
                            var.set(int(value))
                        elif isinstance(var, tk.DoubleVar):
                            var.set(float(value))
                        else:
                            var.set(value)

                        loaded_count += 1

                self._log(f"{Icons.LOAD} Loaded project '{name}' ({loaded_count} settings)")

                # Refresh UI estimates
                self._update_effective_files()
                self._update_model_size_estimate()

                d.destroy()
                messagebox.showinfo("Success", f"Project '{name}' loaded!\n{loaded_count} settings applied.")

            except json.JSONDecodeError as e:
                messagebox.showerror("Error", f"Invalid JSON:\n{e}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                messagebox.showerror("Error", f"Failed to load:\n{e}")

        # Buttons
        btn_frame = ttk.Frame(d)
        btn_frame.pack(fill="x", padx=15, pady=15)

        ttk.Button(btn_frame, text="Cancel", command=d.destroy).pack(side="right", padx=(10, 0))
        ttk.Button(btn_frame, text=f"{Icons.LOAD} Load Project", style="Accent.TButton",
                   command=load_selected).pack(side="right")

        on_select()

    def _save_project_settings(self):
        """Save settings and update sllm_projects.settings index."""
        name = self.project_name_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Project name required")
            return

        # 1. Determine Folder
        current_folder = None
        index_file = "sllm_projects.settings"

        if os.path.exists(index_file):
            with open(index_file, 'r') as f:
                for line in f:
                    if line.startswith(f"{name} = "):
                        current_folder = line.strip().split(" = ", 1)[1]
                        break

        # If new project, confirm creation
        if not current_folder:
            if not messagebox.askyesno("New Project", f"Create new project '{name}'?"):
                return
            current_folder = os.path.join("projects", name.replace(" ", "_"))

            # Append to index
            with open(index_file, 'a') as f:
                f.write(f"{name} = {current_folder}\n")

        os.makedirs(current_folder, exist_ok=True)

        # 2. Update Checkpoint Output to inside project folder
        ckpt_dir = os.path.join(current_folder, "checkpoints").replace("\\", "/")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.output_var.set(ckpt_dir)

        # Update live training config if running
        if hasattr(self, 'ckpt_cfg'):
            self.ckpt_cfg.output_dir = ckpt_dir

        # 3. Collect all _var attributes (ORIGINAL LOGIC)
        settings = {}
        path_keys = ['folder_var', 'output_var', 'resume_var']

        for attr in dir(self):
            if attr.endswith('_var'):
                var = getattr(self, attr)
                if isinstance(var, (tk.StringVar, tk.BooleanVar, tk.IntVar, tk.DoubleVar)):
                    value = var.get()

                    # Normalize paths
                    if attr in path_keys and isinstance(value, str) and value:
                        value = value.replace("\\", "/")

                    settings[attr] = value

        try:
            json_path = os.path.join(current_folder, "settings.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, sort_keys=True)
            self._log(f"{Icons.SAVE} Saved project '{name}' ({len(settings)} settings)")
            messagebox.showinfo("Success", f"Project '{name}' saved!\n{len(settings)} settings saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    def _on_close(self):
        """Handle application close."""
        if self.app_state.state == AppState.TRAINING:
            if not messagebox.askyesno("Quit", "Training is running. Stop and quit?"):
                return
            self._stop_training()
            
        self.root.destroy()
        sys.exit(0)

########################## END OF CLASS LLMTrainerGUI ################################


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

# ---------------------------------------------------#
# Function name: main
# ---------------------------------------------------#
def main():
    """Main entry point."""
    print("=" * 60)
    print(f"  {Icons.ROCKET} Python sLLM Trainer v1.0 - Professional Edition")
    print("=" * 60)
    import multiprocessing
    multiprocessing.freeze_support()

    if not TORCH_AVAILABLE:
        print(f"\n{Icons.WARNING} PyTorch not found. Training will not work.")
        print("Install with: pip install torch")

    root = tk.Tk()
    app = LLMTrainerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
