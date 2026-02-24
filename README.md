Python LLM Trainer v4.0 - Professional Navy Edition
A production-grade desktop application for training custom transformer language models on code and text datasets with a modern navy-themed GUI.
Features
Core Capabilities

Transformer Architecture: Fully configurable decoder-only transformer with multi-head attention
Smart Tokenization: BPE tokenizer with fallback to simple tokenization
GPU Acceleration: CUDA support with mixed precision training (BF16/FP16/FP32)
Project Management: Multi-project workspace with persistent settings
Incremental Training: Resume from checkpoints, add data without retraining tokenizer
Advanced Data Pipeline: Multi-threaded scanning, encoding, and preprocessing

<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/5c1e558b-de2f-4650-8e38-602a24b5e29f" />


Training Features

Auto-Scaling Presets: GPU-based (8GB/12GB/16GB/24GB), dataset-size based, quality-based
Smart Validation: Configurable validation split with early stopping
Context Extension: Extend model context window with incremental training
DPO/SFT Support: Create preference pairs and supervised fine-tuning datasets
Live Monitoring: Real-time loss curves, GPU/CPU/RAM usage, ETA tracking

<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/50b1f466-f64d-444b-8483-e6c8b442d25d" />


Data Refinement

Quality Scoring: Multi-metric quality assessment (readability, entropy, structure)
Deduplication: Exact hash-based + near-duplicate detection (Jaccard similarity)
Text Cleaning: Specialized cleaners for fiction, code, and documentation
Garbage Detection: Auto-identify minified files, boilerplate, auto-generated code
File Filtering: Size limits, extension filtering, recursive scanning

<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/42af7f18-f75e-4955-997d-bfcc17a93125" />


GUI Features

Professional Theme: Navy blue design with icons, tooltips, and status indicators
Non-Blocking Operations: All heavy tasks run in background threads
Toast Notifications: Non-intrusive feedback system
Hardware Monitor: Real-time CPU/RAM/VRAM graphs in sidebar
Model Tester: Interactive text generation with temperature/top-k/top-p controls

<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/f5363822-c914-4e62-aa24-e6326ebaf84b" />
<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/efc7913b-a0d8-46e9-8a9a-b6d0e36e7bbc" />
<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/1aa7b93c-cdcc-4db7-a53e-9c6aa1b575bb" />

Inbuilt Comprehensive Help
<img width="602" height="732" alt="image" src="https://github.com/user-attachments/assets/2b2f83fa-0559-4ad2-916a-43dd0499ac5e" />


System Requirements
Minimum

Python 3.8+
8GB System RAM
GPU: 6GB VRAM (NVIDIA RTX 2060 or equivalent)
Storage: 10GB+ for checkpoints

Recommended

Python 3.10+
16GB System RAM
GPU: 12GB+ VRAM (RTX 3060/4060 Ti or better)
Storage: 50GB+ SSD for datasets

Dependencies
bash# Required
torch>=2.0.0
psutil

# Optional (for fast tokenization)
tokenizers
Installation
bash# Clone repository
git clone https://github.com/yourusername/llm-trainer.git
cd llm-trainer

# Install dependencies
pip install torch psutil tokenizers

# Run
python llm_trainer_v6.py
```

## Quick Start

1. **Create Project**: Enter project name in sidebar, click Save Project
2. **Select Data**: Browse to folder with .py/.txt files
3. **Choose Preset**: Click Auto-Select or pick GPU/dataset size preset
4. **Train**: Click Start Training

## Model Architectures

### Presets

| Preset | d_model | Layers | Heads | Params | VRAM | Best For |
|--------|---------|--------|-------|--------|------|----------|
| Fast Experiment | 256 | 4 | 4 | ~25M | 4GB | Quick tests |
| Balanced | 512 | 6 | 8 | ~110M | 8GB | General use |
| High Quality | 768 | 12 | 12 | ~200M | 12GB | Production |
| GPU 24GB | 1024 | 16 | 16 | ~350M | 20GB | Large datasets |

### Dataset Scaling

- **Small (<50k files)**: Higher epochs (20), higher dropout (0.15), smaller model
- **Medium (50-200k)**: Balanced configuration
- **Large (200-500k)**: Larger model, fewer epochs (4)
- **XLarge (500k+)**: Maximum capacity, minimal epochs (3)

## Training Strategy

### New Project
```
1. Scan folder → processes all files
2. Train tokenizer → vocabulary generation
3. Encode data → converts to token IDs
4. Train model → transformer training
```

### Incremental Training
```
1. Load checkpoint
2. Add new files to folder
3. Enable "Incremental Mode"
4. Reuses existing tokenizer
5. Learns new + old data
```

## Configuration

### Model Parameters
- **d_model**: Embedding dimension (256/512/768/1024)
- **n_heads**: Attention heads (must divide d_model)
- **n_layers**: Transformer depth (4-16)
- **d_ff**: Feed-forward dimension (typically 4x d_model)

### Training Parameters
- **Learning Rate**: 0.0003 (standard), 0.0001 (stable), 0.00005 (fine-tune)
- **Batch Size**: Limited by VRAM (32 for 8GB, 128 for 24GB)
- **Context Length**: 512 (8GB GPU), 1024+ (16GB+)
- **Precision**: BF16 (RTX 30xx+), FP16 (older), FP32 (debugging)

## Project Structure
```
projects/
└── my_project/
    ├── settings.json          # UI state
    ├── checkpoints/
    │   ├── python_llm_best.pt # Best model
    │   ├── python_llm_last.pt # Latest
    │   └── tokenizer.json     # Vocabulary
    ├── encoded_data/
    │   ├── train_data.npy
    │   └── val_data.npy
    └── stats/
        └── training_stats.json
```

## Data Refinement Workflow

1. **Scan**: Analyze folder for issues
2. **Review**: Inspect quality scores, duplicates
3. **Move Garbage**: Selected files → `_trash` folder
4. **Train**: Use cleaned dataset

### Quality Metrics
- **Code**: AST validity, entropy, comment ratio, function count
- **Text**: Readability, vocabulary diversity, paragraph structure

## Advanced Features

### Context Extension
Increase model's context window:
```
1. Train base model (512 context)
2. Load checkpoint
3. Set new context (1024)
4. Train 1-2 epochs with longer sequences
```

### DPO Creator
Generate preference pairs for alignment:
```
1. Select document type (prose/code)
2. Extract samples
3. Generate chosen/rejected pairs
4. Export JSON for DPO training
Performance Tips

Mixed Precision: Use BF16 on Ampere+ GPUs
Gradient Accumulation: Simulate larger batches
Early Stopping: Enable with patience=3-5
Validation Split: 10% for most datasets, 3% for huge datasets
Checkpointing: Save every 100-500 steps for long runs

Troubleshooting
CUDA Out of Memory

Reduce batch size
Lower context length
Use gradient checkpointing
Enable FP16 precision

Slow Tokenization

Install tokenizers package
Reduce vocabulary size
Use incremental mode

Poor Generation Quality

Increase model size
More training epochs
Better data quality (use refinement)
Adjust temperature/top-k/top-p

License
MIT License - See LICENSE file
Credits
Author: Shuvro Basu
Year: 2026
Tools: PY_CLEAN_DOCUMENT v1.0
Contributing
Pull requests welcome. Focus areas:

Additional model architectures (encoder-decoder, MoE)
More data formats (JSON, CSV parsing)
Distributed training support
LoRA/QLoRA integration
