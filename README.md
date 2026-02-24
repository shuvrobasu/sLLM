# Python sLLM (small) Trainer v1.0 - Professional Edition

A production-grade GenAI desktop application for training custom transformer language models on code and text datasets with a modern themed GUI.

![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)
![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C)

## Features

### Core Capabilities

**GPT Model (Decoder-Only Transformer) Base Architecture:**

- Token + positional embeddings
- N transformer blocks with residual connections
- Pre-norm architecture (LayerNorm before attention/FFN)
- Weight tying (token embeddings = output projection)
- Causal self-attention masking
- **BOTH TRAINING AND INFERENCE RUN FROM SAME APP**
- **ABILITY TO TRAIN 1B+ MODELS** (subject to system restrictions)
- **TESTED TRAINING AND INFERENCE OF 450M MODEL**
- **CAN BE USED TO TRAIN ON ANY TEXT** (Python, C, Rust, eBooks, TXT)
- **SEE MY OTHER REPOS**: 
  - [Repo Extractor](https://github.com/shuvrobasu/repo_view_extract)
  - [eBook Extractor](https://github.com/shuvrobasu/ebook_convert_extract)

![Main Interface]("https://github.com/user-attachments/assets/4cc2b11c-26ee-48b0-b701-1afd128ee1eb")

### Components

**1. MultiHeadAttention**

- Fused QKV projection (3 × d_model)
- Flash attention compatible (F.scaled_dot_product_attention)
- Causal masking for autoregressive generation
- Dropout after attention

**2. TransformerBlock**

- Pre-LayerNorm architecture
- Multi-head self-attention with residual
- Feed-forward network (d_model → d_ff → d_model)
- GELU activation
- Optional gradient checkpointing (trade compute for memory)

**3. Generation Capabilities**

- Temperature-based sampling
- Top-K filtering
- Top-P (nucleus) sampling
- Repetition penalty (configurable: >1.0 = penalize, <1.0 = encourage)
- EOS token detection
- Max token limiting

### Architecture Configurations

![Architecture Configurations](https://github.com/user-attachments/assets/e6ca5dae-5eb5-4fdf-a8ce-dda3a8b8952a)

## Model Specifications

### Embedding

- **Vocabulary**: 16K - 50K tokens (BPE)
- **Dimension**: Must be divisible by number of heads
- **Position Encoding**: Learned embeddings up to max_seq_len

### Attention

- **Head Dimension**: d_model / n_heads (typically 64 or 128)
- **Mechanism**: Scaled dot-product attention
- **Masking**: Causal masking (lower triangular)
- **Dropout**: Applied post-attention

### Feed-Forward

- **Expansion Ratio**: 4x (d_ff = 4 × d_model)
- **Activation**: GELU (smooth, differentiable)
- **Dropout**: After both linear layers

### Normalization

- **Type**: LayerNorm before each sublayer (pre-norm)
- **Epsilon**: 1e-5
- **Final**: LayerNorm before output projection

### Initialization

- **Weights**: Normal(mean=0, std=0.02)
- **Biases**: Zeros
- **Scheme**: Following GPT-2 initialization

## Training Features

### Precision Modes

- **BF16 (bfloat16)**: RTX 30xx/40xx, no scaling needed
- **FP16 (float16)**: Older GPUs, requires loss scaling
- **FP32 (float32)**: Full precision fallback

### Optimizations

- **Gradient Checkpointing**: Recompute activations during backward pass
- **Gradient Accumulation**: Simulate larger batches
- **Mixed Precision Training**: Faster compute, lower memory
- **Non-blocking Tensor Transfers**: Overlap CPU/GPU operations

### Learning Rate Schedule

- **Linear Warmup**: First 10% of training
- **Cosine Decay**: Smooth reduction to 0
- **Formula**: `lr = base_lr × 0.5 × (1 + cos(π × progress))`

### Loss Function

- **Type**: Cross-entropy over vocabulary
- **Objective**: Next-token prediction
- **Averaging**: Over sequence length and batch

## DPO Training Support

**Direct Preference Optimization for alignment:**<br>
_Requires a model like microsoft/phi-2 to be downloaded and saved in a folder and configured in App_

### DPO Loss
```
L = -log(σ(β × (log π(y_w|x) - log π(y_l|x))))
```

**Where:**

- **β**: KL penalty coefficient (default: 0.1)
- **y_w**: Chosen/preferred response
- **y_l**: Rejected/dispreferred response
- **σ**: Sigmoid function

### Features

- Preference pair generation from documents
- Reference model policy (frozen)
- Bradley-Terry preference model
- Reward margin tracking

![DPO Training 1](https://github.com/user-attachments/assets/50b1f466-f64d-444b-8483-e6c8b442d25d)

![DPO Training 2](https://github.com/user-attachments/assets/42af7f18-f75e-4955-997d-bfcc17a93125)

## GUI Features

- **Professional Theme**: Navy blue design with icons, tooltips, and status indicators
- **Non-Blocking Operations**: All heavy tasks run in background threads
- **Toast Notifications**: Non-intrusive feedback system
- **Hardware Monitor**: Real-time CPU/RAM/VRAM graphs in sidebar
- **Model Tester**: Interactive text generation with temperature/top-k/top-p controls

![GUI Feature 1](https://github.com/user-attachments/assets/f5363822-c914-4e62-aa24-e6326ebaf84b)

![GUI Feature 2](https://github.com/user-attachments/assets/efc7913b-a0d8-46e9-8a9a-b6d0e36e7bbc)

![GUI Feature 3](https://github.com/user-attachments/assets/1aa7b93c-cdcc-4db7-a53e-9c6aa1b575bb)

### Inbuilt Comprehensive Help

![Help System](https://github.com/user-attachments/assets/2b2f83fa-0559-4ad2-916a-43dd0499ac5e)

## Model File Format

### Checkpoint Structure (.pt)
```python
{
    'model_state_dict': OrderedDict,  # Model weights
    'optimizer_state_dict': dict,      # Optimizer state
    'epoch': int,                      # Current epoch
    'loss': float,                     # Training loss
    'config': dict,                    # Model configuration
    'tokenizer_config': dict           # Tokenizer settings
}
```

### Parameter Count Estimation

![Parameter Count](https://github.com/user-attachments/assets/59f134cb-339a-49e8-bd36-f35952de4c4d)


**Formula:**
```
Total Params ≈ (vocab_size × d_model) + (max_seq_len × d_model) 
               + n_layers × (4 × d_model² + 8 × d_model × d_ff)
```

## VRAM Requirements

### Breakdown

**Model Parameters:**
- ~4 bytes per param (FP32) or ~2 bytes (FP16/BF16)

**Activations:**
- batch_size × seq_len × d_model × n_layers × factor

**Optimizer State:**
- 2x model params (Adam: momentum + variance)

**Gradients:**
- 1x model params

### Example (110M params, BF16)

- **Model**: 110M × 2 = 220MB
- **Optimizer**: 110M × 2 × 4 = 880MB (FP32 state)
- **Activations** (batch=64, seq=512): ~6GB
- **Gradients**: 220MB
- **Total**: ~8-9GB VRAM

## Context Extension

Increase model context length post-training:

### Method

1. Train base model (e.g., 512 tokens)
2. Load checkpoint
3. Extend position embeddings (interpolate or random init)
4. Fine-tune 1-2 epochs with longer sequences
5. Gradually increase to target length

### Position Interpolation
```python
# Linearly interpolate position embeddings
old_positions = model.pos_emb.weight.data
new_positions = F.interpolate(
    old_positions.T.unsqueeze(0), 
    size=new_max_len, 
    mode='linear'
).squeeze(0).T
```

## Generation Parameters

### Temperature (0.1 - 2.0)

- **Low (0.1-0.5)**: Deterministic, focused
- **Medium (0.6-0.9)**: Balanced creativity
- **High (1.0-2.0)**: Creative, diverse

### Top-K (1 - 100)

- Limits sampling to K most probable tokens
- Lower = more focused
- **Default**: 40

### Top-P (0.0 - 1.0)

- Nucleus sampling
- Samples from smallest set with cumulative probability ≥ p
- **Default**: 0.9

### Repetition Penalty (0.5 - 2.0)

- **< 1.0**: Encourages repetition
- **= 1.0**: No effect
- **> 1.0**: Penalizes repetition
- **Default**: 1.2

## Technical Details

### Dataset Processing

**CodeDataset:**

- Sliding window with configurable stride
- Efficient numpy memory mapping
- Batched loading with PyTorch DataLoader
- Pin memory for faster GPU transfer

### Tokenization

- **BPE (Byte Pair Encoding)**: Via HuggingFace tokenizers
- **Fallback**: Simple word-level tokenizer
- **Special Tokens**: 
  - [PAD] = 0
  - [UNK] = 1
  - [BOS] = 2
  - [EOS] = 3
- **Vocabulary Pruning**: By frequency

### Validation Strategy

**Metrics:**

- **Perplexity**: exp(val_loss)
- **Loss**: Raw cross-entropy loss
- **Gradient Norms**: For monitoring stability

**Early Stopping:**

- **Patience**: N epochs without improvement
- **Metric**: Validation loss
- **Restoration**: Load best checkpoint on stop

## System Requirements

### Minimum

- Python 3.8+
- 8GB System RAM
- GPU: 6GB VRAM (NVIDIA RTX 2060 or equivalent)
- Storage: 10GB+ for checkpoints

### Recommended

- Python 3.10+
- 16GB System RAM
- GPU: 12GB+ VRAM (RTX 3060/4060 Ti/50xx or better)
- Storage: 50GB+ SSD for datasets

## Dependencies

### Required
```bash
torch>=2.0.0
psutil
```

### Optional (for fast tokenization)
```bash
tokenizers
```

### CUDA

- CUDA 13.0 or compatible version

## Installation
```bash
# Clone repository
git clone https://github.com/shuvrobasu/sLLM_trainer.git
cd sLLM_trainer

# Install dependencies
pip install torch psutil tokenizers

# Run application
python sllm.py
```

## Quick Start

1. **Create Project**: Enter project name in sidebar, click Save Project
2. **Select Data**: Browse to folder with .py/.txt files
3. **Choose Preset**: Click Auto-Select or pick GPU/dataset size preset
4. **Train**: Click Start Training

<img width="1920" height="1040" alt="image" src="https://github.com/user-attachments/assets/fc049288-589e-488b-ae4e-9450c65f708c" />

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

1. **Scan folder** → processes all files
2. **Train tokenizer** → vocabulary generation
3. **Encode data** → converts to token IDs
4. **Train model** → transformer training

### Incremental Training

1. Load checkpoint
2. Add new files to folder
3. Enable "Incremental Mode"
4. Reuses existing tokenizer
5. Learns new + old data

## Configuration

### Model Parameters

- **d_model**: Embedding dimension (256/512/768/1024)
- **n_heads**: Attention heads (must divide d_model)
- **n_layers**: Transformer depth (4-16)
- **d_ff**: Feed-forward dimension (typically 4x d_model)

### Training Parameters

- **Learning Rate**: 
  - 0.0003 (standard)
  - 0.0001 (stable)
  - 0.00005 (fine-tune)
- **Batch Size**: Limited by VRAM
  - 32 for 8GB
  - 128 for 24GB
- **Context Length**: 
  - 512 (8GB GPU)
  - 1024+ (16GB+)
- **Precision**: 
  - BF16 (RTX 30xx+)
  - FP16 (older GPUs)
  - FP32 (debugging)

### Project Structure

![Project Structure](https://github.com/user-attachments/assets/ac3d918e-04c4-48dc-88b7-bc1f730cd859)
```
project_name/
|
+-- checkpoints/
|   +-- model_epoch_1.pt
|   +-- model_epoch_2.pt
|   +-- best_model.pt
|
+-- tokenizer/
|   +-- tokenizer.json
|   +-- vocab.txt
|
+-- encoded_data/
|   +-- train.npy
|   +-- val.npy
|
+-- logs/
|   +-- training.log
|   +-- metrics.csv
|
+-- config.json
```

## Data Refinement Workflow

1. **Scan**: Analyze folder for issues
2. **Review**: Inspect quality scores, duplicates
3. **Move Garbage**: Selected files → `_trash` folder
4. **Train**: Use cleaned dataset

### Quality Metrics

**Code:**
- AST validity
- Entropy
- Comment ratio
- Function count

**Text:**
- Readability scores
- Vocabulary diversity
- Paragraph structure

## Advanced Features

### Context Extension

Increase model's context window:

1. Train base model (512 context)
2. Load checkpoint
3. Set new context (1024)
4. Train 1-2 epochs with longer sequences

### DPO Creator

Generate preference pairs for alignment:

1. Select document type (prose/code)
2. Extract samples
3. Generate chosen/rejected pairs
4. Export JSON for DPO training

## Performance Tips

- **Mixed Precision**: Use BF16 on Ampere+ GPUs
- **Gradient Accumulation**: Simulate larger batches
- **Early Stopping**: Enable with patience=3-5
- **Validation Split**: 10% for most datasets, 3% for huge datasets
- **Checkpointing**: Save every 100-500 steps for long runs

## Troubleshooting

### CUDA Out of Memory

**Solutions:**
- Reduce batch size
- Lower context length
- Use gradient checkpointing
- Enable FP16/BF16 precision

### Slow Tokenization

**Solutions:**
- Install `tokenizers` package
- Reduce vocabulary size
- Use incremental mode

### Poor Generation Quality

**Solutions:**
- Increase model size
- More training epochs
- Better data quality (use refinement)
- Adjust temperature/top-k/top-p

## Test System
- CPU Core 7 Ultra 265K
- RAM DDR5 @5600 MHz 64 GB
- RTX 5080 16GB with Cuda 13
- SSD (Samsung)
- Lots of patience :-)

**Recommend H/W and Notes**
- At least 14+ Core CPU
- Min 48GB DDR5 RAM
- RTX 4090 or later (more VRAM the faster it is). Ensure you have the right CUDA build for PyTorch, Transformers etc. 
- SDD (Samsung, Hynix, Crucial, WD)
- At the time of this release many ML modules are still not CUDA 13 compatible natively like ONXX-GPU, xFormers, Transformers (some modules have isses), PyTorch Compile. So you may need to tweak the code to make it work.
- The code is around 10k lines, and modular. Read the help section in the tool to understand things better.
- See the code for hardcoded paths. Change them to suit your needs.
- Drop the .ico and .llm file in the same path as sLLM.py
  
## Future Roadmap

- Additional model architectures (encoder-decoder, MoE)
- More data formats (JSON, CSV parsing)
- Distributed training support
- LoRA/QLoRA integration
- Model quantization (INT8, INT4)
- ONNX export
- Multi-GPU training

## License

MIT License - See LICENSE file

## Credits

**Author**: Shuvro Basu  
**Year**: 2026  
**Contact**: [GitHub](https://github.com/shuvrobasu)

## Contributing

Pull requests welcome! Focus areas:

- Performance optimizations
- Additional model architectures
- Better data preprocessing
- UI/UX improvements
- Documentation
- Bug fixes

---

**Made with ❤️ for ML researchers and enthusiasts**
