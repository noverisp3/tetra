# Tetra — Pure Ternary LLM

**Tetra** is a decoder-only transformer trained entirely with **ternary weights** ({-1, 0, +1}) and exported to a **3.7 MB C++ binary** that runs at **420+ tok/s** on CPU.

Three training modes:

- **STE** (Straight-Through Estimator) — FP32 latent shadow weights quantized on-the-fly via absmean, gradient flows through STE. (BitNet b1.58 approach)
- **Stochastic Bit-Flip** — no latent weights. Weights stored as packed 2-bit ternary. Gradient sign accumulated in FP32 accumulator; weight flips when |accumulator| > threshold.
- **Hybrid SSM-Attention** — 80% Ternary SSM (Mamba-style) + 20% Ternary Attention layers. SSM scan via vectorized parallel prefix (O(T), no Python loop).

## Architecture

Base BitNet b1.58-style transformer, optionally hybridized:

| Component | STE / Stochastic | Hybrid |
|-----------|-----------------|--------|
| **Weights** | {-1, 0, +1} via absmean (STE) or packed 2-bit (Stochastic) | Same per-layer |
| **Attention** | Causal multi-head, KV cache, ternary Q/K/V/O projections | 20% of layers |
| **SSM Block** | — | 80% of layers: RMSNorm → TernaryLinear(expand 2×) → depthwise Conv1d → SiLU → parallel-prefix SSM scan → gate → TernaryLinear(project back) |
| **FFN** | SwiGLU: fused gate+up into one ternary matmul (2× FFN dim) | Same |
| **Sparsification** | Optional `--topk RATIO`: keep top-k% activations after norm, zero rest (STE backward) | Same |
| **INT8 Forward** | Optional `--int8`: quantize activations → int8 before matmul (QAT effect) | Same |
| **Normalization** | Pre-norm RMSNorm (always FP32 internally) | Same |
| **Tokenizer** | Custom BPE (default, vocab=8192) or GPT-2 (`--tokenizer-dir gpt2`, vocab=50257) | Same |

Key design decisions:
- **Attention & FFN compute in FP32** for numerical stability (q/k/v cast to fp32 before `scaled_dot_product_attention`, SwiGLU hidden cast to fp32 before down-projection). Avoids float16 overflow on large hidden dims.
- **`activation_dtype`** used instead of `torch.amp.autocast` for explicit control over precision.
- **RMSNorm always runs in FP32** regardless of activation dtype.

## Presets

| Preset | Params | Ternary | Non-ternary | hidden_dim | layers | heads | ffn_dim | Head dim |
|--------|--------|---------|-------------|------------|--------|-------|---------|----------|
| **tiny** | 8.5M | 6.3M | 2.2M | 256 | 6 | 8 | 1024 | 32 |
| **medium** | 54.6M | 53.9M | 660K | 512 | 12 | 8 | 2048 | 64 |
| **large** | 91.3M | 90.6M | 720K | 768 | 12 | 12 | 2048 | 64 |
| **500m** | 516M | 494M | 22M | 2560 | 6 | 40 | 6826 | 64 |

## Quick Start

```bash
# Train tiny on TinyStories (auto-downloads if missing)
python train.py --preset tiny --steps 5000 --dtype float16 --graph

# Resume from latest checkpoint + plot full history
python train.py --preset tiny --steps 10000 --dtype float16 --graph --resume

# Export to C++ binary (3.7 MB) and run inference
python inference/export_model.py checkpoints/checkpoint_005000.pt inference/tetra_model.bin
cd inference && build.bat avx2 && cd ..
python inference/run_inference.py inference/tetra_model.bin "Once upon a time" --max-tokens 50

# Use GPT-2 tokenizer instead of custom BPE
python train.py --preset tiny --steps 5000 --dtype float16 --tokenizer-dir gpt2

# Multi-source data (1B tokens from FineWeb/Cosmopedia/Orca)
python scripts/prepare_data.py --data-cache data
python train.py --preset 500m --steps 15000 --dtype float16 --data-cache data --batch-size 4 --grad-accum 8
```

## Mixed Precision

Manual `activation_dtype` casting (no `autocast`):

| `--dtype` | CUDA | DirectML | CPU |
|-----------|------|----------|-----|
| `float16` | activation_dtype=fp16 + GradScaler | activation_dtype=fp16 | — |
| `bfloat16` | activation_dtype=bf16 (if supported) | falls back to fp32 | — |
| `float32` | full fp32 | full fp32 | full fp32 |

On CUDA, GradScaler is active for float16. Attention q/k/v and FFN SwiGLU hidden are cast to FP32 before compute to prevent overflow.

## Data

- **TinyStories** (default, auto-download): ~535M tokens, simple children stories. Ideal for small models.
- **Multi-source** (FineWeb 50% + Cosmopedia 30% + Orca 20%): ~1B tokens, GPT-2 tokenizer. For 500M+ models.
- **Tokenizer**: Custom BPE (vocab=8192, trained on TinyStories) by default. GPT-2 (vocab=50257) via `--tokenizer-dir gpt2`.

## C++ Inference Engine

Pure C++17 inference engine (`inference/tetra.h`, no dependencies):

| Feature | Detail |
|---------|--------|
| **File size** | 3.7 MB (ternary weights 2-bit packed, embeddings INT8 quantized) |
| **Speed** | 420+ tok/s (AVX2, CPU), 370+ tok/s (scalar) |
| **Prefill** | Batch prefill — all prompt tokens in one forward pass |
| **Sampling** | Top-k + top-p + temperature, matches PyTorch order |
| **Build** | `build.bat avx2` (auto-detects VS via vswhere) |

### Export & Run

```bash
python inference/export_model.py checkpoints/checkpoint_*.pt inference/tetra_model.bin
cd inference && build.bat avx2
# Direct: provide token IDs
tetra_avx2.exe tetra_model.bin "373,378,67,338" 100 0.8 50 0.9
# Or via Python with tokenizer
python run_inference.py tetra_model.bin "Once upon a time" --max-tokens 100
```

### Binary Format (v3)

| Section | Encoding |
|---------|----------|
| Header (64B) | magic `TETR`, version, dims, param counts |
| Ternary weights | name, shape, alpha, 2-bit packed (4 weights/byte) |
| FP32/INT8 weights | name, shape, dtype byte: `0`=FP32, `1`=INT8 + scale |
| Embeddings | INT8 (token 8K×256→2.0 MB, pos 512×256→0.13 MB) |
| Norms | FP32 (tiny, ~12 KB) |

## Examples

### Tiny 8.5M on TinyStories

Trained for 5,000 steps on Intel Iris Xe (DirectML) in ~3 hours:

| Metric | Value |
|--------|-------|
| **Total params** | 8,523,008 (6.3M ternary + 2.2M FP32) |
| **Exported binary** | **3.7 MB** (INT8 embedding + 2-bit ternary weights) |
| **C++ inference** | **420+ tok/s** (AVX2) / 370+ tok/s (scalar) |
| **Dataset** | TinyStoriesV2-GPT4, 535M tokens, 267K stories |
| **Tokenizer** | Custom BPE, vocab=8192 |
| **Mode** | STE (latent weights) |
| **Batch** | 16 × 4 grad_accum = 64 effective |
| **Speed** | 2.26s/step |
| **Final train loss** | 4.3092 |
| **Final val loss** | 4.3182 |
| **Loss trend** | 60.96 → 4.31 (converged smoothly) |

<p align="center">
  <img src="examples/tiny/loss_plot.png" alt="Training Loss Plot" width="85%">
  <br>
  <em><b>Figure 1:</b> Convergence curve of Tetra 8.5M (STE) on TinyStories (5,000 steps, Cosine LR Decay with Warmup). Plot includes raw + EMA-smoothed train loss and validation loss.</em>
</p>

Sample output (C++ inference, AVX2, 3.7 MB binary):
> "a small . He looked . They all . They all day of the bird . She was very happy was very happy and the dog of . They all day long and the big dog and gave . She put the big hole and laughed had not and Lily"

Limited but recognizable — expected for 8.5M ternary params on simple stories. Training to 20k–30k steps on TinyStoriesV2 significantly improves coherence.


## Project Structure

```
train.py                    # Main entry point

scripts/
  benchmark_speed.py        # Speed benchmark across presets
  prepare_data.py           # Stream data from HF → tokenized chunks
  train_tokenizer.py        # Train BPE tokenizer on TinyStories

ternary_llm/
  quantization.py           # STE + Stochastic Bit-Flip autograd functions, pack/unpack
  layers.py                 # TernaryLinear, StochasticTernaryLinear, RMSNorm, TopKActivation
  attention.py              # MultiHeadAttention with KV cache
  ffn.py                    # SwiGLU FFN: fused gate_up_proj (2×FFN dim)
  ssm.py                    # Ternary SSM block (parallel-prefix scan)
  hybrid.py                 # Hybrid SSM-Attention transformer model
  transformer.py            # Full model, generate with KV cache, sample
  data.py                   # ChunkedDataset, MultiSourceChunkedDataset
  trainer.py                # TernaryTrainer + DMLAdamW
  int8.py                   # INT8 fake-quantization
  csrc/
    ternary_ops_avx2.cpp    # C++ SIMD pack/unpack (AVX2)
    ternary_ops_avx512.cpp  # C++ SIMD pack/unpack (AVX-512)
    setup.py                # PyTorch extension build

inference/
  tetra.h                   # C++ inference engine (RMSNorm, SiLU, softmax, sampling, forward)
  tetra.cpp                 # CLI entry point, generation loop
  export_model.py           # Checkpoint → binary format (v3, INT8 embedding)
  run_inference.py          # Python wrapper around C++ inference
  benchmark_ppl.py          # Perplexity measurement
  build.bat                 # MSVC build script (auto-detects VS via vswhere)

tests/
  test_quantization.py
  test_layers.py
  test_transformer.py
  test_prototype.py
  test_convergence.py
```