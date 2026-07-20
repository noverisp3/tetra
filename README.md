# Tetra — Pure Ternary LLM

**Tetra** is a decoder-only transformer trained entirely with **ternary weights** ({-1, 0, +1}). Two training modes:

- **STE** (Straight-Through Estimator) — FP32 latent shadow weights quantized on-the-fly via absmean, gradient flows through STE. (BitNet b1.58 approach)
- **Stochastic Bit-Flip** — no latent weights. Weights stored as packed 2-bit ternary. Gradient sign accumulated in FP32 accumulator; weight flips when |accumulator| > threshold.

## Architecture

BitNet b1.58-style transformer decoder:

| Component | Detail |
|-----------|--------|
| **Weights** | {-1, 0, +1} via absmean quantization; no FP32 latent in stochastic mode |
| **Embedding** | Token embedding **tied** with LM head; RoPE applied to queries/keys |
| **Attention** | Causal multi-head with KV cache; ternary Q/K/V/O projections |
| **FFN** | SwiGLU variant: fused gate+up projection into one ternary matmul (2× FFN dim) |
| **Normalization** | Pre-norm RMSNorm (FP32, not quantized) |
| **Tokenizer** | Custom BPE, vocab=8192, PAD=0 / BOS=1 / EOS=2 / UNK=3 |

### Mixed Precision

Manual activation casting (no `autocast`, which crashes on DirectML):

| Device | `--dtype` | Activation dtype |
|--------|-----------|-----------------|
| CUDA (bf16-capable) | `bfloat16` | `torch.bfloat16` |
| CUDA (no bf16) | `float16` | `torch.float16` |
| DirectML | `float16` | `torch.float16` |
| CPU | `float32` | None |

On CUDA with bf16, the wider exponent range avoids overflow compared to float16.

## Presets

| Preset | Params | Ternary | Non-ternary | hidden_dim | layers | heads | ffn_dim | Head dim |
|--------|--------|---------|-------------|------------|--------|-------|---------|----------|
| **tiny** | 2.16M | 2.09M | 66K | 128 | 4 | 4 | 512 | 32 |
| **medium** | 54.6M | 53.9M | 660K | 512 | 12 | 8 | 2048 | 64 |
| **large** | 91.3M | 90.6M | 720K | 768 | 12 | 12 | 2048 | 64 |
| **500m** | 516M | 494M | 22M | 2560 | 6 | 40 | 6826 | 64 |

Param count breakdown for **500m**: each layer has Q/K/V/O (4× 2560×2560), gate_up (6826×2560), down (2560×6826) — all ternary. Non-ternary includes embeddings, norms, and tied lm_head.

## Benchmark (vocab=1024, hidden=128, layers=2, 200 steps, Zipf-1024)

| Config | Train Loss | Val Loss | Ternary Mem | FP32 Mem | Optimizer Extra |
|--------|-----------|---------|-------------|----------|-----------------|
| **STE** (scale=0.7) | 4.61 | 5.10 | 2048KB FP32 + 128KB packed | 2594KB | 4096KB (AdamW) |
| **Stochastic** (th=20, auto) | **4.49** | **4.93** | 32KB packed only | 546KB | 0KB |
| Stochastic (th=10) | 4.75 | 5.20 | 32KB packed only | 546KB | 0KB |
| Stochastic (sc=0.7, auto) | 4.73 | 5.11 | 32KB packed only | 546KB | 0KB |

### Key Results

- **Stochastic (th=20) beats STE on val loss** (4.93 vs 5.07)
- **14× less ternary memory** (32KB vs 2048KB FP32 + 4096KB AdamW)
- Auto-threshold `threshold = 20 / scale` hits the sweet spot
- STE ~50% faster on small models (matmul not dominant); gap shrinks at scale

### Threshold Tuning (Stochastic)

The invariant: `threshold × scale = 20`

```
threshold = 20.0 / scale   # default auto-compute
```

Lower threshold → too much flipping (diverges). Higher → not enough learning (underfits).

## Training Speed

Batch=4, Seq=1024 (except 500m CPU: batch=2, seq=512). DML = Intel Iris Xe via DirectML. CPU = same hardware, MKL.

| Preset | Device | Params | Fwd | Bwd | Total/step | RAM | Est. 10K steps |
|--------|--------|--------|-----|-----|-----------|-----|-----------------|
| **tiny** | DML | 2.2M | 0.13s | 0.53s | 0.66s | — | 1.8h |
| **medium** | DML | 55M | 1.64s | 3.24s | 4.88s | — | 13.6h |
| **medium** | CPU | 55M | 0.47s | 2.36s | 2.83s | ~1GB | 7.9h |
| **large** | DML | 92M | 2.40s | 4.59s | 6.99s | — | 19.4h |
| **500m** | DML | 516M | ~10s | ~25s | ~35s | OOM step 2 | — |
| **500m** | CPU | 516M | 7.5s | 17.1s | 24.5s | 2.8GB | 68h |

CPU avoids DML OOM on 16GB shared VRAM. C++ SIMD pack/unpack (~5s for 500M weights) accelerates unpack on CPU. Bottleneck is matmul (77% of time) on Iris Xe 4-core CPU with MKL. `torch.compile` and `torch.amp.autocast` are not supported on DirectML.

## C++ Extension

SIMD-accelerated pack/unpack and ternary matmul (auto-loaded, Python fallback):

```bash
python build_cpp.py
```

| Function | Python (DML, 500M) | C++ SIMD (CPU, 500M) |
|----------|-------------------|---------------------|
| pack_ternary | ~6s | 3.7s |
| unpack_ternary | ~6s | 1.2s |
| ternary_matmul (CPU) | — | 7s (512×2048, AVX2) |

Includes fused stochastic backward (grad_x + accumulator update in one pass). See `ternary_llm/csrc/ternary_ops.cpp`.

## Usage

```bash
# Train STE mode (default)
python train.py --preset tiny --steps 1000

# Stochastic Bit-Flip
python train.py --mode stochastic --preset medium --steps 5000

# 500M model on CPU (stable, no OOM)
python train.py --mode stochastic --preset 500m --steps 10000 --batch-size 2 --device cpu --dtype float32

# 500M model on DirectML (faster, may OOM on 16GB)
python train.py --mode stochastic --preset 500m --steps 10000 --batch-size 4 --device directml --hybrid --dtype float16

# Build C++ SIMD extension for faster pack/unpack
python build_cpp.py

# Resume from latest checkpoint
python train.py --resume

# Benchmark speed across presets
python scripts/benchmark_speed.py --steps 10
```

### All Flags

| Flag | Description |
|------|-------------|
| `--mode ste\|stochastic` | Training mode (default: ste) |
| `--ternary-scale` | [STE] Scale factor or [Stochastic] weight magnitude (default: 0.7) |
| `--threshold` | [Stochastic] Flip threshold (default: 20/scale) |
| `--flip-every-n-steps` | [Stochastic] Check & flip bits every N steps (default: 5, cuts ~80% bit overhead) |
| `--per-channel` | [STE] Per-channel quantization threshold |
| `--preset tiny\|medium\|large\|500m` | Model size |
| `--data-cache` | Multi-source data directory (FineWeb+Cosmopedia+Orca) |
| `--batch-size` | Batch size per step (default: 16) |
| `--grad-accum` | Gradient accumulation steps (default: 4) |
| `--steps` | Total optimizer steps (default: 10000) |
| `--lr` | Peak learning rate (default: 1e-3) |
| `--device` | `cpu`, `cuda`, or `directml` |
| `--dtype` | `float32`, `float16`, or `bfloat16` |
| `--hybrid` | Model on GPU, optimizer on CPU (avoids DML fallback ops) |
| `--resume` | Auto-resume from latest checkpoint |

### C++ Extension Build

```bash
python build_cpp.py          # AVX2 auto-detected
```

Requires MSVC (Visual Studio 2022). The extension auto-loads at runtime; falls back to Python if not built.

## C++ Inference

SIMD-accelerated engine with precomputed dequantized weights:

| Approach | Speed | Quality |
|----------|-------|---------|
| Precomputed floats + AVX-512 | ~890 tok/s | Exact (matches PyTorch) |
| XNOR + popcount | faster | Approximate (compounding errors) |

### Build

Requires MSVC. Multiple SIMD targets:

```bash
cd inference
build.bat              # scalar fallback
build.bat avx2         # AVX2+FMA (widest compatibility)
build.bat avx512       # AVX-512 (legacy Intel)
build.bat avx10        # AVX10 (future Intel)
```

### Export & Run

```bash
python inference/export_model.py checkpoints/checkpoint_010000.pt inference/tetra_model.bin
python inference/run_inference.py inference/tetra_model.bin "Once upon a time"
```

Or directly:
```bash
inference\tetra.exe inference\tetra_model.bin "373,378,67,338" 100 0.8 50 0.9
```

Arguments: `model.bin token_ids max_tokens temperature top_k top_p`

## Perplexity

```bash
python inference/benchmark_ppl.py --checkpoint checkpoints/checkpoint_010000.pt
```

## Data

- **Sources**: FineWeb (50%), Cosmopedia (30%), Orca (20%)
- **Total**: ~1B tokens
- **Tokenizer**: Custom BPE, trained on TinyStories, vocab=8192
- **Preprocessing**: `python scripts/prepare_data.py`

## Project Structure

```
train.py                  # Main entry point

scripts/
  benchmark_speed.py      # Speed benchmark across presets
  prepare_data.py         # Stream data from HF → tokenized chunks
  train_tokenizer.py      # Train BPE tokenizer on TinyStories

ternary_llm/
  quantization.py         # STE + Stochastic Bit-Flip autograd functions, pack/unpack
  layers.py               # TernaryLinear, StochasticTernaryLinear, RMSNorm
  attention.py            # MultiHeadAttention with KV cache
  ffn.py                  # SwiGLU FFN: fused gate_up_proj (2×FFN dim)
  transformer.py          # Full model, generate with KV cache, sample
  data.py                 # ChunkedDataset, MultiSourceChunkedDataset
  trainer.py              # TernaryTrainer (handles both modes, hybrid opt, checkpoint)
  int8.py                 # INT8 fake-quantization (reference, not used in model)
  __init__.py
  csrc/
    ternary_ops.cpp       # C++ SIMD pack/unpack/matmul (AVX2)
    setup.py              # PyTorch extension build
    __init__.py

inference/
  tetra.h / tetra.cpp     # C++ inference engine (SIMD matmul, KV cache, sampling)
  export_model.py         # Checkpoint → binary format for C++
  run_inference.py        # Python wrapper around C++ inference
  benchmark_ppl.py        # Perplexity measurement
  verify_export.py        # Weight comparison: PyTorch vs binary
  build.bat               # MSVC build script

tests/
  test_quantization.py    # TernaryQuantizer, Int8Quantizer, pack/unpack
  test_layers.py          # TernaryLinear, RMSNorm, StochasticTernaryLinear
  test_transformer.py     # Full model forward, generate, KV cache
  test_prototype.py       # Original research prototype tests
  test_convergence.py     # Stochastic vs STE convergence comparison
```
