# Tetra

Pure ternary LLM with {-1, 0, +1} weights. Two training modes available.

## Training Modes

### 1. STE (default) — Straight-Through Estimator

FP32 latent (shadow) weights, quantized on-the-fly via absmean. Gradient flows through STE. Standard approach used by BitNet b1.58.

```
python train.py --mode ste                    # default
python train.py --mode ste --ternary-scale 0.7
```

### 2. Stochastic Bit-Flip

**No latent weights.** Weights stored as packed 2-bit ternary. Gradient sign accumulated in FP32 accumulator; weight flips when |accumulator| > threshold. Threshold auto-computed as `20.0 / scale`.

```
python train.py --mode stochastic
python train.py --mode stochastic --ternary-scale 0.7
python train.py --mode stochastic --threshold 15.0  # manual override
```

## Benchmark (vocab=1024, hidden=128, layers=2, 200 steps)

| Config | Train | Val | Ternary Mem | FP32 Mem | Optimizer Extra |
|--------|-------|-----|-------------|----------|-----------------|
| **STE** (scale=0.7) | 4.61 | 5.10 | 2048KB FP32 + 128KB packed | 2594KB | 4096KB (AdamW) |
| **Stoch** (th=20, auto) | **4.49** | **4.93** | 32KB packed only | 546KB | 0KB |
| Stoch (th=10) | 4.75 | 5.20 | 32KB packed only | 546KB | 0KB |
| Stoch (sc=0.7, auto) | 4.73 | 5.11 | 32KB packed only | 546KB | 0KB |

**Key results:**
- Stochastic (th=20) **beats STE on val loss** (4.93 vs 5.10-5.07)
- **14× less memory** for ternary weights (32KB vs 2048KB FP32 + 4096KB AdamW)
- Auto-threshold formula `threshold = 20 / scale` matches optimal tuning
- STE is ~50% faster on small models (matmul not dominant); gap shrinks at scale

### Threshold Tuning

The invariant: `threshold × scale = 20` is the sweet spot. Lower → too much flipping (diverges). Higher → not enough learning (underfits).

```
threshold = 20.0 / scale   # default auto-compute
```

## Architecture

BitNet b1.58-style transformer decoder:
- **Ternary weights**: {-1, 0, +1} — no FP32 latent in stochastic mode
- **Token embedding tied** with LM head
- **SwiGLU FFN** with fused gate+up projection
- **RMSNorm** (pre-norm)
- **Causal attention** with RoPE

## Training

```
python train.py                              # STE mode (default)
python train.py --mode stochastic            # Stochastic Bit-Flip
python train.py --preset 500m --steps 10000  # 500M param model
python train.py --resume                     # Auto-resume from latest checkpoint
```

### Flags

| Flag | Description |
|------|-------------|
| `--mode ste\|stochastic` | Training mode (default: ste) |
| `--ternary-scale` | [STE] Scale factor (default: 0.7) |
| | [Stochastic] Weight magnitude |
| `--threshold` | [Stochastic] Flip threshold (default: 20/scale) |
| `--per-channel` | [STE] Per-channel quantization |
| `--data-cache` | Multi-source data directory |
| `--hybrid` | Model on GPU, optimizer on CPU |
| `--preset tiny\|medium\|large\|500m` | Model size |

## C++ Inference

SIMD-accelerated engine using precomputed dequantized weights:

| Approach | Speed | Quality |
|----------|-------|---------|
| Precomputed floats + AVX-512 | 890 tok/s | Exact (matches PyTorch) |
| XNOR + popcount | faster | Approximate (compounding errors) |

### Build

Requires MSVC with AVX-512 support:

```bash
cd inference
build.bat          # outputs tetra.exe
```

### Export model

```bash
python inference/export_model.py checkpoints/checkpoint_010000.pt inference/tetra_model.bin
```

### Run

```bash
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

## Project Structure

```
ternary_llm/
  quantization.py   # STE and Stochastic Bit-Flip autograd functions
  layers.py         # TernaryLinear, StochasticTernaryLinear, RMSNorm
  attention.py      # MultiHeadAttention (STE + Stochastic variants)
  ffn.py            # SwiGLU FFN (STE + Stochastic variants)
  transformer.py    # Full model, generate, sample
  data.py           # ChunkedDataset, tokenizer
  trainer.py        # TernaryTrainer (handles both modes)

inference/
  tetra.h           # C++ inference engine (SIMD matmul, KV cache, sampling)
  tetra.cpp         # CLI runner with streaming output
  export_model.py   # Checkpoint -> binary format
  benchmark_ppl.py  # Perplexity measurement
  build.bat         # MSVC build script
```
