# Tetra

Pure ternary LLM with {-1, 0, +1} weights. Trained from scratch on TinyStories.

## Architecture

BitNet b1.58-style transformer decoder:
- **Ternary weights**: {-1, 0, +1} via absmean quantization (STE for gradient flow)
- **Shadow weights**: FP32 latent weights during training, quantized on-the-fly
- **128 hidden dim, 4 layers, 4 heads, 512 FFN dim** — 2.16M params total
  - Ternary: 1,048,576 (256 KB packed at 2 bits/weight)
  - FP32: 1,115,264 (embeddings + norms, 4.3 MB)

## Training

- **Dataset**: TinyStories V2 GPT-4 (267K stories, 534M tokens)
- **Tokenizer**: Custom BPE (8192 vocab, HuggingFace tokenizers)
- **Config**: batch=16, LR=1e-3, block_size=128, 10K steps
- **Device**: Intel Iris Xe (DirectML)
- **Duration**: 57 min
- **Loss**: train 3.56, val 3.61

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
  quantization.py   # TernaryQuantizer (absmean + STE)
  layers.py         # TernaryLinear, RMSNorm
  attention.py      # TernaryMultiHeadAttention
  ffn.py            # TernaryFFN (SwiGLU)
  transformer.py    # Full model, generate, sample
  data.py           # ChunkedDataset, tokenizer
  trainer.py        # TernaryTrainer

inference/
  tetra.h           # C++ inference engine (SIMD matmul, KV cache, sampling)
  tetra.cpp         # CLI runner with streaming output
  export_model.py   # Checkpoint -> binary format
  benchmark_ppl.py  # Perplexity measurement
  build.bat         # MSVC build script
```
