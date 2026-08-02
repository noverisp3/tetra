<p align="center">
  <img src="banner/tetra_banner.jpg" alt="Tetra Model Banner" width="100%">
</p>

<h1 align="center">Tetra - Pure Ternary LLM</h1>

**Tetra** is a decoder-only transformer trained entirely with **ternary weights** ({-1, 0, +1}) and exported to a **3.6 MB C++ binary** that runs at **300+ tok/s** on CPU (AVX2).

Three training modes:

- **STE** (Straight-Through Estimator) — FP32 latent shadow weights quantized on-the-fly via absmean, gradient flows through STE. (BitNet b1.58 approach)
- **Stochastic Bit-Flip** — no latent weights. Weights stored as packed 2-bit ternary. Gradient sign accumulated in FP32 accumulator; weight flips when |accumulator| > threshold. Supports cosine threshold decay (`--threshold-decay-to`), per-channel scaling (`--per-channel`), and per-group block scaling (`--group-size N`).
- **Hybrid SSM-Attention** — 80% Ternary SSM (Mamba-style) + 20% Ternary Attention layers. SSM scan via vectorized parallel prefix (O(T), no Python loop).

Plus **Multi-head Latent Attention (MLA)** — DeepSeek-V2-style KV compression for attention. Compresses K,V into a small latent vector (`--kv-latent-dim`, default 64) before caching, reducing KV cache by 4×. Uses decoupled RoPE with separate per-head Q/K rope projections (`--rope-per-head`, default 8). Compatible with Stochastic Bit-Flip mode (`--mla` flag).

## Architecture

Base BitNet b1.58-style transformer, optionally hybridized:

| Component | STE / Stochastic | Hybrid | MLA |
|-----------|-----------------|--------|-----|
| **Weights** | {-1, 0, +1} via absmean (STE) or packed 2-bit (Stochastic). Optional per-channel or per-group scaling alpha. Per-group: `--group-size N` splits `in_features` into blocks of N, each with its own alpha. | Same per-layer | Same |
| **Attention** | Causal multi-head, KV cache, ternary Q/K/V/O projections | 20% of layers | MLA: K,V compressed to latent (default 64-dim), decoupled RoPE (default 8-dim/head). 7 ternary projections: q, kv_down, k_up, v_up, q_rope, k_rope, o. KV cache reduced 4×. |
| **SSM Block** | — | 80% of layers: RMSNorm → TernaryLinear(expand 2×) → depthwise Conv1d → SiLU → parallel-prefix SSM scan → gate → TernaryLinear(project back) | — |
| **FFN** | SwiGLU: fused gate+up into one ternary matmul (2× FFN dim) | Same | Same |
| **Sparsification** | Optional `--topk RATIO`: keep top-k% activations after norm, zero rest (STE backward) | Same | Same |
| **INT8 Forward** | Optional `--int8`: quantize activations → int8 before matmul (QAT effect) | Same | Same |
| **Normalization** | Pre-norm RMSNorm (always FP32 internally) | Same | Same |
| **Tokenizer** | Custom BPE (default, vocab=8192) or GPT-2 (`--tokenizer-dir gpt2`, vocab=50257) | Same | Same |

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
python train.py --preset tiny --steps 15000 --dtype float16 --graph

# Resume from latest checkpoint + plot full history
python train.py --preset tiny --steps 30000 --dtype float16 --graph --resume

# Train with Multi-head Latent Attention (MLA, DeepSeek-V2 style)
python train.py --preset tiny --steps 15000 --dtype float16 --graph --mla --kv-latent-dim 64 --rope-per-head 8

# Export to C++ binary and run inference
python inference/export_model.py checkpoints/checkpoint_015000.pt inference/tetra_model.bin
cd inference && build.bat avx2 && cd ..
python inference/run_inference.py inference/tetra_model.bin "Once upon a time" --max-tokens 100 --repeat-penalty 1.1

# Use GPT-2 tokenizer instead of custom BPE
python train.py --preset tiny --steps 15000 --dtype float16 --tokenizer-dir gpt2

# Multi-source data (1B tokens from FineWeb/Cosmopedia/Orca)
python scripts/prepare_data.py --target-tokens 1e9
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
| **File size** | 3.6 MB (ternary weights 2-bit packed, embeddings INT8 quantized) |
| **Speed** | 300+ tok/s (AVX2, CPU), 80+ tok/s (scalar) |
| **MLA** | Full MLA decode + prefill with KV latent compression, decoupled RoPE, and K/V reconstruction from cached latents |
| **Prefill** | Parallel batch prefill — all prompt tokens in one forward pass (OpenMP) |
| **Sampling** | Top-k + top-p + temperature + repetition penalty, matches PyTorch order |
| **Build (Windows)** | `build.bat avx2` (auto-detects VS via vswhere, enables `/openmp`) |
| **Build (Linux)** | `build.sh avx2` (requires g++ with OpenMP) |

### Export & Run

```bash
python inference/export_model.py checkpoints/checkpoint_*.pt inference/tetra_model.bin

# Windows
cd inference && build.bat avx2
tetra_avx2.exe tetra_model.bin "373,378,67,338" 100 0.8 50 0.9 1.0

# Linux
cd inference && bash build.sh avx2
./tetra_avx2 tetra_model.bin "373,378,67,338" 100 0.8 50 0.9 1.0

# Or via Python with tokenizer (cross-platform)
python run_inference.py tetra_model.bin "Once upon a time" --max-tokens 100
```

### Binary Format (v4)

| Section | Encoding |
|---------|----------|
| Header (64B) | magic `TETR`, version, dims, param counts |
| Ternary weights | name (suffix `.latent_weights`), shape, `group_size(2B)` + `num_alphas(2B)` + `alphas(N×4B)`, 2-bit packed (4 weights/byte). `group_size=0` → per-channel or scalar (v3 compat). MLA auto-detected from tensor names containing `kv_down_proj`. |
| FP32/INT8 weights | name, shape, dtype byte: `0`=FP32, `1`=INT8 + scale |
| Embeddings | INT8 (token 8K×256→2.0 MB, pos 512×256→0.13 MB) |
| Norms | FP32 (tiny, ~12 KB) |

Three alpha modes controlled by `group_size` and `num_alphas`:

| Mode | `group_size` | `num_alphas` | Storage |
|------|-------------|-------------|---------|
| Scalar | 0 | 0 | default `alpha=1.0`, no extra bytes |
| Per-channel | 0 | rows | `rows × 4B` |
| Per-group | >0 | rows × ceil(cols/group_size) | `rows × ceil(cols/group_size) × 4B` |

Per-group (`--group-size N`): each output row is split into blocks of N dims, each block having its own FP32 alpha. Overhead: with `group_size=64` (tiny 256-dim) adds ~770 KB.

## Examples

### Tiny 8.5M on TinyStories (STE)

Trained for 15,000 steps on Intel Iris Xe (DirectML) in ~9.5 hours:

| Metric | Value |
|--------|-------|
| **Total params** | 8,523,008 (6.3M ternary + 2.2M FP32) |
| **Exported binary** | **3.7 MB** (INT8 embedding + 2-bit ternary weights) |
| **C++ inference** | **300 tok/s** (AVX2) / 80 tok/s (scalar) |
| **Dataset** | TinyStoriesV2-GPT4, 535M tokens, 80K stories |
| **Tokenizer** | Custom BPE, vocab=8192 |
| **Mode** | STE (latent weights) |
| **Batch** | 16 × 4 grad_accum = 64 effective |
| **Speed** | 2.28s/step |
| **Final train loss** | 3.7507 |
| **Final val loss** | 3.7663 |
| **Loss trend** | 61.02 → 3.75 (converged smoothly) |

<p align="center">
  <img src="examples/tiny/loss_plot_STE.png" alt="Training Loss Plot" width="85%">
  <br>
  <em><b>Figure 1:</b> Convergence curve of Tetra 8.5M (STE) on TinyStories (15,000 steps, Cosine LR Decay with Warmup). Plot includes raw + EMA-smoothed train loss and validation loss.</em>
</p>

C++ inference (exported binary, AVX2, 3.7 MB, 300 tok/s):
> "Once upon a time to play a little girl named Lily. She was very much. Mia saw a big, she was so happy and a new little boy to make her toys, 'Don's time for you?' Lily said, 'I don'm sorry the toy car.' Mia did not listen the girl smiled and said, 'No, we should. I have fun worry. I want to be careful.' Lily said, you are going. 'Thank, you can help me and they are nice.'"

Limited but coherent — expected for 8.5M ternary params trained for 15k steps on TinyStories. Training to 30k–50k steps would further improve narrative quality.

> **Reproduce this run:**
> ```bash
> python train.py --preset tiny --steps 15000 --dtype float16 --graph --data-cache tinydata --hybrid --mode ste
> ```

### Tiny 5.3M on TinyStories (SBF)

Trained for 15,000 steps on Intel Iris Xe (DirectML) in ~9.4 hours:

| Metric | Value |
|--------|-------|
| **Total params** | 5,377,280 (3.14M ternary + 2.2M FP32) |
| **Exported binary** | **3.6 MB** (INT8 embedding + 2-bit ternary weights, 384 KB ternary) |
| **C++ inference** | **300 tok/s** (AVX2) / 80 tok/s (scalar) |
| **Dataset** | TinyStoriesV2-GPT4, 535M tokens, 80K stories |
| **Tokenizer** | Custom BPE, vocab=8192 |
| **Mode** | Stochastic Bit-Flip (no latent FP32 weights) |
| **Batch** | 16 × 4 grad_accum = 64 effective |
| **Speed** | 2.15s/step |
| **Final train loss** | 5.8681 |
| **Final val loss** | 5.8168 |
| **Loss trend** | 61.02 → 5.81 (hit 3.14M capacity ceiling) |

<p align="center">
  <img src="examples/tiny/loss_plot_SBF.png" alt="Training Loss Plot" width="85%">
  <br>
  <em><b>Figure 2:</b> Convergence curve of Tetra 5.3M (SBF) on TinyStories (15,000 steps, Cosine LR Decay with Warmup). Plot includes raw + EMA-smoothed train loss and validation loss.</em>
</p>

C++ inference (exported binary, AVX2, 3.6 MB, 300 tok/s):
> "Hello . to Tim the . One saw time . " , her and the had day . , day , . the Tim , saw the , it friends . the the day endoftext that ." Tim little . and the there . a I , . , she the but to One the time the " to Once a a to little The friends a very a endoftext that was He was . was a a . . . a it to to and He and . " was play . " it " not The . a , , happy and She . It on . , Tim . was to the a , They , had Lily to The the . . big the " |> with the was They to and a . the the the little One . the a day and , endoftext to She he to a It and day One to , . . . and a the there It on a Tim her the , time , , and ' to happy ' . , The , not , said . to to day the the , . They " friends"

Limited due to ultra-compact 3.14M ternary weight capacity (384 KB memory footprint) without latent shadow weights. Demonstrates memory-efficient training behavior under tight capacity constraints.

> **Reproduce this run:**
> ```bash
> python train.py --preset tiny --steps 15000 --dtype float16 --graph --data-cache tinydata --hybrid --mode stochastic
> ```


## Discrete Learning (Gradient-Free) Research

A research track investigating whether a transformer can learn **without backpropagation**, using only local update rules, and whether that learning can continue **on-device in the C++ runtime**.

### Local rules (Python, `train_discrete.py`)

`DiscreteTrainer` trains a ternary discrete model with packed 2-bit weights and FP32 accumulators. No autograd — gradients are replaced by local, biologically-inspired rules computed from captured activations:

| Rule | Update signal | Val CE | Train CE |
|------|--------------|--------|----------|
| **p** (target predictive coding) | `delta ∝ -sign(y_t - ŷ)·x` toward target ŷ | **7.05** | 7.15 |
| **c** (temporal predictive coding) | `delta ∝ -sign(y_t - y_{t-1})·x` (self-prediction) | **7.16** | 7.31 |
| **b** (forward-forward) | Hebbian-style on corrupted positives (corrupt 0.2) | **7.70** | 7.78 |
| Random baseline | — | 9.01 | — |

All three rules beat the random baseline (9.01 = ln(8192)) by a wide margin — the core claim (gradient-free learning works) is validated in PyTorch.

Architecture: 6 layers, hidden 256, 8 heads, FFN 1024, vocab 8192 → 6,291,456 ternary params (2-bit packed), tied embedding.

### Self-learning C++ runtime (`inference/selflearn.cpp`)

The exported model **continues learning in C++** on raw token streams, implementing rule `c`:

- Per-layer local delta `-sign((y_t − y_{t-1})^T x_{t-1})` fed into a **leaky accumulator** (`acc *= 0.99`)
- **Bit-flip kernel**: weight flips ±1 when `|acc| > threshold` (default 20), applied every 5 blocks
- Tied embedding updated with local SGD: per-row gradient clip (norm<1 → normalize) + decoupled weight decay (`lr=1e-4`, `wd=0.1`)
- **Atomic write-back**: `.tmp` file + `MoveFileEx` rename (crash-safe save)

### Binary format v6

v6 = v5 + one **FP32 accumulator** (`rows×cols`) appended to every ternary entry + a trailing `META` section containing a JSON blob with the self-learning config (`sl_rule`, `sl_threshold`, `sl_acc_decay`, `sl_flip_every_n`, `sl_logit_scale`, `sl_lr_embedding`, `sl_wd_embedding`, `sl_block_size`, `_export_version`). Export via `export_model.py ... --self-learning`.

### Validation results

| Check | Result |
|-------|--------|
| C++ inference CE vs PyTorch (held-out) | 7.137 vs 7.16 — matches, C++ numerics correct |
| 60 blocks C++ self-learning on held-out | **7.1373 → 7.1127** (generalizes, not overfitting) |
| Bit-flips applied | 94–100% no-ops; only ~130–1.5K real changes per flip step |
| Round-trip (save → reload → continue) | Works, CE keeps decreasing |
| `tetra.exe` generation on learned model | Works |
| Unit tests (`tests/test_discrete.py`) | 9/9 pass |

### Measured results (held-out slice, 20–40K positions)

| Model | Held-out CE | Notes |
|-------|-------------|-------|
| Random baseline | 9.0109 | ln(8192) |
| Discrete rule c, 250 steps (v6 export) | 7.137 | Python val 7.16 |
| + 60 blocks C++ self-learning | 7.1127 | |
| + 500 blocks C++ self-learning | **6.9964** | on-device, no backprop |
| Backprop SBF baseline, 300 steps | **6.9020** | best at step 200, same architecture |
| Backprop SBF, **ternary frozen** | **6.8891** | same run, `--no-flips` — only embedding/lm_head trained |
| Embedding-only ablation (60 blocks) | 7.1209 | ternary flips off |
| **Cut-the-tail**: learned ternary + frozen trained embedding | **7.157** | rule-c checkpoint 250 |
| **Cut-the-tail**: random ternary + same frozen embedding | **8.120** | avg of 3 seeds |

Findings:

1. **Continued on-device learning works and keeps improving**: held-out CE drops 7.137 → 6.997 across 500 blocks (~1.02M tokens) — no backprop, pure local rule `c` + embedding SGD.
2. **Backprop is better but the gap is small**: at 60 blocks the local rule is ~0.21 nats behind a real optimizer on the same ternary architecture; after 500 blocks the gap shrinks to **~0.09 nats** (6.996 vs 6.902) at ~3.7× the token budget.
3. **Ternary flips contribute, embedding dominates**: embedding-only gets 7.121 vs full 7.113 at 60 blocks — flips add ~1/3 of the total improvement.
4. **No bit churn — flips are mostly saturation no-ops** (diagnostic, 200 blocks): 94–100% of counted flips do not change the weight (already saturated at ±1); only ~0.18% of the 6.29M weights ever change value, and weights flipped ≥4 times are 0.00%. The huge per-step "flip counts" reported earlier were a measurement artifact of counting no-op saturation flips.
5. **Cut-the-tail ablation: the learned ternary weights carry real structure, worth ~1 nat.** With a *frozen trained embedding*, swapping the rule-c ternary (250 steps) for a random one (same init distribution) degrades held-out CE by **+0.96 nats** (7.157 vs 8.120, consistent over 3 seeds). The ternary core is *not* decoration.
6. **How to reconcile with the frozen-ternary backprop result**: backprop trains the embedding/lm_head *from scratch*, so it adapts the FP32 readout to whatever ternary projection it gets — it compensates for random ternary (6.889) and never needs the ternary to learn. But the co-adapted (embedding, ternary) pair from local rules is worth ~1 nat over (embedding, random ternary). So: the FP32 parts dominate the raw CE number, yet the ternary projection is a genuine learned substrate.

### Hyperparameter sweep (100 blocks, held-out CE @10K positions)

| Variant | Held-out CE | Real weight changes |
|---------|-------------|---------------------|
| base (thr=20, decay=0.99, every=5) | 7.1195 | ~6K |
| thr=40 | 7.1179 | fewer |
| thr=10 | **7.1041** | more |
| decay=0.95 | 7.1196 | 0 (never flips) |
| flip every 1 block | 7.1245 | ~similar |

Threshold/decay/flip frequency move held-out CE by <0.02 nats — within noise. The bottleneck is not flip frequency but weight saturation, so tuning these parameters cannot unlock more ternary learning.

**Known limitations / open questions:**
1. Gap to backprop is ~0.09 nats at 3.7× the token budget — a longer/tuned backprop run (or a proper AdamW schedule without early overfitting) may widen it.
2. Backprop baseline showed late-training instability (LR decayed too fast → final CE rose to 8.0); its reported best (step 200) may understate what a tuned run achieves.
3. **Ternary learning is starved by saturation**: 94–100% of flips are no-ops on ±1-saturated weights; only ~0.18% of weights change over 200 blocks. To make bit-flips drive real learning the mechanism must change (e.g., allow moving past saturation, toggle-based flips, or a less-saturated init) — hyperparameter tuning alone does not help.
4. **The ternary substrate learns, but its on-device headroom is small**: cut-the-tail shows the ternary learned during Python training is worth ~1 nat over random, but continued C++ flips add only ~0.008 nats (only 0.18% of weights move). The open question is whether a better on-device mechanism (less-saturated init, toggle flips, or a mechanism that moves the co-adapted embedding+ternary pair) can reproduce more of that 1-nat gain at runtime.
5. Still no standard-quality generation (CE ~7 → PPL ~1090) — gradient-free learning is proven effective but far from fluent text; scaling tokens/steps is untested.

## Project Structure

```
train.py                    # Main entry point
train_discrete.py           # Gradient-free training: local rules (p/c/b), DiscreteTrainer
train_baseline_backprop.py  # Backprop baseline (same architecture, AdamW, eval on same slice)
eval_ternary_ablation.py    # Cut-the-tail: learned vs random ternary with frozen embedding

scripts/
  benchmark_speed.py        # Speed benchmark across presets
  prepare_data.py           # Stream data from HF → tokenized chunks
  train_tokenizer.py        # Train BPE tokenizer on TinyStories

ternary_llm/
  quantization.py           # STE + Stochastic Bit-Flip autograd functions, pack/unpack
  layers.py                 # TernaryLinear, StochasticTernaryLinear, RMSNorm, TopKActivation
  attention.py              # MultiHeadAttention with KV cache
  mla.py                    # StochasticMLAAttention — Multi-head Latent Attention (DeepSeek-V2 style)
  ffn.py                    # SwiGLU FFN: fused gate_up_proj (2×FFN dim)
  ssm.py                    # Ternary SSM block (parallel-prefix scan)
  hybrid.py                 # Hybrid SSM-Attention transformer model
  transformer.py            # Full model, generate with KV cache, sample (includes StochasticMLABlock/StochasticMLAModel)
  discrete.py               # DiscreteConfig, DiscreteTrainer, 5 local rules, accumulator bit-flips
  data.py                   # ChunkedDataset, MultiSourceChunkedDataset
  trainer.py                # TernaryTrainer + DMLAdamW
  int8.py                   # INT8 fake-quantization
  csrc/
    ternary_ops_avx2.cpp    # C++ SIMD pack/unpack (AVX2)
    ternary_ops_avx512.cpp  # C++ SIMD pack/unpack (AVX-512)
    setup.py                # PyTorch extension build

inference/
  tetra.h                   # C++ inference engine (RMSNorm, SiLU, softmax, sampling, forward, v6 loader)
  tetra.cpp                 # CLI entry point, generation loop
  selflearn.cpp             # C++ self-learning runtime (rule c, accumulator bit-flips, --eval mode)
  export_model.py           # Checkpoint → binary format (v4/v6, INT8 embedding, --self-learning)
  run_inference.py          # Python wrapper around C++ inference
  benchmark_ppl.py          # Perplexity measurement
  build.bat                 # MSVC build script (auto-detects VS via vswhere)

tests/
  test_quantization.py
  test_layers.py
  test_transformer.py
  test_prototype.py
  test_convergence.py
  test_discrete.py          # 9 tests: rules, accumulators, bit-flips, checkpoint round-trip
```

## License

Apache License 2.0. See [LICENSE](LICENSE).