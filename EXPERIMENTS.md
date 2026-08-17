# Tetra — Experimental Log (Exp 1–19)

Chronological record of the experiment series, one section per experiment
(hypothesis → setup → results → verdict → reproduce). Sections are ordered
Exp 1 → Exp 19 (Exp 15 is a paused-status note); the README "Experiments"
section links back here.

1. **Exp 1** — Surprise-Gated Bit Flips
2. **Exp 2** — Continual Learning & Catastrophic Forgetting
3. **Exp 3** — Energy Accumulator + Adaptive Threshold
4. **Exp 4** — Applying the Mechanism to the Discrete Trainer
5. **Exp 5** — Flip Mechanism at Scale
6. **Exp 6** — Real New-Domain Continual Learning
7. **Exp 7** — Gradient-Free Continual Learning in C++
8. **Exp 8** — Soft-to-Hard Quantization Warmup
9. **Exp 9** — Ternary-Scale Tuning
10. **Exp 10** — True-Value Outlier Training (v8-forward)
11. **Exp 11** — Outlier-Band Regularization (v8-reg, rejected)
12. **Exp 12** — Learned 5-Level Codebook (LQ, null result)
13. **Exp 13** — Unbiased Stochastic Rounding (SR, rejected)
14. **Exp 14** — Annealed Soft Flips (Gumbel-flip, rejected)
15. **Exp 15** — MLA / Hybrid Attn (status note)
16. **Exp 16** — ARS-Gated Embedding Updates
17. **Exp 17** — KV-Cache Quantization (int8/int16 SIMD)
18. **Exp 18** — UTF Churn-Ramp Pipeline (closed, negative result)
19. **Exp 19** — MatMul-Free Learning Rule (`--no-mul`, Variant 2 beats baseline)

---

## Experiment: Surprise-Gated Bit Flips (Exp 1)

Question: does the threshold "surprise gate" in `apply_bit_flips` save compute
without hurting accuracy, compared to an un-gated flip on every accumulator
sign?

Setup: `train.py --mode stochastic --preset tiny`, 200 steps on `tinydata`
(534M tokens). Slice CE: `tests/eval_ternary_ablation.py` on
`examples/discrete/sliceEval100k.bin` (20K positions, `--scale 1.0`). Same
config for all four runs; only the gate/threshold differs. Flip counts
instrumented in `quantization.apply_bit_flips` (positions where
`w_new != w_raw`).

| Gate | Threshold | % weights flipped / pass | Slice CE |
|---|---|---|---|
| Full (ungated) | 0 | 62.78% | 63.99 |
| Gated | 5 | 62.91% | 59.71 (47.9% became ±2 outliers) |
| Gated | 20 (default) | 17.10% | 49.22 |
| Gated | 100 | 3.09% | 18.18 |

Readings:

- The surprise gate cuts flips from **63% to 3%** (~20× fewer memory
  writes per update pass) and *improves* held-out CE monotonically
  (64 → 18): the gate wins on both axes.
- Absolute numbers are far from the paper baselines (~6.9) because this
  uses `train.py --mode stochastic` defaults (200 steps, LR 1e-3) — compare
  **relative within the table only**.
- Caveat / open question: tightening the gate moves the ternary core toward
  its random initialization (sparsity 30.5% at thr=100 vs 33.5% at thr=20),
  so CE improves because flips *stop destroying* structure, not because gated
  learning *adds* value where it fires. The hypothesis "learn where surprised
  beats a frozen core" (CE_active < CE_frozen) is not yet demonstrated;
  the current sign-grad flip rule is net destructive (densification
  50%→33% sparsity + randomization).
  - **RESOLVED (Exp 7 rerun at the correct logit scale):** continuing the
    Phase-1 backprop checkpoint on fineweb shard 0 for 300 blocks, the plain
    sign-grad flip rule (thr=20, no energy/sparsity) **beats a frozen core**
    on both axes (10000 pos, lr_emb 1e-5): slice **7.0175 vs 7.7022**
    (active −0.68), fineweb **8.8644 vs 9.0954** (active −0.23). On a
    *trained* Phase-1 checkpoint the sign-grad rule is **additive, not
    destructive** — it holds TinyStories retention while embedding-only SGD
    forgets (+0.09 slice). The earlier "net destructive" verdict held only for
    flips from a random init (Exp 1's own setup); from a trained base, "learn
    where surprised" is demonstrated. (Caveat applies to from-init stochastic
    training; the resolution is continual learning — the two are different
    regimes.)

Code changes: `quantization.apply_bit_flips(ungated=, stats=)`,
`StochasticTernaryLinear.ungated` + flip counters, `train.py --flip-ungated`,
`tests/eval_ternary_ablation.py --scale`.

Reproduce:

```bash
python train.py --mode stochastic --preset tiny --steps 200 --data-cache tinydata --save-dir checkpoints_exp1_g20
python train.py --mode stochastic --preset tiny --steps 200 --data-cache tinydata --flip-ungated --save-dir checkpoints_exp1_ungated
python tests/eval_ternary_ablation.py --checkpoint checkpoints_exp1_g20/checkpoint_000200.pt --scale 1.0
python tests/eval_ternary_ablation.py --checkpoint checkpoints_exp1_ungated/checkpoint_000200.pt --scale 1.0
```

## Experiment: Continual Learning & Catastrophic Forgetting (Exp 2)

Question: when a Phase-1 model is switched to a NEW domain, do error-gated
SBF local rules adapt with less **catastrophic forgetting** of the old domain
than standard backprop fine-tuning?

Setup:
- **Phase 1** (old domain): `checkpoints_bp/checkpoint_000200.pt` — the
  200-step backprop SBF baseline trained on TinyStories (`tinydata`). Slice CE
  7.069 on `sliceEval100k.bin` (TinyStories held-out), domain CE 8.284 on
  `data_teacher` (chat).
- **Phase 2** (new domain): fine-tune on `data_teacher/` (2144 chat tokens,
  teacher-generated conversations, manifest format), 100 steps, same
  checkpoint start, accumulators reset on load (replaying Phase-1 acc state is
  unsafe — finding #10/#12).
  - **Method 1**: backprop fine-tune (`train_baseline_backprop.py --resume`).
  - **Method 2**: SBF error-gated local rules, rule `c`
    (`train_discrete.py --load-checkpoint --rule c --logit-scale 1.0`).
- Metric: **SliceEval100k CE** (old-domain forgetting, higher = worse) and
  domain CE (new-domain adaptation, lower = better).

| Method | Phase-2 steps | Slice CE | Δ slice (forgetting) | Domain CE | Δ domain (adaptation) |
|---|---|---|---|---|---|
| Phase-1 baseline | 0 | 7.069 | 0 | 8.284 | 0 |
| **M1 backprop** (flips on, LR 3e-4) | 100 | **23.99** | **+16.9** | 15.93 | +7.6 |
| M1 backprop (flips on, LR 1e-4) | 100 | 24.32 | +17.2 | 15.82 | +7.5 |
| **M2 local rules** (rule c, seed 0) | 100 | **8.34** | **+1.27** | 8.37 | +0.09 |
| M2 local rules (rule c, seed 1) | 100 | 8.44 | +1.37 | 8.38 | +0.10 |
| M2 local rules (rule c, toggle) | 300 | 9.20 | +2.13 | 8.26 | −0.03 |
| Control: backprop, ternary frozen | 100 | 8.00 | +0.93 | **6.63** | **−1.66** |

Readings:

- **Catastrophic forgetting is a backprop-flips failure mode, not an SBF
  one.** Standard backprop fine-tune on the new domain destroys TinyStories
  knowledge (slice CE 7.07 → 24.0, +17 nats — worse than the random baseline
  9.01). Mechanism: backprop's *ungated* ternary flips on new-domain gradients
  mass-kick the old weights (the same matrix-inversion seen when replaying
  accumulator state, finding #10/#12). Error-gated local rules forget ~13×
  less (slice CE 8.34, +1.3 nats) because their sign-grad flips are sparse and
  threshold-gated.
- **But local rules barely *adapt* on this tiny domain**: domain CE stays
  ~8.3 (Δ≈0) for rule `c` — the 2144-token teacher set is too small for the
  local delta to accumulate meaningful signal in 100 steps. Toggle adds more
  flips but just erodes the old domain further (slice 9.20, domain 8.26).
- **The best-forgetting+adaptation tradeoff is the frozen-ternary control**:
  backprop on the embedding only (ternary flips disabled) forgets +0.93 and
  adapts domain CE 8.28 → 6.63 (−1.66). So on a small new domain the model's
  FP32 embedding can absorb the distribution shift; the ternary core is what
  backprop fine-tuning destroys.
- Caveat: data_teacher is a 2144-token POC (14 conversations). The forgetting
  asymmetry (M1 +17 vs M2 +1.3) is robust across LR and seeds, but the
  adaptation result for M2 needs a larger new-domain set (see
  `scripts/generate_teacher_data.py`, target ~2M tokens).

Code changes: `DiscreteTrainer.load_checkpoint()` (Phase-1 resume, accumulators
reset, FP16→FP32), `train_discrete.py --load-checkpoint/--logit-scale` +
manifest.json (multi-source) data, `train_baseline_backprop.py
--resume/--lr-domain/--domain-eval` + manifest data + accumulator reset.

Reproduce:

```bash
# Method 1: backprop fine-tune on the new domain (with SBF flips)
python train_baseline_backprop.py --resume checkpoints_bp/checkpoint_000200.pt \
    --steps 100 --data-cache data_teacher --domain-eval data_teacher/teacher_poc_0000.bin \
    --lr-domain 3e-4 --batch-size 8 --save-dir checkpoints_exp2_bp

# Method 2: SBF error-gated local rules on the new domain
python train_discrete.py --rule c --steps 100 --data-cache data_teacher \
    --load-checkpoint checkpoints_bp/checkpoint_000200.pt --logit-scale 1.0 \
    --batch-size 8 --save-dir checkpoints_exp2_local

# Control: backprop, ternary flips frozen (embedding-only fine-tune)
python train_baseline_backprop.py --resume checkpoints_bp/checkpoint_000200.pt \
    --steps 100 --data-cache data_teacher --domain-eval data_teacher/teacher_poc_0000.bin \
    --lr-domain 3e-4 --batch-size 8 --no-flips --save-dir checkpoints_exp2_bp_nf
```

## Experiment: Energy Accumulator + Adaptive Threshold (Exp 3)

Question: can the ternary core actually *learn* on a new domain without
destroying the old one — by redesigning the flip mechanics (Direction 1)?

Motivation (from Exp 1/2): the sign-grad flip rule (`acc += -sign(grad)`, ±1
votes) is net destructive — backprop fine-tune on a new domain forgets
catastrophically (+16.9 nats slice, Exp 2 M1) because *ungated* flips mass-kick
the ternary weights on new-domain gradients. The gate only helped by *stopping*
flips (Exp 1).

Fix (this experiment): two coupled changes to `quantization.py`:

1. **Energy accumulator** (`--acc-energy`): the backward pass accumulates a
   leaky EMA of the *negative gradient* instead of ±1 sign votes:
   `acc = acc_decay*acc - grad_w`. Magnitude now carries the signal (a weight
   only flips when the accumulated gradient *energy* is large), and the decay
   makes old-domain energy fade naturally over the new domain.
2. **Adaptive threshold** (`--adaptive-thr k`): the flip threshold is computed
   per output channel as `tau = k * RMS(acc)` instead of a fixed integer (20).
   Scale-invariant — `k` is an interpretable knob on the flip *budget*
   (measured: k=2 → ~2% of weights/pass, k=3 → ~0.1%, k=4 → ~0.0%).

Setup: same as Exp 2 — continue `checkpoints_bp/checkpoint_000200.pt`
(Phase-1 TinyStories, slice CE 7.069, domain CE 8.284) on `data_teacher`
(2144 chat tokens), 100 steps, `--lr-domain 1e-4 --batch-size 8`, accumulator
reset on load. Same slice/domain CE metrics. **The frozen-ternary control was
re-run at the SAME LR (1e-4)** so the comparison is apples-to-apples (the
Exp-2 frozen control used 3e-4).

| Run | Steps | Slice CE (Δ) | Domain CE (Δ) | Ternary bits flipped |
|---|---|---|---|---|
| Phase-1 baseline | 0 | 7.069 (0) | 8.284 (0) | — |
| Exp2 M1 backprop (flips) | 100 | 23.99 (+16.9) | 15.93 (+7.6) | — |
| Exp2 M2 local rules | 100 | 8.34 (+1.27) | 8.37 (+0.09) | — |
| **Exp3 frozen control (1e-4)** | 100 | **7.36 (+0.29)** | **7.40 (−0.90)** | 0 |
| Exp3 k=2 | 100 | 10.58 (+3.51) | 9.21 (+0.93) | 1.24% |
| Exp3 k=3 | 100 | 7.34 (+0.27) | 7.34 (−0.95) | 0.053% |
| Exp3 k=4 | 100 | 7.36 (+0.30) | 7.40 (−0.89) | 0.0025% |

Readings (honest):

- **The "adaptation" is the FP32 embedding, not the ternary core.** At k=3/k=4
  the ternary core is effectively frozen (0.05%/0.003% of bits flipped over 100
  steps) and the results are statistically identical to the frozen-ternary
  control at the same LR (slice +0.27 vs +0.29; domain −0.95 vs −0.90). The
  adaptive threshold does not make the ternary core learn — at these k values
  it makes the ternary core *stop moving entirely*, and the FP32 embedding
  absorbs the distribution shift (same mechanism as the Exp-2 frozen control).
- **k is a safety dial, not a learning rate.** Lower k (k=2) allows more flips
  and they *hurt* (+3.5 nats slice, domain *worse* +0.93). Higher k freezes.
  There is no k where flipping the ternary core beats the frozen control. This
  is consistent with Exp 1: flips are net destructive; every improvement in the
  chain came from *stopping* flips, never from learning via flips.
- **Why the mechanism still matters for continual learning**: it gives a
  principled, scale-invariant way to be *nearly* frozen on the old domain while
  the (FP32) embedding adapts — replacing the manual accumulator reset and the
  magic threshold-20 with one interpretable knob. But it does NOT demonstrate
  that ternary weights can learn via local sign-flip dynamics.

Code changes: `StochasticBitFlipLinear`/`Int8StochasticBitFlipLinear` accept
`acc_decay`/`energy` (accumulate `-grad_w` with leaky decay when energy mode),
`apply_bit_flips(adaptive_thr=)` computes `tau = k*RMS(acc)` per channel,
`StochasticTernaryLinear.set_flip_config()` + model-level `set_flip_config()`
plumb config through, `train.py`/`train_baseline_backprop.py` add
`--acc-energy/--acc-decay/--adaptive-thr`.

Reproduce:

```bash
# Mechanism run
python train_baseline_backprop.py --resume checkpoints_bp/checkpoint_000200.pt \
    --steps 100 --data-cache data_teacher --domain-eval data_teacher/teacher_poc_0000.bin \
    --lr-domain 1e-4 --batch-size 8 --acc-energy --acc-decay 0.99 --adaptive-thr 3.0 \
    --save-dir checkpoints_exp3_k3
# Same-LR frozen control (attribution)
python train_baseline_backprop.py --resume checkpoints_bp/checkpoint_000200.pt \
    --steps 100 --data-cache data_teacher --domain-eval data_teacher/teacher_poc_0000.bin \
    --lr-domain 1e-4 --batch-size 8 --no-flips --save-dir checkpoints_exp3_ctrl1e4
```

## Experiment: Applying the Mechanism to the Discrete Trainer (Exp 4)

Question (Direction 1, choice a): porting the energy-accumulator + adaptive-τ
mechanism to the on-device local-rule trainer (`train_discrete.py`) — can
**backprop-free** continual learning adapt to a new domain while resisting
forgetting?

Two missing halves had to be added to the discrete pipeline:

1. **`--rule-energy`**: the local rules (`predictive_coding_delta`,
   `forward_forward_delta`, etc.) returned `±sign(grad)` — discarding magnitude,
   so the accumulator held uniform ±1 votes with `|acc|/RMS < 1.2` (no heavy
   tail for the adaptive threshold to select). With `--rule-energy` they return
   `±grad` (magnitude-weighted), giving `max |acc|/RMS ≈ 4.5` like backprop.
2. **`--adaptive-thr k`** (plus existing `acc_decay`): `tau = k*RMS(acc)` per
   output channel, as in Exp 3. Wired into `apply_bit_flips` in the discrete
   trainer's flip pass.

Setup: continue the same Phase-1 checkpoint on `data_teacher`, 100 steps,
`--batch-size 8 --logit-scale 1.0`. **Control: flips disabled entirely
(`--flip-every-n 100000`) — embedding-only adaptation.**

| Run | Slice CE (Δ) | Domain CE (Δ) | Ternary bits flipped |
|---|---|---|---|
| Phase-1 baseline | 7.069 (0) | 8.284 (0) | — |
| Exp2 M2 local rules (no mechanism) | 8.34 (+1.27) | 8.37 (+0.09) | — |
| **Exp4 control (flips off, embedding only)** | **7.85 (+0.78)** | **6.29 (−2.01)** | 0 |
| Exp4 rule-c energy k=3 | 7.85 (+0.78) | 6.29 (−2.01) | 0.0001% |
| Exp4 rule-c energy k=2 | 7.87 (+0.80) | 6.31 (−1.99) | 0.03% |
| Exp4 rule-c (no energy) k=3 | 7.85 (+0.78) | 6.29 (−2.01) | 0 |

Readings (honest):

- **The domain adaptation (−2.01 nats) is 100% the FP32 embedding.** The
  flips-off control achieves exactly the same slice/domain CE as every
  `--rule-energy --adaptive-thr` run, because at k≥2 the ternary core is
  effectively frozen (≤0.03% of bits flipped). The discrete pipeline's
  *embedding* adapts to the new domain as well as (slightly better than) the
  backprop frozen control (−2.01 vs −0.90), with no backprop through the
  ternary core at all.
- **The local-rule ternary core still cannot learn the new domain.** With the
  embedding frozen (`--no-train-embedding`) and k=1.5 (1.5% of bits flipped),
  slice CE holds (+0.45, good) but domain CE gets *worse* (+1.01) — the local
  deltas carry no usable new-domain signal at this data scale (2144 tokens).
  This is a data-scale / rule-signal limitation, not a flip-mechanics one.
- **Conclusion across Exp 1-4**: at the 2144-token teacher scale, the bit-flip
  mechanism, in every variant tested (sign-gate, energy acc, adaptive τ,
  backprop or local rules), is a *preservation* tool, not a *learning* tool —
  the gains all come from stopping destructive flips + letting the FP32
  embedding adapt. **But Exp 5 shows this was a data-scale artifact**: at real
  scale (534M tokens) the energy + adaptive-τ mechanism *does* make the ternary
  core learn (k=3: 5.910 vs frozen 6.892, 1% of bits flipped, stable).

Code changes: `predictive_coding_delta`/`forward_forward_delta`/`hebbian_delta`/
`entropy_delta` take `energy=` (return ±grad instead of ±sign(grad)),
`DiscreteConfig.rule_energy` + `_feed_local_deltas`/`_accumulate_target`
thread it through, `train_discrete.py --rule-energy/--adaptive-thr`.

Reproduce:

```bash
# Mechanism run (matches flips-off control -> attribution = embedding)
python train_discrete.py --rule c --steps 100 --data-cache data_teacher \
    --load-checkpoint checkpoints_bp/checkpoint_000200.pt --logit-scale 1.0 \
    --batch-size 8 --acc-decay 0.99 --adaptive-thr 3.0 --rule-energy \
    --save-dir checkpoints_exp4_de_k3
# Flips-off control
python train_discrete.py --rule c --steps 100 --data-cache data_teacher \
    --load-checkpoint checkpoints_bp/checkpoint_000200.pt --logit-scale 1.0 \
    --batch-size 8 --flip-every-n 100000 --save-dir checkpoints_exp4_ctrl
```

## Experiment: Flip Mechanism at Scale (Exp 5)

Question: Exp 3/4 concluded the ternary core is a *preservation* tool, not a
learning tool — but that was on the 2144-token teacher POC. At real data scale
(TinyStories, 534M tokens, same domain the Phase-1 model was trained on), does
the energy-accumulator + adaptive-τ mechanism let the ternary core actually
*learn* (reduce held-out CE beyond what the FP32 embedding alone achieves)?

Setup: continue `checkpoints_bp/checkpoint_000200.pt` (slice CE 7.069) on
`tinydata`, 1000 steps, `--lr-domain 1e-4 --batch-size 8`. Frozen control =
`--no-flips` (embedding/lm_head train, ternary frozen). Measured: held-out
`sliceEval100k.bin` CE + % of ternary bits changed vs Phase-1.

| Run | Slice CE @1000 (Δ) | Ternary bits flipped | Trajectory |
|---|---|---|---|
| Phase-1 baseline | 7.069 (0) | — | — |
| **Frozen control** | **6.892 (−0.177)** | 0% | slow, smooth |
| Energy k=4 | 6.863 (−0.206) | 0.05% | ≈ frozen |
| **Energy k=3** | **5.910 (−1.159)** | **1.04%** | monotonic, stable |
| Energy k=2 | 6.256 (−0.813) | 11.6% | unstable (spikes to 6.59) |
| Sign-vote (fixed thr 20) | 5.605 (−1.464) | 90.3% | catastrophic transient (13.8 @ step 100), recovers |

Readings (honest):

- **The mechanism IS a learning tool at scale — this overturns the Exp 3/4
  conclusion.** At k=3 the ternary core beats the frozen control by **0.98
  nats** (5.910 vs 6.892) while flipping only 1.04% of its bits. The Exp 3/4
  "preservation-only" result was a data-scale artifact: 2144 tokens carry no
  learnable signal for 6.3M ternary bits, 534M tokens do.
- **k=3 is the sweet spot and it is stable.** Monotonic decrease, no
  catastrophic transient — exactly the property continual learning needs
  (unlike sign-vote's 13.8-nats spike).
- **The old sign-vote mechanism "wins" on final CE only by mass-destroying and
  re-learning.** It flips 90% of bits, blowing up CE to 13.8 at step 100, then
  recovers *because the training data is the same domain*. On a genuinely new
  domain that transient is exactly Exp 2 M1's catastrophic forgetting (+16.9
  nats) — unrecoverable.
- **k controls the flip budget as designed**: k=4 → 0.05% flips ≈ frozen;
  k=3 → 1% (best); k=2 → 11.6% (too aggressive, unstable). The energy +
  adaptive-τ mechanism with `k` is a genuine, interpretable "ternary learning
  rate" at scale.

Reproduce:

```bash
# Frozen control
python train_baseline_backprop.py --resume checkpoints_bp/checkpoint_000200.pt \
    --steps 1000 --data-cache tinydata --lr-domain 1e-4 --batch-size 8 \
    --no-flips --save-dir checkpoints_exp5_ctrl
# Energy accumulator + adaptive threshold
python train_baseline_backprop.py --resume checkpoints_bp/checkpoint_000200.pt \
    --steps 1000 --data-cache tinydata --lr-domain 1e-4 --batch-size 8 \
    --acc-energy --acc-decay 0.99 --adaptive-thr 3.0 --save-dir checkpoints_exp5_k3
```

## Experiment: Real New-Domain Continual Learning (Exp 6)

Question: Exp 5 showed the mechanism learns on the *same* domain at scale. The
original research question — does the ternary core adapt to a genuinely NEW
domain while preserving the old one — could not be answered on the 2144-token
teacher POC. Does it hold on a real domain shift at scale?

Setup: `data/fineweb_10bt/` (494 shards × 25M tokens = 12.3B tokens, same BPE
vocab 8192 — a genuine domain shift vs Phase-1 TinyStories). Added
`manifest.json` (multi-source format, ratio 1.0) + carved a held-out eval slice
from the last shard (`examples/continual/fineweb_eval100k.bin`, committed).
Continue the same Phase-1 checkpoint, 1000 steps, `--lr-domain 1e-4 --batch-size 8`, on
fineweb. Phase-1 baseline: slice 7.069, **domain (fineweb) 8.922** (~1.85 nats
harder — real shift).

| Run | Slice CE (retention) Δ | Domain CE (adaptation) Δ | Ternary bits flipped |
|---|---|---|---|
| Phase-1 baseline | 7.069 (0) | 8.922 (0) | — |
| Frozen control (embedding only) | 7.989 (**+0.92**) | 7.838 (−1.08) | 0% |
| **Energy k=3** | **7.362 (+0.29)** | **6.915 (−2.01)** | 1.03% |

Readings (honest):

- **The mechanism wins on BOTH axes at once — the decisive result of the
  chain.** Energy k=3 adapts to fineweb nearly twice as well as the frozen
  control (domain −2.01 vs −1.08) while forgetting **~3× less** on TinyStories
  (slice +0.29 vs +0.92). It is not a preservation-vs-adaptation tradeoff —
  flipping 1% of ternary bits *simultaneously* improves new-domain CE and
  protects old-domain CE, because the frozen control's only adaptation channel
  (the FP32 embedding) drifts the shared embedding toward fineweb and hurts
  TinyStories.
- **This answers the original research question**: ternary core + energy acc +
  adaptive τ = genuine on-device continual learning across a real domain shift.
  The Exp 2 M1 (sign-vote backprop) result (+16.9 nats slice, catastrophic) is
  not the mechanism's fault — it was the flip *rule* (90% of bits, step-100
  transient). With energy + adaptive threshold at ~1% budget the ternary core
  is a stable learning tool.
- **Fixed overhead**: no extra params, no replay buffer, no per-task gates —
  the mechanism is the leaky accumulator + per-channel adaptive threshold
  already shipped in Exp 3.

Reproduce:

```bash
# Frozen control (embedding-only adaptation channel)
python train_baseline_backprop.py --resume checkpoints_bp/checkpoint_000200.pt \
    --steps 1000 --data-cache data/fineweb_10bt \
    --eval-slice examples/discrete/sliceEval100k.bin \
    --domain-eval examples/continual/fineweb_eval100k.bin \
    --lr-domain 1e-4 --batch-size 8 --no-flips --save-dir checkpoints_exp6_ctrl
# Energy accumulator + adaptive threshold (ternary core learns the new domain)
python train_baseline_backprop.py --resume checkpoints_bp/checkpoint_000200.pt \
    --steps 1000 --data-cache data/fineweb_10bt \
    --eval-slice examples/discrete/sliceEval100k.bin \
    --domain-eval examples/continual/fineweb_eval100k.bin \
    --lr-domain 1e-4 --batch-size 8 \
    --acc-energy --acc-decay 0.99 --adaptive-thr 3.0 --save-dir checkpoints_exp6_k3
```


## Experiment: Gradient-Free Continual Learning in C++ (Exp 7)

Question: Exp 6 proved the energy-accumulator + adaptive-threshold mechanism wins on both axes
(new-domain CE and old-domain retention) — but that run used **STE backprop gradients** (heavy
tail). The C++ runtime is the real product: **gradient-free** rule 'c' on the device. Does the
mechanism survive without backprop?

Setup: same Phase-1 checkpoint, exported to v6 (`--sl-reset-acc`), trained with
`selflearn_avx2.exe --energy --adaptive-thr 3.0` on fineweb shard 0 for 300 blocks (~214K
tokens). Control = embedding-only (`--no-ternary`, no flips).

> **Correction (logit-scale bug):** an earlier version of this section reported baseline slice
> 8.8564 / fineweb 8.9680 and small deltas (−0.232/−0.112). Those numbers came from
> `export_model.py` always writing `sl_logit_scale = 1/sqrt(hidden_dim) = 0.0625` — the discrete
> trainer's calibration — **even for STE backprop checkpoints that train in the natural regime
> (scale=1.0)**. The C++ runtime then compressed every softmax toward uniform, inflating all CE
> by ~+1.8 nats and shrinking the embedding gradient's effective LR by ~√V. This silently
> **under-reported the mechanism**: the qualitative Exp 7 result survives at the correct scale
> with larger deltas. Fixed in `export_model.py` (STE → scale 1.0; discrete → 1/sqrt(H);
> `--sl-logit-scale` override).
>
> **Correction 2 (eval build):** the numbers in this section were originally measured with the
> 8/8 C++ build, which still clamped attention scores to ±80. The clamp was removed in commit
> 56ce183 ("Fix C++/Python eval gap", 8/8 20:30) to match the Python runtime (which never
> clamps) — this shifts the absolute CEs (old-build baseline slice 7.6135 → 8.1814). The table
> below has been re-run with the current build (`selflearn_avx2.exe --eval … 10000`, scale 1.0);
> the qualitative result is unchanged. The 8/8 15:51 v6 exports also carried a stale
> `sl_logit_scale=0.0625`; the `checkpoints/exp7_*.bin` files have been re-patched to scale 1.0
> (originals kept as `.bak`).

**Embedding LR at the natural scale:** at `sl_lr_embedding=1e-4` (the value tuned for the
compressed scale) the natural-regime gradient concentrates full magnitude on the target row, so
300 blocks of embedding SGD **catastrophically destroy** the model (held-out CE ~12.2, worse
than the 9.01 random floor, on both axes). With `--sl-lr-embedding 1e-5` the run is stable.
Baseline CE (10000 pos, scale 1.0): slice **8.1814**, fineweb **9.3376**.

| Run (lr_emb 1e-5) | Slice CE Δ (retention) | Fineweb CE Δ (adapt) |
|---|---|---|
| Control (embedding-only, 500 blk) | 8.2533 (+0.07) | 9.2038 (−0.13) |
| Energy k=3 + sparsity 0.01 (300 blk) | 7.8767 (−0.31) | 9.0853 (−0.25) |
| **Energy k=3 + sparsity 0.01 (500 blk)** | **7.8244 (−0.36)** | **8.9774 (−0.36)** |

Readings (honest):

- **The mechanism works gradient-free — with sparsity.** Plain `--energy --adaptive-thr 3.0`
  flips nothing on rule 'c' (0 real changes; accumulator grows unbounded, τ grows with it, a
  permanent 0-flip freeze). This is the exact pitfall the Python sign-vote warning anticipated,
  now seen on magnitude accumulators too. **Top-k sparsification fixes it**: concentrating energy
  on the top-1% per-row gradient gives the accumulator the heavy tail the adaptive τ selects.
- **Both axes again, now on-device — and the win grows with budget.** Energy k=3 + sparsity
  adapts to fineweb **2.7× more than the embedding-only control** (−0.36 vs −0.13) **and
  protects the old domain where control loses it** (−0.36 vs +0.07 slice: the ternary core holds
  TinyStories retention while embedding-only SGD forgets). Going 300 → 500 blocks keeps slice
  retention protected (−0.31 → −0.36, slightly deeper, no tradeoff) while fineweb adaptation
  deepens (−0.25 → −0.36).
  Same qualitative win as Exp 6's STE run, with larger deltas than the buggy-scale readings.
- **Scope honesty**: 500 blocks is still a much smaller budget than Exp 6's 1000 steps over
  12.3B tokens, and both runs also benefit from the embedding SGD channel. The absolute deltas
  are small; the *relative* mechanism-vs-control win on both axes is the result. Gradient-free is
  noisier/slower than STE (larger flip budget per useful bit) but the direction matches.

Reproduce:

```bash
python inference/export_model.py checkpoints_bp/checkpoint_000200.pt -o checkpoints/exp7_v6.bin \
    --self-learning --sl-reset-acc --sl-energy --sl-adaptive-thr 3.0 --sl-sparsity 0.01 \
    --sl-lr-embedding 1e-5
cd inference && build.bat avx2
selflearn_avx2.exe ..\checkpoints\exp7_v6.bin ..\data\fineweb_10bt\fineweb_0000.bin ..\checkpoints\exp7_trained.bin 500 100 0
selflearn_avx2.exe --eval ..\checkpoints\exp7_trained.bin ..\examples\continual\fineweb_eval100k.bin 10000
selflearn_avx2.exe --eval ..\checkpoints\exp7_trained.bin ..\examples\discrete\sliceEval100k.bin 10000
```

## Experiment: Soft-to-Hard Quantization Warmup (Exp 8)

Question: hard round(W/Δ) + STE gives zero gradient inside the dead zone — a weight sitting in
(-0.5Δ, +0.5Δ) receives no signal to leave it, and the discrete jump at ±0.5/±1.5 adds noise
that early training must fight. Does replacing the hard round with a continuous surrogate
during a short warmup, then annealing it back to hard round, let STE training start cleaner?

Surrogate (exact, 5-level boundaries {-2,-1,0,+1,+2} matched to the v7 encoding):

    Q~(x; γ) = -2 + σ(γ(x+1.5)) + σ(γ(x+0.5)) + σ(γ(x-0.5)) + σ(γ(x-1.5)),  x = W/Δ

Backward passes the **exact** surrogate gradient dQ~/dx = γ·Σ_k σ(z_k)(1-σ(z_k)) chained
through x_n = W/Δ, **including the gradient of Δ** (no detach), verified by
`torch.autograd.gradcheck` at γ ∈ {2, 25} × per-tensor/per-channel. γ follows a log-linear
schedule 2 → γ_max over the warmup window, then switches to hard round + STE (γ=None) —
the deployment path is untouched (`TernaryQuantizer` round in `export_model.py`).

Usage:

```bash
# baseline: hard round + STE from step 0
python train.py --preset tiny --steps 300 --data-cache tinydata --mode ste
# soft-to-hard: γ 2→50 log-linear over 75 steps (25%), then hard STE
python train.py --preset tiny --steps 300 --data-cache tinydata --mode ste \
    --soft-quant-gamma --soft-quant-steps 75
# hybrid: γ 2→25 over 30 steps (10%), then hard STE
python train.py --preset tiny --steps 300 --data-cache tinydata --mode ste \
    --soft-quant-gamma --soft-quant-steps 30 --soft-quant-gamma-max 25
```

Results (300 steps, preset tiny, tinydata, STE):

| Run | Final train CE @300 | Note |
|---|---|---|
| Baseline (hard STE) | **5.8817** | — |
| Soft (γ 2→50, 75 steps) | 8.2548 | gradient dies at γ→50 (σ' → 0), wastes warmup tail |
| Hybrid (γ 2→25, 30 steps) | 5.9553 | +0.074 vs baseline |

Readings (honest):

- **No loss spike at the switch point** — the first run's step-75 handoff to hard STE showed
  no discontinuity (soft 22.5 → 20.1 vs baseline 22.5 → 20.3 around step 70-80), validating
  the sigmoid-surrogate convergence claim.
- **Vanishing gradient from the saturating surrogate is real**: pushing γ to 50 collapses
  σ'(x) ≈ 0 for nearly all weights before warmup ends; the model flatlines and the run loses
  budget it never recovers (8.25 vs 8.05 in the first comparison).
- **Hybrid fixes the decay but not the phase cost**: γ_max=25 keeps a usable slope
  (σ'(25×0.1) ≈ 0.1), and the hybrid run actually *overtakes* baseline at step ~200
  (8.00 vs 8.16) after trailing badly in the first 50 steps. The residual −0.074 at step 300
  is the phase-1 debt: the warmup occupies the highest-LR region of a 300-step cosine
  schedule, and switching to hard round at step 30 shocks weights that have only seen the
  soft surrogate.
- **300 steps is too short to judge**: on a production 15k-step budget the warmup is 0.2%
  of training instead of 10%, and the mid-training crossover seen here (hybrid > baseline
  from step ~200) would dominate. The mechanism is correctly implemented (gradcheck-clean,
  exact Δ-gradient, no deployment change); its value proposition is a longer-horizon question.

**Post-mortem diagnostic (final answer):** the surrogate changes *nothing* about the final
weight distribution. Loading both step-300 checkpoints and binning every latent weight by
quantization region (`scripts/diagnose_soft_quant.py`):

| Region | Baseline (hard STE) | Hybrid (γ 2→25, 30 steps) |
|---|---|---|
| Dead zone (\|x_n\| < 0.5) | 17.9% | 17.9% |
| Plateau ±1 (0.5–1.5) | 35.7% | 35.9% |
| Outlier ±2 (≥1.5) | 46.4% | 46.2% |

Identical within noise, zero near-zero rows in both. The dead-zone-escape hypothesis is
falsified: STE's pass-through gradient is *not* zero inside the dead zone, so weights leave
it on their own — there was nothing for the surrogate to rescue. Every cost it paid (highest-LR
phase, plateau starvation, boundary slope spikes, Δ-gradient noise) was pure loss with no
measurable benefit. **Exp 8 verdict: null result — hard round + STE is not improved by a
smooth-surrogate warmup, and the code is retained only as a documented dead end.**

(Aside: 46% of weights land in the ±2 outlier band at `--ternary-scale 0.7`, i.e. the v7
side-channel sign blob covers ~half the matrix — a scale >0.9 would push the distribution
back toward ±1 and shrink the side channel; untested.)

## Experiment: Ternary-Scale Tuning (Exp 9)

The Exp 8 diagnostic surfaced a suspicious distribution: at `--ternary-scale 0.7` (default)
**46% of trained weights sit in the ±2 outlier band**, so the v7 side-channel sign blob covers
nearly half the matrix. Since Δ = scale × mean|W|, a larger scale widens the ±1 plateau and
should shift the distribution back toward {-1, 0, +1} — while also changing what the model
can represent. Sweep (300 steps, preset tiny, tinydata, STE, `scripts/diagnose_soft_quant.py`
for occupancy):

| Scale | Final train CE @300 (2 seeds) | Dead zone | Plateau ±1 | Outlier ±2 |
|---|---|---|---|---|
| 0.7 (default) | 5.8817 / 5.7939 (mean 5.838) | 17.9% | 35.7% | 46.4% |
| 0.9 | 5.8184 / — | 23.0% | 45.7% | 31.3% |
| **1.0** | 5.7844 / 5.8138 (mean 5.799) | 25.5% | 50.4% | 24.1% |
| 1.3 | 5.8671 / — | 33.1% | 61.8% | 5.2% |

Readings (honest):

- **The occupancy shift is deterministic, not seed luck**: re-running with `--seed 1` gives
  byte-identical occupancy (0.7: 17.9/35.7/46.4; 1.0: 25.5/50.4/24.1). The scale→distribution
  mapping is structural: Δ = scale·mean|W| pins the region boundaries to a fixed equilibrium.
- **The CE win is real but smaller than the first seed suggested**: two-seed means are 5.838
  (0.7) vs 5.799 (1.0), a −0.039 delta that sits inside per-seed spread (±0.05). At the
  300-step budget the CE comparison is not conclusive; the structural side-channel win
  (46.4% → 24.1% outlier share, ~half the side blob) is the solid result.
- **There is a real optimum, not a monotone direction**: 1.3 overturns the trend (5.8671) as
  the dead zone grows to 33% (scale 1.3: 33.1% dead / 61.8% plateau / 5.2% outlier) — too many
  weights quantize to 0 and the model loses representational capacity. Optimum ≈ 1.0–1.1.
- The decisive test is the production 15k-step budget (variance shrinks relative to the CE
  scale) and/or the C++ runtime benchmark before promoting 1.0 to the default.

Reproduce:

```bash
python train.py --preset tiny --steps 300 --data-cache tinydata --mode ste --ternary-scale 1.0 \
    --seed 1 --save-dir checkpoints_exp9_s10_seed1
python scripts/diagnose_soft_quant.py checkpoints_exp9_s10_seed1/checkpoint_000300.pt "SCALE 1.0"
```

(Note: `train.py` gained a `--seed` flag during Exp 9 — earlier runs used unseeded RNG, so
their CE values carry unknown but bounded seed variance; occupancy is seed-invariant.)

**Verdict: default `--ternary-scale` promoted 0.7 → 1.0** — the CE comparison is neutral-to-
slightly-better (mean −0.039) and the side-channel shrink is deterministic, so the new
default is strictly safer than the old one. To reproduce the old behavior pass
`--ternary-scale 0.7`.

## Experiment: True-Value Outlier Training (v8-forward, Exp 10)

The v8 format section measured a **regression** when true-value outliers are applied at
export time only (+0.15 nats): STE training clamps outliers at ±2, so replacing ±2 with the
latent true value at inference is a distribution shift. This experiment closes the gap the
other way — **train with the exact v8 dequant**, so the model can genuinely use real outlier
magnitude: `--v8-forward` makes the STE forward dequantize outliers to
`round((W/Δ)·32)/32` (the v8 blob encoding) instead of clamping at ±2. Backward is
unchanged (plain STE).

Setup: same as Exp 9 (tiny, 300 steps, tinydata, ternary-scale 1.0, seed 1) + `--v8-forward`.
Three arms, measured with the C++ runtime (`selflearn --eval`, 40K positions, logit scale 1.0):

| Arm | sliceEval100k @40K | tinydata @1M window @40K |
|---|---|---|
| baseline (clamp ±2) → v7 export | 5.7604 | 5.8748 |
| v8-forward → v7 export (clamp ±2) | 5.8556 (+0.095) | 5.9816 (+0.107) |
| **v8-forward → v8 export (true values)** | **5.7404 (−0.020)** | **5.8578 (−0.017)** |

Readings (honest):

- **The train/infer gap is real and bidirectional.** A clamp-±2-trained model read with true
  values loses +0.15 (v8 section); a true-value-trained model read with clamp-±2 loses
  +0.10. Each representation is only as good as the training it was calibrated for.
- **v8-forward + v8 export beats the v7 baseline — consistently but by a small margin.**
  −0.017/−0.020 nats on two independent 40K windows; the direction agrees in both, so it
  is beyond sampling noise at this size, yet far below the ±3.97 headroom the true-value
  channel theoretically offers.
- **Training dynamics are unaffected**: final train loss 5.7995 vs baseline seeds
  5.784/5.814 — v8-forward costs nothing during training.

Code: `FusedTernaryLinear` `v8_forward` flag, `TernaryLinear.set_v8_forward()`,
`train.py --v8-forward` (STE only), `export_model.py` requires `--v8` for v8-forward
checkpoints (header/body version mismatch would corrupt the format otherwise).

Reproduce:

```bash
python train.py --mode ste --preset tiny --steps 300 --data-cache tinydata \
    --ternary-scale 1.0 --seed 1 --v8-forward --save-dir checkpoints_exp10_v8f
python inference/export_model.py checkpoints_exp10_v8f/checkpoint_000300.pt -o exp10_v8.bin --v8
python inference/export_model.py checkpoints_exp10_v8f/checkpoint_000300.pt -o exp10_v7.bin --v7
cd inference && build.bat avx2 && cd ..
inference/selflearn_avx2.exe --eval exp10_v8.bin examples/discrete/sliceEval100k.bin 40000
```

## Experiment: Outlier-Band Regularization (v8-reg, Exp 11) — rejected

Exp 10 left the true-value channel exploited only slightly (−0.02 nats), and the
measured outlier magnitudes sat mostly in the 1.5–2.0 "linger band" right above the
threshold — 83.8% of outliers, mean |W/Δ| = 1.80. Hypothesis: those values waste the
1/32-resolution blob, so forcing them to a target magnitude should let the channel
carry more information. Implemented as a penalty (mean over outliers of
`max(0, target − |W/Δ|)²`, Δ detached, same v8 mask; `train.py --v8-reg LAMBDA
--v8-reg-target T`, STE only). Two strengths run at 300 steps, seed 1:

| Run | final train CE | sliceEval100k @40K (v8) | tinydata @1M @40K (v8) |
|---|---|---|---|
| exp10 v8-forward (no reg) | 5.7995 | 5.7404 | 5.8578 |
| exp11 reg λ=10, target 2.0 | 5.9127 | 5.8454 (+0.105) | 5.9533 (+0.096) |
| exp11 reg λ=40, target 2.0 | 6.0401 | 5.9549 (+0.215) | 6.0757 (+0.218) |

The penalty did exactly what it was designed to do — measured outlier distribution at
step 300 (`scripts/diagnose_v8_reg.py`):

| Checkpoint | outlier % | mean \|W/Δ\| | in (1.5, 2.0) | ≥ 2.0 | max |
|---|---|---|---|---|---|
| exp9 / exp10 (unregularized) | 24.1% | 1.80 | 83.8% | 16.2% | 4.3 |
| exp11 λ=10 | 26.5% | 2.05 | 20.7% | 79.3% | 3.6 |
| exp11 λ=40 | 26.7% | 2.05 | 9.4% | 90.6% | 3.3 |

And every forced shift made the model worse, monotonically in λ, at both train time
(+0.11/+0.24 CE) and inference (+0.10/+0.21 nats on both windows; the gap to the v7
reading also grows — the model leans even harder on true values, yet the absolute CE
degrades).

**Verdict: rejected.** The 1.5–2.0 band is the model's *learned optimum*, not wasted
headroom: at |W/Δ| ≈ 1.8 the blob's ±1/32 step is already ±0.9% relative error, and the
CE loss had 300 steps of free choice over magnitudes — the unregularized exp10 model
already picked its favorite scale. Forcing magnitudes up constrains the weight space
and costs capacity. The true-value channel gains its small win not from bigger
outliers but from the model being allowed to use whatever magnitude it wants; the
flags `--v8-reg` / `--v8-reg-target` remain in `train.py` for completeness but are
documented as a failed experiment. `scripts/diagnose_v8_reg.py` reports the outlier
magnitude distribution of any checkpoint.

## Experiment: Learned 5-Level Codebook (LQ, Exp 12) — null result

Question: the 2-bit format is fixed to levels {-2, -1, 0, +1, +2}·Δ. Exp 10/11 showed the
outlier *magnitudes* are a learned optimum, but the *code values themselves* are the
last non-learnable part of the representation. If each matrix could choose its own
levels, the model should fit better with zero extra binary cost (5 floats per matrix vs
6.3M packed weights). Implemented as a symmetric codebook {−b, −a, 0, a, b} per matrix
(`--lq`, STE only; a, b trainable, init 1.0/2.0 — bit-exact parity with the fixed path
at step 0). Assignment = nearest code with detached index (STE on the latent weights
unchanged); code values train in backward by accumulating the STE gradient per code
(finite-difference verified). Export is refused (`export_model.py` guard — the binary
format has no per-matrix level table yet).

Setup: same as Exp 9/10 (tiny, 300 steps, tinydata, ternary-scale 1.0, seed 1) + `--lq`.

| Run | final train CE | sliceEval100k @39.6K (torch) | tinydata @1M @39.6K (torch) |
|---|---|---|---|
| exp9 baseline (fixed levels) | 5.8138 | 5.8421 | 5.9766 |
| exp12 LQ (learned levels) | 5.8671 (+0.053) | 5.8749 (+0.033) | 6.0195 (+0.043) |

Converged code values (all 36 matrices): a ∈ [0.886, 1.006] (mean 0.983), b ∈ [1.938,
2.066] (mean 1.976).

Readings (honest):

- **The learnable levels do not move.** After 300 steps of free access to level
  positions, every matrix keeps them within 2–6% of the fixed values. The fixed
  {-2,-1,0,1,2} placement was already the optimum of the STE-trained weight
  distribution — the same conclusion as Exp 11 from the other side: the 5-level
  representation itself is well-tuned, and it is the *training* that is the binding
  constraint, not the code values.
- **The small drift that does occur costs CE** (+0.03/+0.04 eval, +0.05 train): free
  capacity here is noise, not signal, at this scale.
- Cost: the per-code backward (5 masked sums per matrix) roughly **doubles training
  time on DML** (23 min vs ~12 min for 300 steps).

**Verdict: null result — learned code values do not help; the fixed levels were already
optimal.** `--lq` remains in `train.py` (documented failed experiment, export guard
stays). `tests/eval_ckpt_ce.py` (added here) evaluates any STE checkpoint on a token
window directly, without export — the tool used for the LQ numbers.

## Experiment: Unbiased Stochastic Rounding (SR, Exp 13) — rejected

Question: deterministic `round(W/Δ)` has a per-element bias (a weight at 1.4 always
quantizes to 1), and STE keeps that bias in the gradient. Standard low-precision
training (FP8) removes it with **stochastic rounding**: round up with probability equal
to the fractional part, so E[Q(x)] = clamp(x, −2, 2) exactly and the STE gradient is
unbiased by construction. Tested as the last training-side quantizer algorithm
(`--sr`, train-only — eval/export stay deterministic; verified: E[Q] = clamp(x) within
0.007 over 2000 samples, eval forward bit-identical to the fixed path).

Setup: same as Exp 9–12 (tiny, 300 steps, tinydata, ternary-scale 1.0, seed 1) + `--sr`.

| Run | final train CE | sliceEval100k @39.6K (torch) | tinydata @1M @39.6K (torch) |
|---|---|---|---|
| exp9 baseline (deterministic round) | 5.8138 | 5.8421 | 5.9766 |
| exp13 SR (unbiased stochastic) | 6.2098 (+0.396) | 6.1788 (+0.337) | 6.3166 (+0.340) |

Readings (honest):

- **Unbiasedness is not free: single-sample variance.** Q(x) has variance p(1−p) with
  p = fractional part — up to 0.25, i.e. a *quarter of a level* in the 1.0-spaced ternary
  grid. The optimizer sees one noisy forward per step, and 300 steps cannot amortize
  that noise: the model converges to a worse point (train +0.40, eval +0.34, both
  windows agree).
- **Why FP8 SR works but 2-bit SR does not**: SR's variance is fixed in *absolute* units
  (≤0.25) while its benefit (bias removal) scales with resolution — at 8-bit the step is
  1/256 of the range, so the variance-to-signal ratio is tiny; at 2-bit the step is 25%
  of the range, so the bias the deterministic quantizer carries is *smaller* than the
  variance SR would inject to remove it. The deterministic bias is also what the
  STE-trained weights are calibrated against (Exp 10–12: magnitudes and levels are
  already at their learned optimum).

**Verdict: rejected — at 2-bit resolution, unbiasedness costs more variance than the
bias is worth.** This closes the quantizer-algorithm line: representation capacity
(Exp 11/12), level placement (Exp 12), and rounding bias (Exp 13) are all already
near-optimal; the binding constraint at this scale is the training budget itself.
`--sr` remains in `train.py` (documented failed experiment).

## Experiment: Annealed Soft Flips (Exp 14)

Question: the SBF flip decision is a hard step (`|acc| > τ`). Exp 12–13 showed the
binding constraint is the *training dynamics*, not the quantizer. Does softening
the flip decision — a sigmoid band around the threshold, annealed toward the
deterministic rule — make the ternary core learn better? (The Gumbel-flip /
surrogate-gradient idea, in flip-decision space: SBF has no forward quantization
noise and no latent weights, so the "surrogate" must live in the flip rule.)

Setup: Exp 5/6 mechanism (`--acc-energy --acc-decay 0.99 --adaptive-thr 3.0`),
1000 steps, `--lr-domain 1e-4 --batch-size 8`, Phase-1 checkpoint
(`checkpoints_bp/checkpoint_000200.pt`, slice CE 7.069). Added `--soft-flip-temp s`:
within the band `[τ(1−2s), τ(1+2s)]` the flip probability is `σ(margin/s)`
(`margin = (|acc|−τ)/(s·τ)`); hard 0 below the band (dead zone — near-zero
accumulators must never flip, or the whole matrix mass-churns) and hard 1 above.
`s = 0.25` cosine-annealed to `0.02` over the run (converges to the deterministic
rule). Both tinydata (same domain, Exp 5 setup) and fineweb (domain shift,
Exp 6 setup) runs.

| Run (tinydata) | Slice CE @1000 (Δ vs 7.069) | Bits flipped |
|---|---|---|
| Frozen control (Exp 5) | 6.892 (−0.177) | 0% |
| **Energy k=3 (Exp 5)** | **5.910 (−1.159)** | **~1%** |
| Exp 14 soft flips | **5.5540 (−1.515)** | 2.37% |

| Run (fineweb) | Slice CE (retention) Δ | Domain CE (adaptation) Δ | Bits flipped |
|---|---|---|---|
| Frozen control (Exp 6) | 7.989 (+0.92) | 7.838 (−1.08) | 0% |
| **Energy k=3 (Exp 6)** | **7.362 (+0.29)** | **6.915 (−2.01)** | **~1%** |
| Exp 14 soft flips | 7.411 (+0.34) | 7.149 (−1.77) | 2.17% |

Readings (honest):

- **The tinydata "win" is a flip-budget artifact, not a smoothness win.** Soft
  flips consumed ~16× the k=3 budget (2.37% vs 0.15% of bits, same measurement
  method). On the *same* domain more flips = more fitting = lower CE — any
  mechanism with a bigger budget wins there, and the trajectory confirms it
  (soft run crosses k=3's final CE at ~step 600, then keeps fitting).
- **On a genuine domain shift the same extra budget is harmful on BOTH axes.**
  fineweb: retention 7.411 vs 7.362 (+0.05 worse), adaptation 7.149 vs 6.915
  (+0.23 worse). More flips on a new domain = more overwriting of old-domain
  knowledge, and the noisy domain-gradient band does not improve adaptation
  either.
- **Budget is the causal variable; the shape of the decision is not.** Hard
  k=2 (11.6% flips, Exp 5) was unstable and worse; soft 2.4% is stable and
  better than k=3 on tinydata — but both are just different points on the same
  budget-vs-domain curve. At matched budget (s→0.02 ≈ deterministic) soft flips
  ARE k=3, and the mechanism contributes nothing beyond that. The Exp 5/6 sweet
  spot (~1% budget, hard rule) stands.
- **Mechanics verified**: dead zone works (zero-acc entries never flip), the
  band converges exactly to the deterministic rule as s→0, and the σ band never
  produced the catastrophic mass-churn of sign-vote (Exp 2 M1) — the failure
  mode is mild drift, not transient explosion.

**Verdict: rejected — smoothing the flip decision adds no transferable gain;
the flip budget, set by k, is what the SBF mechanism actually trades in.** This
reinforces the chain conclusion: with the k=3 rule the SBF learning dynamics are
at their frontier within this training budget. `--soft-flip-temp` remains in
`train_baseline_backprop.py` (documented failed experiment).

Reproduce:

```bash
# tinydata (same domain)
python train_baseline_backprop.py --resume checkpoints_bp/checkpoint_000200.pt \
    --steps 1000 --data-cache tinydata --lr-domain 1e-4 --batch-size 8 \
    --acc-energy --acc-decay 0.99 --adaptive-thr 3.0 --soft-flip-temp 0.25 \
    --save-dir checkpoints_exp14_sft
# fineweb (domain shift)
python train_baseline_backprop.py --resume checkpoints_bp/checkpoint_000200.pt \
    --steps 1000 --data-cache data/fineweb_10bt \
    --eval-slice examples/discrete/sliceEval100k.bin \
    --domain-eval examples/continual/fineweb_eval100k.bin \
    --lr-domain 1e-4 --batch-size 8 \
    --acc-energy --acc-decay 0.99 --adaptive-thr 3.0 --soft-flip-temp 0.25 \
    --save-dir checkpoints_exp14_sft_fw
```
---

## Experiment: MLA / Hybrid Attn (Exp 15) — status note

Status: **paused / shelved**. Core ternary (dense) pipeline is healthy; MLA is an
experimental branch and is not the focus.

Findings (2026-08-11):

- **exp15_mla.bin (checkpoints/exp15_mla.bin) is degenerate**: trained with
  --steps 300 --warmup-steps 200, per_channel=False, no alphas on any of the
  54 ternary tensors. Even a correct torch/numpy forward gives CE ~4.7e8
  (activations explode to ~1e7, logits ~1e8). Weights in the binary were
  verified 68/68 tensors exact (verify_export/verify_mla).
- **MLA trains fine with a sane config**: --per-channel (alphas_init=0.02):
  eval_CE_100k ~4.9 at step 300, stable through ~step 900; eval diverges to
  100+ from ~step 1500 onward (train loss stays ~5.75) - open question about
  the torch eval path at longer context, not investigated further.
- **C++ MLA decode path shows undefined behavior on the degenerate model**
  (same address, same run: vup_d[10240]=0 vs floats[10240]=-1; values vary
  between runs). Never reproduced on a healthy MLA model - no healthy MLA
  checkpoint exists yet (the sane-config 6000-step run was killed by timeout
  before saving; default save dir did not receive checkpoints).

Next time an MLA checkpoint is trained to a sane loss:
1. export it with inference/export_model.py, then run
   selflearn_avx2.exe --eval <model>.bin examples/discrete/sliceEval100k.bin 3.
2. Finite CE -> C++ MLA path is fine (NaN was garbage-in). NaN -> investigate
   the MLA expand-loop / weights path in tetra.h (expand loop reads
   model.tw(kun/vun).floats.data(); suspect an OOB write or stale pointer
   near the k_full/v_full cache, or heap corruption from the degenerate
   magnitudes).

Relevant code: MLA decode branch in inference/tetra.h (guarded by
model.is_mla), 	ernary_llm/mla.py (torch side).

## Experiment: ARS-Gated Embedding Updates (Exp 16)

Question: flip proposals are gated chunk-by-chunk through a block probe
(`--ars`, README section above), but the embedding SGD step has no such gate —
any LR update that pushes probe CE up is committed anyway, and on-device
embedding gradients are as noisy as flip proposals. Can the same
margin-gated retrial clean up the embedding update (propose → probe → revert
if worse), and does it cost measurable accuracy / wall-clock?

Setup: `selflearn_avx2.exe` (AVX2+FMA, 2026-08-11 build). Shared probe
machinery refactored out of the flip loop into `ars_probe_block()`
(`--ars-probe` tokens split into `--ars-windows` tail-first windows, margin
`max(--ars-margin, --ars-margin-rel × L_cur)` — same as flips). New flags:
`--ars-emb` gates the top `--ars-emb-max` embedding rows by |grad| (default
4/block): apply the full LR step, probe CE, keep if within the margin, else
revert the row. Decoupled WD still applies to every row (including reverted
ones). `--sl-lr-embedding F` overrides the exported embedding LR (>0; 0 keeps
metadata). Non-ARS arithmetic is bit-identical to the previous combined
per-row update (reordered per row, same op sequence: `er -= lr·grad·s`, then
`er *= (1 − lr·wd)`).

Results (smoke, `exp_tog_s0_zacc.bin` + `slice100k.bin`, 10 blocks,
lrEmb 1e-4, probe 64/2 windows):

| Run | blocks | wall/block | ARS-emb tried | accepted |
|---|---|---|---|---|
| no ARS-emb (parity) | 3 | 629 ms | — | — |
| `--ars --ars-emb` | 10 | 1192 ms | 4/block | 0–4 (≈50% overall) |

- Gate is exercising: 0/4 rejected at step 7 when probeCE jumped 6.95 → 7.26
  on the first trial; probeCE wobbled 6.26–7.26 across blocks (no diverging
  drift in a 10-block window).
- Cost of embedding gating: +5 probes per block (1 baseline + up to 4
  trials) ≈ +47% wall vs the flip-only ARS run at probe 64/2.
- Parity check: same seed/params with `--ars` and `--ars-emb` removed
  reproduces the pre-change block CEs (6.8898 / 6.7274 / 6.8387) and saves
  a distinct file when gating is on (updates applied / skipped correctly).

**Verdict: gate mechanics validated, adaptive benefit not yet
measured.** The gate strictly rejects harmful rows and preserves the exact
non-ARS trajectory, but a 10-block smoke cannot show whether gated
embeddings train *better* than ungated. Next step: 200-block A/B
(`--ars-emb` on/off) on finewebEval100k, same protocol as the flip-gate A/B
in README, to see if gate-confirmed rows wash the embedding-noise floor.

**A/B (200 blocks, fineweb, 2026-08-11): NULL RESULT — gating is free at
working LRs, so `--ars-emb` stays OFF by default.** Three arms on
`exp7_v6_lr5.bin` (natural scale, metadata `energy/adaptiveThr=3/sparsity=0.01`
active), `data/fineweb_10bt/fineweb_0000.bin`, thr=1.0 decay=0.99
flipEvery=4 toggle=1, 200 blocks:

| Arm | probes/block | wall/block | slice @10K | fineweb @10K |
|---|---|---|---|---|
| baseline eval | — | — | 8.2026 | 9.3451 |
| `--ars` (flip gate) | 0 + flip trials | 2.68 s | 8.2107 | 9.2599 |
| `--ars --ars-emb` | +5 | 4.55 s (+70%) | 8.2107 | 9.2599 |
| `--ars --ars-emb --ars-block` | +6 | 5.15 s (+92%) | 8.2107 | 9.2599 |

All three trained models are **byte-identical (same SHA256)**: with
`lr_emb=1e-5` (the natural-scale-correct LR baked into the metadata) every
proposal is accepted — the ARS-emb gate rejects 0/4 rows on every block,
the block gate sees pre==post CE and never fires — while the probe traffic
costs +70–92% wall. The gate only exercises at higher LRs (the compressed-
scale smoke at `lr_emb=1e-4` rejected 0–3 rows/block); it is a tool for
that regime, not a default. The runs themselves did show the continual-
learning signal (fineweb eval −0.085 nats, slice +0.008).

**Bug fixed along the way:** `ars_probe_block` now restores `cache.pos` to
the block end. Before, each call left pos at the *last probe window's* end,
so consecutive probes (ARS-emb baselines/trials, the block gate) drifted to
progressively smaller windows (128→96→32 tokens) and under-measured the
block; flip-ARS was unaffected (it probes from a fixed captured `blk`).

**Open issue (pre-existing, not from this work):** `--eval` on
`fineweb_eval100k.bin @10000` positions crashes nondeterministically with
`0xC0000409` (fast-fail) ~1 in 3 runs (hit on arm_a once, arm_b twice;
retries succeed, @9000 never crashes, slice @10000 never crashes). The eval
code path is untouched by the ARS work; suspect a latent heap/race issue in
long evals. Needs its own investigation (reproduce under
`/MTd`/sanitizer build).

Reproduce:

```bash
# build
inference\build.bat avx2
# gated embedding smoke (10 blocks, probe 64/2, 4 rows/block trialled)
inference\selflearn_avx2.exe checkpoints_discrete_c3\exp_tog_s0_zacc.bin examples\discrete\slice100k.bin tmp_arsemb.bin 10 1 10 20 0.99 5 0 --ars --ars-emb --ars-probe 64 --ars-windows 2 --ars-emb-max 4
# whole-block gate (--ars-block): pre/post probe each step, full rollback on breach
inference\selflearn_avx2.exe checkpoints_discrete_c3\exp_tog_s0_zacc.bin examples\discrete\slice100k.bin tmp_blk.bin 6 1 6 20 0.99 5 0 --ars --ars-emb --ars-block --ars-probe 64 --ars-windows 2 --ars-emb-max 4
# parity: identical block CEs to the pre-Exp-16 build, no ARS-emb lines
inference\selflearn_avx2.exe checkpoints_discrete_c3\exp_tog_s0_zacc.bin examples\discrete\slice100k.bin tmp_noars.bin 3 1 3 20 0.99 5 0
```

---


## Experiment: KV-Cache Quantization (Exp 17) — FP32 baseline 2026-08-11

Goal: cut the FP32 KV cache 4x. Baseline measured BEFORE any change (selflearn_prof.exe, /DTETRA_PROFILE AVX2, exp7_v6_lr5.bin tiny 6L/256H/8hd/32, sliceEval100k.bin, FP32 lm_head):

| run | positions | tokens/s | ms/token |
|---|---|---|---|
| single window (ramp 1->2048, no roll) | 2048 | 235.6-239.0 | ~4.2 |
| steady decode (10 rolls at 2048) | 20480 | 244.4 | 4.09 |

Stage breakdown (82.2 s / 20480 positions): attn_scores 43.2% | gate_up 27.2% | down_proj 9.8% | qkv_matmul 9.1% | lm_head 7.2% | o_proj 3.2% | norm 0.4%.

KV cache footprint (FP32): 6 x 2 x 2048 x 256 x 4 B = 25.2 MB; decode reads ~24 MB of K/V per token (4 MB/layer), i.e. ~6 GB/s of the ~4 ms token budget is KV traffic alone — attn_scores (43%) is the KV-bound stage to attack. Reference at 500m preset: 251 MB FP32 (6L x 2560).


## Exp 17, Round 2: int8/int16 KV cache implemented + SIMD (2026-08-11)

- K: int16 per-row scale (max/32767) + per-head int16 query once per token -> score loop is a pure integer SIMD madd (AVX2 _mm256_madd_epi16). V: int8 with 8-lane cvtepi8_epi32 accumulate.
- --kv-int8 flag (selflearn eval/train, tetra.cpp arg 9). Drift vs FP32 (tetra_model.bin, 6L/256/8H, decode 4000 pos): CE 8.5275 -> 8.5278; prefill logits mean 2.9e-4 (max 6.6e-4), MLA 2.4e-4 (max 1.2e-3).
- Speed (selflearn_avx2.exe, --eval 4000): scalar int8 v1 333.6 tok/s (-14% vs FP32 364.7; earlier FP32 389.7 at session start). int16-SIMD v2: 384.9 / 387.4 tok/s -> +6% vs same-session FP32; RAM for K+V: 4 B -> ~2.5 B/elem + per-row scale.
- NOTE: intermittent 0xC0000409 / 0xC0000005 crashes on 4000-8000 pos eval this session reproduce on the PRE-CHANGE (7b6b371) binary too; unaffected kv-int8 runs 6/6 clean. Not caused by Exp 17 changes; left for follow-up.


## Exp 17, Round 3: int16 score overflow on v8 models — FIXED (2026-08-12)

- **Found via baseline upgrade**: exported the best STE checkpoint (exp10_v8f, Exp 10 v8-forward) to a new canonical baseline `inference/exp10_v8.bin` (v8, verified 51/51 exact vs torch). C++ FP32 eval: CE 5.7404 (vs 8.53 for the old Phase-1 v4 `tetra_model.bin`).
- **Bug**: `--kv-int8` on the v8 baseline drifted **+1.037 nats** (5.7404 -> 6.7776, PPL 311 -> 878) and was 11% SLOWER. Root cause: `kvq16_row` scaled to max/32767, so K/Q int16 values reach full magnitude; `_mm256_madd_epi16` sums int16 PAIRS into int32 and a single pair at 2*32767^2 = 2.147e9 already overflows int32 (before the 8-lane horizontal add). v4 (Phase-1) activations are tightly bounded so products stayed ~1e6; v8 true-value outliers make fat-tailed activations hit the max regularly -> overflow -> garbage scores.
- **Fix** (`tetra.h` `kvq16_row`): scale = max/**4096** (12-bit range) — provably overflow-free: worst pair-sum 2*4096^2 = 3.36e7, AVX2 8-lane horizontal <= 1.07e9, AVX-512 16-lane <= 5.4e8, all < 2^31. Score math is scale-invariant (dot*scale_q*scale_k), precision 12 bits > fp16 mantissa.
- **Results** (same session, selflearn_avx2.exe, sliceEval100k @40K):
  - v8: FP32 5.7404 | kv-int8 5.7405 -> drift **+0.0001** (was +1.037)
  - v4: FP32 8.5309 | kv-int8 8.5309 -> drift **0.0000**
  - avx512 v8 kv-int8: 5.7405 @ 292.5 tok/s (avx2 268.5; avx2 FP32 279.7) — on v8 the eval loop is prefill-dominated (v8 outlier blobs), the decode-only win from Round 2 (v4: +6%) still applies to generation.
- All exes rebuilt (scalar/avx2/avx10/avx512, tetra + selflearn). `exp10_v8.bin` is the new canonical baseline (CE 5.74, full v8 true-value outliers), `tetra_model.bin` untouched.


## Exp 18: UTF churn-ramp pipeline — CLOSED, negative result (2026-08-12)

Goal (Tier-1 item 4): raise the per-block churn safely above the fixed-k rate — the ARS gate (Exp 16/17) caps real changes at 1024/pass (16 trials x 64 chunk) even with 100% chunk acceptance. Design: `--churn-ramp RATE` lowers the adaptive-thr multiplier k_eff each flip pass and scales the probe capacity with the proposal budget (k0/k_eff); a safety reflex walks k back + halves the rate on chunk-acceptance violation or --ars-block revert.

Setup: new self-learning baselines `checkpoints/exp10_v7_sl.bin` / `exp10_v8_sl.bin` — v7/v8 exports of the best STE checkpoint with `--sl-energy --sl-adaptive-thr 3.0 --sl-sparsity 0.01 --sl-lr-embedding 1e-5`. Untrained v8sl: fineweb_eval100k @10K CE 12.7169, sliceEval100k @10K CE 5.7848.

- **Export fix (prerequisite)**: `export_self_learning` on STE checkpoints was broken — `StochasticTransformerModel.load_state_dict(strict=False)` left `packed_weights` at random init (STE checkpoints store only `latent_weights`) → first export gave CE 27.57 garbage. Fixed in `inference/export_model.py`: STE + `--self-learning` now routes through `TernaryTransformerModel` + `export_model()` (requires `--v7` or `--v8`; v6 has no outlier channel). Parity verified exact vs Exp 10: v7 CE 5.8556, v8 CE 5.7404 @40K sliceEval.
- **Bug found via reload crash**: saved self-learning binaries crashed 0xC0000005 on load at ALL eval lengths (deterministic; file 50,044 B smaller than source). ASAN build (`cl /fsanitize=address`, vcvarsall from VS18 Insiders) → heap-buffer-overflow READ of size 1 = `dequantize_row` reading past the outlier blob. Root cause: outlier boundary INCONSISTENCY — `ars_repack` used `|v| >= 1.5` while apply_bit_flips' repack and save_model's blob recount used strict `> 1.5`. The exporter rounds any genuine |x_n| > 1.5 to level byte 48 = exactly 1.5 (1/32Δ resolution), so the two rules diverged: packed code-11 count 61,690 vs blob recount 59,763 in the trained file → OOB.
- **Fix** (now uniform INCLUSIVE `|v| >= 1.5` in ars_repack, apply_bit_flips repack, v8/v7 blob recounts, and `is_std` boundary `< 1.5`): bit-invariant with the Python exporter (byte 48 only ever written for genuine outliers). Strict `> 1.5` additionally silently demoted ~0.7% of true-outlier weights (|x_n| in (47/32, 48/32]) to ±1 on the first full-matrix repack — roughly +0.2 nats of hidden regression.
- **Honest re-runs** (40 blocks, fineweb_0000.bin, fixed arm k=3.0 vs ramp arm `--churn-ramp 0.25`):
  - Churn: both arms ever ≈ 0.07–0.09% (~4,400–5,600 distinct weights of 6.29M); most flip passes deliver **0 accepted changes** (the ARS chunk probes reject ~100% of the 70–80K proposals); the rare passing pass hits exactly the 1024 chunk cap.
  - Ramp: k_eff never descended below ≈ 2.98. Pre/post probe noise (±0.06–0.07 CE) exceeds the 0.5% block margin on 30–50% of flip passes; each spurious revert punches k back and halves ramp_rate (0.25 → 0.001 within 40 blocks). Capacity scaling never engaged.
  - Eval @10K vs untrained baseline: fixed +0.150 fineweb / +0.137 slice; ramp +0.207 / +0.187. **Both arms net-regress despite ARS gating.**
- **Conclusion**: negative. The churn frontier at the Exp 10 tile is QUALITY-bound, not capacity-bound — lowering k only adds junk proposals (ARS rightly rejects >99%), and the ~1% of flips that pass still regress long-horizon CE; the block gate's probe noise makes the reflex fire at random and crushes the ramp. UTF machinery retained (flags `--churn-ramp --churn-min-k --churn-min-accept --churn-max-trials --churn-max-chunk --churn-warmup`) but not recommended for this regime; revisit only with a higher-quality acceptance signal (multi-window probes, held-out gate) and after the layer-wise churn question is separated from embedding updates.


## Exp 19: MatMul-Free Learning Rule (`--no-mul`) — Variant 2 beats the float baseline (2026-08-17)

Question: does the on-device gradient-free learning rule still learn if we remove the float multiply entirely from the weight and embedding updates? The forward pass is already matmul-free (ternary weights → select-add), but the rule-'c' gradient still does `grad[o][i] += e · x[i]` with a full-precision float product. If the last multiply can go, the whole train→inference pipeline (STE base training aside) becomes multiply-free.

Setup: `checkpoints/exp7_v6_lr5_fp32emb.bin` (v6, fp32 embedding, self-learning config energy=1 adaptiveThr=3.0 sparsity=0.01), learn on `examples/discrete/slice100k.bin` (200 blocks), eval on `examples/discrete/sliceEval100k.bin` (2000 positions). Baseline eval before learning: **CE 8.0847**. Three arms, same config, only the learning rule differs:

| Arm | rule 'c' gradient | embedding gradient | eval CE @2000 | Δ vs 8.085 |
|---|---|---|---|---|
| Baseline (float) | `e · x[i]` (full float product) | `gv · h[i]` | **7.7027** | −0.382 |
| **Variant 2 `--no-mul`** | `e · ternary(x[i])` (absmean thr) | `gv · ternary(h[i])` | **7.5140** | **−0.571** |
| Variant 1 (sign-sign) | `sign(e) · sign(x[i])` (±1 only) | `sign(gv) · sign(h[i])` | 9.5574 | +1.472 |

- **Variant 2** (accepted): keep the error `e` / `gv` in full precision; quantize the **activation** to ternary {-1,0,+1} with an **absmean threshold** — `alpha = mean(|x|)`, `ternary(x) = sign(x)` if `|x| > alpha/2` else 0, the same absmean convention the STE base training uses. The update `e · ternary(x)` is select/add — no real multiply. **Learns and beats the float baseline by −0.19 nats on the same budget.** Block CE also tracks the baseline (6.85 vs 6.89 @200 steps), churn/flip counts comparable (~34.7K vs ~68K flips/pass), at ~+21% block time (absmean scan is a second pass over the activation).
- **Variant 1** (rejected): collapsing **both** factors to sign (±1) discards the error magnitude entirely — the accumulator only ever receives ±1, flips drop to ~12K/pass and the model regresses +1.47 nats. Magnitude on the error side is load-bearing; on the activation side it is not.
- **Implementation**: `--no-mul` flag in `selflearn.cpp`, applied in rule-'c' (`sl_feed_predictive` feed loop) and the embedding-gradient path. Threshold computed once per layer / per position (not per row/v).
- **Verdict**: POSITIVE — the matmul-free learning rule learns on-device and slightly outperforms the float-multiply baseline. The removal is on the **activation** factor; error magnitude stays float (the accumulator remains FP32 by design). Follow-ups: sweep the ternary threshold (static |x|>0.5 vs absmean scaling), longer horizons (500–1000 blocks), and whether the same rule helps the STE-trained base path.
