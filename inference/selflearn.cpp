// Tetra - self-learning runtime (pure local rules, gradient-free)
//
// Usage: selflearn.exe <model.bin> <tokens.bin> <out.bin> [steps] [log_every] [save_every]
//   model.bin   : v6 binary exported with --self-learning (has accumulators + sl_* config)
//   tokens.bin  : raw uint16 LE token stream (same format as tinydata/tinystories.bin)
//   out.bin     : path to write the updated model (atomic .tmp + rename)
//   steps       : number of blocks to train (default: 200)
//   log_every   : print block loss every N steps (default: 50, 0 = off)
//   save_every  : write out.bin every N steps (default: 100, 0 = only at end)
//
// Implements rule 'c' (temporal predictive coding) matching the PyTorch
// DiscreteTrainer: per-layer delta = -sign((y_t - y_{t-1})^T x_{t-1}) fed into a
// leaky accumulator (acc *= decay), bit flips applied when |acc| > threshold, and
// the tied embedding updated with plain local SGD (per-row clip + decoupled WD).

#include "tetra.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>

using namespace tetra;

static std::vector<uint16_t> read_tokens(const char* path, size_t max_bytes = 0) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return {}; }
    if (max_bytes > 0 && (size_t)sz > max_bytes) sz = (long)max_bytes;
    std::vector<uint16_t> toks(sz / 2);
    size_t got = fread(toks.data(), 2, toks.size(), f);
    fclose(f);
    toks.resize(got);
    return toks;
}

// Quantize an FP32 weight in-place to INT8 (per-tensor scale = max|w|/127,
// matching export_model.py quantize_fp32_to_int8) so forward() takes the
// matmul_int8_decode LM-head path (4x less memory bandwidth).
static void quantize_fp32_to_int8(FP32Weight& w) {
    float mx = 0.0f;
    for (float v : w.data) mx = (std::max)(mx, std::fabs(v));
    float scale = (mx < 1e-10f) ? 1.0f : mx / 127.0f;
    w.int8_scale = scale;
    w.int8_data.resize(w.data.size());
    for (size_t i = 0; i < w.data.size(); i++) {
        float q = std::round(w.data[i] / scale);
        q = (std::max)(-128.0f, (std::min)(127.0f, q));
        w.int8_data[i] = (int8_t)q;
    }
    fprintf(stderr, "quantized %s -> int8 (scale=%.6f, %zu elems)\n",
            "token_embedding.weight", scale, w.data.size());
}

// Accept-Reject Search (--ars): after apply_bit_flips proposes flips, trial
// them chunk-by-chunk on probe windows replayed from the live cache and keep
// only the chunks whose probe CE does not exceed the current best by more
// than the margin. Rejected weights consume their (already zeroed)
// accumulator suggestion and revert to the pre-flip value.
//
// Statistical power: the probe is split into --ars-windows windows spread
// evenly across the block (default 128 tokens in 4x32), and the accept
// margin is max(--ars-margin, --ars-margin-rel * L_cur) so the gate widens
// proportionally at high-CE (noisy) regimes instead of using a fixed
// absolute nats threshold.
static double ars_probe_ce(Model& model, const std::vector<uint16_t>& tokens,
                           size_t w_start, size_t w_len, float scale,
                           std::vector<float>& softmax_buf, KVCache& cache,
                           int V) {
    // Replay tokens [w_start, w_start+w_len) at cache positions
    // [cache.pos - w_len, cache.pos). The caller sets cache.pos to the end of
    // the replay; the rewinds land on the same positions and K/V is recomputed
    // under the current weights each trial. Callee guarantees the window
    // stays inside the linear cache region.
    const size_t w0 = (size_t)cache.pos - w_len;
    cache.pos = (int)w0;
    double loss = 0.0;
    size_t cnt = 0;
    for (size_t t = w_start; t < w_start + w_len; t++) {
        std::vector<int> single = {tokens[t]};
        std::vector<float> logits = forward(model, single, cache, nullptr);
        float mx = -1e30f;
        for (int i = 0; i < V; i++) { float s = logits[i] * scale; if (s > mx) mx = s; }
        double sum = 0.0;
        for (int i = 0; i < V; i++) {
            softmax_buf[i] = (float)std::exp((double)logits[i] * scale - mx);
            sum += softmax_buf[i];
        }
        int tgt = tokens[t + 1];
        loss += -std::log((double)(softmax_buf[tgt] / (float)(sum + 1e-12)) + 1e-12);
        cnt++;
    }
    return cnt > 0 ? loss / (double)cnt : 0.0;
}

// Probe CE over the block: --ars-probe tokens split into --ars-windows
// tail-first windows replayed in place. Shared by flip and embedding trials.
static double ars_probe_block(Model& model, const std::vector<uint16_t>& tokens,
                              size_t start, size_t blk, int ars_probe_total,
                              int ars_windows, float scale,
                              std::vector<float>& softmax_buf, KVCache& cache,
                              int V) {
    const size_t wl = (std::max)((size_t)1,
                        (size_t)ars_probe_total / (size_t)((std::max)(1, ars_windows)));
    const size_t K = (size_t)((std::max)(1, ars_windows));
    double tot = 0.0;
    size_t wsum = 0;
    for (size_t i = 0; i < K; i++) {
        size_t off = blk - (i + 1) * wl;
        if ((long long)off < 0 || off + wl > blk) break;
        cache.pos = (int)(off + wl);
        double ce = ars_probe_ce(model, tokens, start + off, wl, scale,
                                 softmax_buf, cache, V);
        tot += ce * (double)wl;
        wsum += wl;
    }
    // Replay is position-neutral per window, but the loop leaves pos at the
    // last window's end; restore the block end so consecutive probes (and
    // anything reading cache.pos after) always see the same state.
    cache.pos = (int)blk;
    return wsum > 0 ? tot / (double)wsum : 0.0;
}

// Accept margin: max of the absolute floor and a fraction of the best CE,
// so the gate widens proportionally in the high-CE (noisy) regime.
static float ars_margin_eff(float ars_margin, float ars_margin_rel, double L_cur) {
    return (std::max)(ars_margin, ars_margin_rel * (float)L_cur);
}

// Rebuild the 2-bit packed codes from the staged float weights after
// Accept-Reject Search rollbacks (save_model writes w.packed verbatim,
// so it must match floats).
static void ars_repack(Model& model) {
    for (auto& kv : model.ternary_weights) {
        TernaryWeightXNOR& w = kv.second;
        const int row_bytes = (w.cols + 3) / 4;
        std::fill(w.packed.begin(), w.packed.end(), 0u);
        for (int r = 0; r < w.rows; r++) {
            const float* prow = w.floats.data() + (size_t)r * w.cols;
            uint8_t* packed_row = w.packed.data() + (size_t)r * row_bytes;
            for (int c = 0; c < w.cols; c++) {
                float v = prow[c];
                int enc;
                // NOTE: INCLUSIVE |v| >= 1.5 boundary — MUST match the Python
                // exporter semantics AND apply_bit_flips' repack AND
                // save_model's blob recount. Values whose |x_n| > 1.5 get
                // rounded to level 48/32 = 1.5 exactly by the exporter, so a
                // strict > would demote them to ±1 on every full-matrix repack
                // (~0.7% of weights, ~+0.2 nats of silent regression). The
                // inclusive rule is bit-invariant with the Python blob: a
                // float lands AT +/-1.5 only via byte 48, which Python only
                // writes for genuine outliers.
                if (v >= 1.5f || v <= -1.5f) enc = 3;   // ±2 outlier (code 11)
                else if (v > 0.5f) enc = 2;             // +1
                else if (v > -0.5f) enc = 1;            // 0
                else enc = 0;                           // -1
                packed_row[c >> 2] |= (uint8_t)(enc << (6 - (c & 3) * 2));
            }
        }
    }
}

int main(int argc, char** argv) {
    // Eval-only mode: --eval <model.bin> <tokens.bin> [max_positions] [--fast-lmhead]
    // Reports average next-token cross-entropy without any learning.
    // --fast-lmhead: quantize the (tied) embedding to INT8 in memory so the
    // LM head uses matmul_int8_decode (4x less memory bandwidth). Use with
    // --compare-lmhead to measure the Top-1/Top-5 prediction divergence.
    if (argc >= 4 && strcmp(argv[1], "--eval") == 0) {
        bool fast_lmhead = false;
        bool kv_int8 = false;
        float scale_override = 0.0f;
        int pos_arg = 4;
        for (int i = 4; i < argc; i++) {
            if (strcmp(argv[i], "--fast-lmhead") == 0) fast_lmhead = true;
            else if (strcmp(argv[i], "--kv-int8") == 0) kv_int8 = true;
            else if (strcmp(argv[i], "--sl-logit-scale") == 0 && i + 1 < argc) {
                scale_override = (float)atof(argv[i + 1]);
                i++;
            }
            else if (pos_arg == 4) pos_arg = i;
        }
        Model model = load_model(argv[2]);
        if (fast_lmhead) {
            quantize_fp32_to_int8(model.fp32_weights.at("token_embedding.weight"));
        }
        std::vector<uint16_t> tokens = read_tokens(argv[3]);
        size_t limit = tokens.size() > 1 ? tokens.size() - 1 : 0;
        if (argc > pos_arg) {
            long n = atol(argv[pos_arg]);
            if (n > 0 && (size_t)n < limit) limit = (size_t)n;
        }
        fprintf(stderr, "Eval: %zu tokens (%zu positions) lmhead=%s\n",
                tokens.size(), limit, fast_lmhead ? "INT8" : "FP32");
        if (limit == 0) { fprintf(stderr, "No tokens\n"); return 1; }

        const int H = model.header.hidden_dim;
        const int V = model.header.vocab_size;
        const float scale = scale_override > 0.0f ? scale_override : model.sl_logit_scale;
        KVCache cache;
        cache.init(model.header.num_layers, model.header.max_seq_len, H,
                   model.header.num_heads, model.is_mla, model.kv_latent_dim,
                   model.rope_dim, kv_int8);

        std::vector<float> softmax_buf(V);
        double loss = 0.0;
        size_t pos = 0;
        auto t_start = std::chrono::steady_clock::now();
        for (size_t t = 0; t < limit; t++) {
            if (t > 0 && t % (size_t)model.header.max_seq_len == 0) cache.clear();
            std::vector<int> single = {tokens[t]};
            std::vector<float> logits = forward(model, single, cache, nullptr);
            float mx = -1e30f;
            for (int i = 0; i < V; i++) { float s = logits[i] * scale; if (s > mx) mx = s; }
            double sum = 0.0;
            for (int i = 0; i < V; i++) {
                softmax_buf[i] = (float)std::exp((double)logits[i] * scale - mx);
                sum += softmax_buf[i];
            }
            int target = tokens[t + 1];
            loss += -std::log((double)(softmax_buf[target] / (float)(sum + 1e-12)) + 1e-12);
            pos++;
            if (pos % 1000 == 0)
                fprintf(stderr, "  eval %zu | avg CE %.4f\n", pos, loss / pos);
        }
        auto t_end = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t_end - t_start).count();
        fprintf(stderr, "Eval done. avg CE %.4f | PPL %.4f | %zu positions | %.2f s | %.1f tokens/s\n",
                loss / pos, std::exp(loss / pos), pos, secs, pos / secs);
#ifdef TETRA_PROFILE
        tetra_profile_report();
#endif
        return 0;
    }

    // Compare LM head precision: --compare-lmhead <model.bin> <tokens.bin> [max_positions]
    // Runs FP32 and INT8 lm_head on the same context and reports Top-1/Top-5
    // agreement, CE, and per-position logit drift.
    if (argc >= 4 && strcmp(argv[1], "--compare-lmhead") == 0) {
        Model model = load_model(argv[2]);
        std::vector<uint16_t> tokens = read_tokens(argv[3]);
        size_t limit = tokens.size() > 1 ? tokens.size() - 1 : 0;
        if (argc > 4) {
            long n = atol(argv[4]);
            if (n > 0 && (size_t)n < limit) limit = (size_t)n;
        }
        if (limit > 1000) limit = 1000;
        fprintf(stderr, "Compare lm_head FP32 vs INT8: %zu positions\n", limit);
        if (limit == 0) { fprintf(stderr, "No tokens\n"); return 1; }

        bool kv_int8 = false;
        for (int i = 4; i < argc; i++) {
            if (strcmp(argv[i], "--kv-int8") == 0) kv_int8 = true;
        }

        const int H = model.header.hidden_dim;
        const int V = model.header.vocab_size;
        const float scale = model.sl_logit_scale;
        FP32Weight& emb = model.fp32_weights.at("token_embedding.weight");

        // Pass 1: FP32 logits
        std::vector<int> top1_fp32(limit), top1_int8(limit);
        std::vector<std::vector<int>> top5_fp32(limit), top5_int8(limit);
        std::vector<float> ce_fp32(limit), ce_int8(limit);
        // top-5: keep indices of the 5 highest logits in ascending order
        auto top5_of = [&](const std::vector<float>& logits) {
            std::vector<int> idx(V);
            for (int i = 0; i < V; i++) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                              [&](int a, int b) { return logits[a] > logits[b]; });
            idx.resize(5);
            return idx;
        };
        auto run_pass = [&](std::vector<int>* t1_out,
                            std::vector<std::vector<int>>* t5_out,
                            std::vector<float>* ce_out) {
            KVCache cache;
            cache.init(model.header.num_layers, model.header.max_seq_len, H,
                       model.header.num_heads, model.is_mla, model.kv_latent_dim,
                       model.rope_dim, kv_int8);
            for (size_t t = 0; t < limit; t++) {
                if (t > 0 && t % (size_t)model.header.max_seq_len == 0) cache.clear();
                std::vector<float> logits = forward(model, {tokens[t]}, cache, nullptr);
                (*t1_out)[t] = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
                (*t5_out)[t] = top5_of(logits);
                float mx = -1e30f;
                for (int i = 0; i < V; i++) mx = (std::max)(mx, logits[i] * scale);
                double sum = 0.0;
                for (int i = 0; i < V; i++) sum += std::exp((double)logits[i] * scale - mx);
                int target = tokens[t + 1];
                (*ce_out)[t] = (float)(-std::log((double)(std::exp((double)logits[target] * scale - mx) / sum) + 1e-12));
            }
        };
        run_pass(&top1_fp32, &top5_fp32, &ce_fp32);

        // Pass 2: INT8 logits (quantize in place)
        quantize_fp32_to_int8(emb);
        run_pass(&top1_int8, &top5_int8, &ce_int8);

        // Compare
        int same_top1 = 0, t1_in5_f = 0, t1_in5_i = 0;
        double drift_accum = 0.0;
        std::vector<double> ce_delta(limit), logit_delta(limit);
        for (size_t t = 0; t < limit; t++) {
            if (top1_fp32[t] == top1_int8[t]) same_top1++;
            if (std::find(top5_int8[t].begin(), top5_int8[t].end(), top1_fp32[t]) != top5_int8[t].end())
                t1_in5_i++;
            if (std::find(top5_fp32[t].begin(), top5_fp32[t].end(), top1_int8[t]) != top5_fp32[t].end())
                t1_in5_f++;
            ce_delta[t] = ce_int8[t] - ce_fp32[t];
            drift_accum += ce_delta[t];
        }
        double ce_f = 0, ce_i = 0;
        for (size_t t = 0; t < limit; t++) { ce_f += ce_fp32[t]; ce_i += ce_int8[t]; }
        auto max_abs = [](const std::vector<double>& v) {
            return *std::max_element(v.begin(), v.end(),
                [](double a, double b) { return std::fabs(a) < std::fabs(b); });
        };
        fprintf(stderr,
                "\n=== lm_head FP32 vs INT8 (%zu positions) ===\n"
                "  Top-1 identical:        %d/%zu (%.1f%%)\n"
                "  FP32 top-1 in INT8 top-5: %d/%zu (%.1f%%)\n"
                "  INT8 top-1 in FP32 top-5: %d/%zu (%.1f%%)\n"
                "  CE FP32: %.4f | CE INT8: %.4f | delta: %+.4f\n"
                "  CE drift: mean %+.5f | max %+.4f\n",
                limit, same_top1, limit, 100.0 * same_top1 / limit,
                t1_in5_i, limit, 100.0 * t1_in5_i / limit,
                t1_in5_f, limit, 100.0 * t1_in5_f / limit,
                ce_f / limit, ce_i / limit, (ce_i - ce_f) / limit,
                drift_accum / limit, max_abs(ce_delta));
        return 0;
    }

    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <model.bin> <tokens.bin> <out.bin> [steps] [log_every] [save_every] [thr] [decay] [flip_every] [toggle] [--toggle-window N] [--thr-anneal RATE] [--energy] [--adaptive-thr K]\n",
            argv[0]);
        fprintf(stderr,
            "  Named flags (preferred; positional form above still works). Value flags\n"
            "  default to the v6 metadata; 0 keeps metadata (-1 for --toggle).\n"
            "    --steps N  --log-every N  --save-every N\n"
            "    --thr F  --decay F  --flip-every N  --toggle [0|1|-1]  --no-toggle\n");
        fprintf(stderr,
            "  thr/decay/flip_every/toggle override the v6 metadata (0 = keep metadata value; -1 for toggle = keep metadata)\n"
            "  --sl-logit-scale F: override the metadata logit scale (>0; 0 = keep metadata)\n");
        fprintf(stderr,
            "  --toggle-window N: only toggle for the first N blocks, then no-op flips (annealing, finding #12)\n"
            "  --thr-anneal RATE: raise the flip threshold by RATE per pass (finding #12 refinement)\n"
            "  --energy: feed -grad magnitude into accumulators (Exp 3, needed with --adaptive-thr)\n"
            "  --adaptive-thr K: per-channel flip threshold tau = K * RMS(acc) (Exp 3, scale-invariant)\n"
            "  --sparsity S: top-k feed — keep only the top fraction S of per-row |grad| (Exp 3, heavy tail)\n"
            "  --ars (Accept-Reject Search): trial proposed flips chunk-by-chunk on probe\n"
            "         windows replayed from the live cache; keep chunks whose probe CE\n"
            "         stays within the margin of the running best (--ars-chunk,\n"
            "         --ars-trials, --ars-probe, --ars-windows, --ars-margin,\n"
            "         --ars-margin-rel; defaults 64 / 16 / 128 / 4 / 0.02 / 0.005)\n"
            "  --ars-emb: also gate embedding row updates through the block probe\n"
            "         (top --ars-emb-max rows by |grad| per block; default 4)\n"
            "  --sl-lr-embedding F: override the exported embedding LR (>0; 0 = keep)\n"
            "  --ars-block: whole-block gate — probe CE before/after each step's\n"
            "         updates and roll back flips+embedding if CE degrades past\n"
            "         the standard ARS margin (full undo, incl. weight decay)\n"
            "  --no-mul: matmul-free learning rule — quantize the activation to\n"
            "         ternary {-1,0,+1} via absmean threshold (alpha=mean(|x|),\n"
            "         keep |x|>alpha/2) and accumulate the full-precision error e\n"
            "         through it (e*ternary(x) = select/add, no real multiply).\n");
        return 1;
    }
    const char* model_path = argv[1];
    const char* data_path  = argv[2];
    const char* out_path   = argv[3];
    int steps      = (argc > 4) ? atoi(argv[4]) : 200;
    int log_every  = (argc > 5) ? atoi(argv[5]) : 50;
    int save_every = (argc > 6) ? atoi(argv[6]) : 100;
    float thr_override  = (argc > 7) ? (float)atof(argv[7]) : 0.0f;
    float decay_override = (argc > 8) ? (float)atof(argv[8]) : 0.0f;
    int   every_override = (argc > 9) ? atoi(argv[9]) : 0;
    int   toggle_override = (argc > 10) ? atoi(argv[10]) : -1;

    // Ablation: pass "--no-ternary" as the FIRST argument to keep only the
    // embedding local SGD (no ternary deltas, no bit flips).
    bool no_ternary = false;
    if (argc > 1 && strcmp(argv[1], "--no-ternary") == 0) {
        no_ternary = true;
        model_path = argv[2];
        data_path  = argv[3];
        out_path   = argv[4];
        steps      = (argc > 5) ? atoi(argv[5]) : 200;
        log_every  = (argc > 6) ? atoi(argv[6]) : 50;
        save_every = (argc > 7) ? atoi(argv[7]) : 100;
        thr_override  = (argc > 8) ? (float)atof(argv[8]) : 0.0f;
        decay_override = (argc > 9) ? (float)atof(argv[9]) : 0.0f;
        every_override = (argc > 10) ? atoi(argv[10]) : 0;
    }

    // Parity mode: full loop + bit flips, but the embedding is frozen. This
    // makes the run bit-reproducible from the exported file alone — the Python
    // mirror is inference/parity_check.py (finding #10, toggle parity).
    bool flip_only = false;
    if (argc > 1 && strcmp(argv[1], "--flip-only") == 0) {
        flip_only = true;
        model_path = argv[2];
        data_path  = argv[3];
        out_path   = argv[4];
        steps      = (argc > 5) ? atoi(argv[5]) : 200;
        log_every  = (argc > 6) ? atoi(argv[6]) : 50;
        save_every = (argc > 7) ? atoi(argv[7]) : 100;
        thr_override  = (argc > 8) ? (float)atof(argv[8]) : 0.0f;
        decay_override = (argc > 9) ? (float)atof(argv[9]) : 0.0f;
        every_override = (argc > 10) ? atoi(argv[10]) : 0;
        toggle_override = (argc > 11) ? atoi(argv[11]) : -1;
    }

    // Optional annealing: "--toggle-window N" (any argv position) limits the
    // toggle kick to the first N blocks, then falls back to no-op flips. This
    // captures the saturation-wall rescue (finding #12) without the sustained
    // churn that destroys structure past ~100 blocks.
    int toggle_window = 0;
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--toggle-window") == 0) {
            toggle_window = atoi(argv[i + 1]);
        }
    }
    // "--thr-anneal RATE" (finding #12 refinement): effective flip threshold
    // rises by RATE each flip pass (thr, thr+RATE, thr+2*RATE, ...). The goal
    // is to make the saturation-wall crossing non-periodic: after a few passes
    // the bar is too high for the residual accumulator drift to re-cross, so
    // the model freezes instead of re-kicking every ~25 blocks.
    float thr_anneal = 0.0f;
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--thr-anneal") == 0) {
            thr_anneal = (float)atof(argv[i + 1]);
        }
    }
    // Accept-Reject Search (--ars): the flip pass proposes candidates, then
    // re-trials them chunk-by-chunk on a probe window replayed from the live
    // cache; chunks whose probe CE beats the running best (within the margin)
    // are kept, the rest are rolled back.
    bool ars = false;
    int ars_chunk = 64;
    int ars_trials = 16;
    int ars_probe = 128;
    int ars_windows = 4;
    float ars_margin = 0.02f;
    float ars_margin_rel = 0.005f;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ars") == 0) ars = true;
    }
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--ars-chunk") == 0) ars_chunk = atoi(argv[i + 1]);
        if (strcmp(argv[i], "--ars-trials") == 0) ars_trials = atoi(argv[i + 1]);
        if (strcmp(argv[i], "--ars-probe") == 0) ars_probe = atoi(argv[i + 1]);
        if (strcmp(argv[i], "--ars-windows") == 0) ars_windows = atoi(argv[i + 1]);
        if (strcmp(argv[i], "--ars-margin") == 0) ars_margin = (float)atof(argv[i + 1]);
        if (strcmp(argv[i], "--ars-margin-rel") == 0) ars_margin_rel = (float)atof(argv[i + 1]);
    }
    if (ars_windows < 1) ars_windows = 1;
    if (ars_probe < ars_windows) ars_probe = ars_windows;
    // --ars-emb: gate embedding row updates through the block probe (top
    // --ars-emb-max rows per block). --sl-lr-embedding overrides the exported
    // embedding LR (>0; 0 keeps the metadata value).
    bool ars_emb = false;
    int ars_emb_max = 4;
    float lr_emb_override = 0.0f;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ars-emb") == 0) ars_emb = true;
    }
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--ars-emb-max") == 0) ars_emb_max = atoi(argv[i + 1]);
        if (strcmp(argv[i], "--sl-lr-embedding") == 0) lr_emb_override = (float)atof(argv[i + 1]);
    }
    if (ars_emb_max < 1) ars_emb_max = 1;
    // --ars-block gates the whole step: probe CE before/after the updates and
    // roll back every parameter change if it degrades past the ARS margin.
    bool ars_block = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ars-block") == 0) ars_block = true;
    }
    // Exp 3 flip mechanics: "--energy" switches the accumulator feed from
    // -sign(grad) votes to -grad magnitude; "--adaptive-thr K" sets the
    // per-channel flip threshold tau = K * RMS(acc); "--sparsity S" keeps only
    // the top fraction S of per-row gradient (heavy tail for the adaptive tau).
    // Negative values keep the exported metadata value.
    bool energy_override = false;
    float adaptive_override = -1.0f;
    float sparsity_override = -1.0f;
    float scale_override = 0.0f;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--energy") == 0) energy_override = true;
    }
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--adaptive-thr") == 0) {
            adaptive_override = (float)atof(argv[i + 1]);
        }
        if (strcmp(argv[i], "--sparsity") == 0) {
            sparsity_override = (float)atof(argv[i + 1]);
        }
        if (strcmp(argv[i], "--sl-logit-scale") == 0) {
            scale_override = (float)atof(argv[i + 1]);
        }
    }
    // --kv-int8: store the KV cache as int8 (symmetric per-row scale),
    // halving cache RAM at a small accuracy cost.
    bool kv_int8 = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--kv-int8") == 0) kv_int8 = true;
    }
    // --no-mul (Exp 19, Variant 2): matmul-free learning rule. Keep the error
    // in full precision but quantize the activation to ternary {-1,0,+1} with
    // an absmean threshold (alpha = mean(|x|), keep |x| > alpha/2); the update
    // e * ternary(x) is select/add — no real float multiply in the rule. This
    // tests whether the ternary model still learns when the last multiply in
    // the on-device updates is removed.
    bool no_mul = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-mul") == 0) no_mul = true;
    }
    // Churn ramp (Exp 18): --churn-ramp RATE lowers the adaptive-thr
    // multiplier k by RATE per flip pass and grows the ARS probe capacity
    // (trials x chunk) with the proposal budget, so sustained churn is not
    // capped by the gate. The safety reflex walks k back (and halves the
    // rate) when a flip pass violates --churn-min-accept chunk acceptance
    // or the --ars-block gate reverts the step. Requires --ars.
    float churn_ramp = 0.0f;
    float churn_min_k = 1.0f;
    float churn_min_accept = 0.5f;
    int churn_max_trials = 64;
    int churn_max_chunk = 1024;
    int churn_warmup = 0;
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--churn-ramp") == 0) churn_ramp = (float)atof(argv[i + 1]);
        if (strcmp(argv[i], "--churn-min-k") == 0) churn_min_k = (float)atof(argv[i + 1]);
        if (strcmp(argv[i], "--churn-min-accept") == 0) churn_min_accept = (float)atof(argv[i + 1]);
        if (strcmp(argv[i], "--churn-max-trials") == 0) churn_max_trials = atoi(argv[i + 1]);
        if (strcmp(argv[i], "--churn-max-chunk") == 0) churn_max_chunk = atoi(argv[i + 1]);
        if (strcmp(argv[i], "--churn-warmup") == 0) churn_warmup = atoi(argv[i + 1]);
    }
    if (churn_ramp > 0.0f && !ars) {
        fprintf(stderr, "ERROR: --churn-ramp requires --ars (no safety signal without probes)\n");
        return 1;
    }

    // Named flags for the positional override arguments (preferred over the
    // legacy positional form, which stays supported). All value flags accept
    // a value <= 0 (or -1 for --toggle) to keep the exported metadata value.
    for (int i = 1; i + 1 < argc; i++) {
        if      (strcmp(argv[i], "--steps")      == 0) steps          = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--log-every")  == 0) log_every      = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--save-every") == 0) save_every     = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--thr")        == 0) thr_override   = (float)atof(argv[i + 1]);
        else if (strcmp(argv[i], "--decay")      == 0) decay_override = (float)atof(argv[i + 1]);
        else if (strcmp(argv[i], "--flip-every") == 0) every_override = atoi(argv[i + 1]);
        else if (strcmp(argv[i], "--toggle")     == 0) {
            // "--toggle" alone enables kick; "--toggle 0"/"--toggle 1" explicit.
            const char* v = argv[i + 1];
            bool has_val = (v[0] == '-' || v[0] == '+' || (v[0] >= '0' && v[0] <= '9'))
                        && (v[0] != '\0');
            toggle_override = has_val ? atoi(v) : 1;
            if (has_val) i++;
        }
    }
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-toggle") == 0) toggle_override = 0;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    Model model = load_model(model_path);
    if (!model.sl_enabled) {
        fprintf(stderr, "WARNING: model is not a v6 self-learning export "
                        "(no sl_* config); using defaults.\n");
    }
    if (model.is_mla) {
        fprintf(stderr, "ERROR: self-learning is only supported for standard attention.\n");
        return 1;
    }
    if (model.sl_rule != 0) {
        fprintf(stderr, "ERROR: only rule 'c' (predictive coding) is implemented in C++ "
                        "so far (sl_rule=%d).\n", model.sl_rule);
        return 1;
    }

    std::vector<uint16_t> tokens = read_tokens(data_path);
    fprintf(stderr, "Tokens: %zu (uint16)\n", tokens.size());
    if (tokens.size() < 4) { fprintf(stderr, "Need at least 4 tokens\n"); return 1; }

    const int H = model.header.hidden_dim;
    const int V = model.header.vocab_size;
    const int block = model.sl_block_size;
    const float thr = thr_override > 0.0f ? thr_override : model.sl_threshold;
    const float decay = decay_override > 0.0f ? decay_override : model.sl_acc_decay;
    const int flip_every = every_override > 0 ? every_override : model.sl_flip_every_n;
    const bool toggle = toggle_override >= 0 ? (toggle_override != 0) : (model.sl_toggle != 0);
    const float scale = scale_override > 0.0f ? scale_override : model.sl_logit_scale;
    const float lr_emb = lr_emb_override > 0.0f ? lr_emb_override : model.sl_lr_embedding;
    const float wd_emb = model.sl_wd_embedding;
    const bool energy = energy_override ? true : (model.sl_energy != 0);
    const float adaptive_thr = adaptive_override >= 0.0f ? adaptive_override : model.sl_adaptive_thr;
    const float sparsity = sparsity_override >= 0.0f ? sparsity_override : model.sl_sparsity;

    // Per-block gradient buffers, keyed by layer name (filled from first capture).
    std::vector<std::string> names;
    std::vector<std::vector<float>> grads;

    // Churn tracking: stable list of ternary weights + per-weight flip counters.
    std::vector<TernaryWeightXNOR*> wlist;
    std::vector<std::vector<uint32_t>> hists;
    long long total_w = 0;
    for (auto& kv : model.ternary_weights) {
        wlist.push_back(&kv.second);
        hists.emplace_back(kv.second.floats.size(), 0u);
        total_w += (long long)kv.second.floats.size();
    }

    // Embedding gradient buffer (V x H) and its flattened view.
    std::vector<float> gradE((size_t)V * H, 0.0f);
    float* emb = model.fp32_weights.at("token_embedding.weight").data.data();

    std::vector<float> softmax_buf(V);
    auto t1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "Init in %.1f ms | block=%d thr=%.1f decay=%.3f flipEvery=%d toggle=%d%s "
                    "scale=%.6f lrEmb=%.1e wdEmb=%.2f%s%s energy=%d adaptiveThr=%.3f sparsity=%.3f\n",
            std::chrono::duration<double, std::milli>(t1 - t0).count(),
            block, thr, decay, flip_every, toggle ? 1 : 0,
            (toggle_window > 0 ? " (anneal off after block " + std::to_string(toggle_window) + ")" : "").c_str(),
            scale, lr_emb, wd_emb,
            (thr_anneal > 0.0f ? " | thr-anneal +" + std::to_string(thr_anneal) + "/pass" : "").c_str(),
            no_ternary ? " | embedding-only" : (flip_only ? " | flip-only" : (ars ? " | ARS" : "")),
            energy ? 1 : 0, adaptive_thr, sparsity);
    if (churn_ramp > 0.0f)
        fprintf(stderr, "  [churn ramp] rate=%.3f minK=%.2f minAccept=%.2f maxTrials=%d maxChunk=%d warmup=%d\n",
                churn_ramp, churn_min_k, churn_min_accept, churn_max_trials, churn_max_chunk, churn_warmup);

    double ms_total = 0.0;
    // Exp 18 churn-ramp state: effective adaptive-thr multiplier + probe
    // capacity, ramped per flip pass and walked back on gate violations.
    float k_eff = adaptive_thr;
    float ramp_rate = churn_ramp;
    int flip_passes = 0;
    for (int step = 0; step < steps; step++) {
        auto ts = std::chrono::high_resolution_clock::now();
        KVCache cache;
        cache.init(model.header.num_layers, model.header.max_seq_len, H,
                   model.header.num_heads, model.is_mla, model.kv_latent_dim,
                   model.rope_dim, kv_int8);

        // Reset per-block gradients.
        names.clear();
        grads.clear();
        std::fill(gradE.begin(), gradE.end(), 0.0f);

        size_t start = ((size_t)step * block) % (tokens.size() - 1);
        size_t end = (std::min)(tokens.size() - 1, start + block);

        double block_loss = 0.0;
        int valid_positions = 0;
        bool have_prev = false;
        Capture prev;

        for (size_t t = start; t < end; t++) {
            int tok = tokens[t];
            int target = tokens[t + 1];
            std::vector<int> single = {tok};

            Capture cap;
            std::vector<float> logits = forward(model, single, cache, &cap);

            // First position: just record names + activations.
            if (!have_prev) {
                for (const auto& lc : cap.layers) {
                    names.push_back(lc.name);
                    grads.emplace_back((size_t)lc.rows * lc.cols, 0.0f);
                }
                prev = std::move(cap);
                have_prev = true;
                continue;
            }

            // Rule 'c': accumulate grad[o][i] += (y_t - y_{t-1})[o] * x_{t-1}[i].
            for (size_t l = 0; l < cap.layers.size() && l < prev.layers.size(); l++) {
                const auto& cur = cap.layers[l];
                const auto& pr  = prev.layers[l];
                if (cur.rows != pr.rows || cur.cols != pr.cols) continue;
                std::vector<float>& g = grads[l];
                const size_t rows = (size_t)cur.rows, cols = (size_t)cur.cols;
                // --no-mul (Variant 2): quantize the activation x to ternary
                // {-1,0,+1} with an absmean threshold (alpha = mean(|x|),
                // ternary = sign(x) if |x| > alpha/2 else 0, the BitNet
                // convention this repo already uses in STE training), but keep
                // the error e in full precision. The update e * ternary(x) is a
                // vector-times-ternary product = select/add — no real matmul.
                float no_mul_alpha = 0.0f, no_mul_thr = 0.0f;
                if (no_mul) {
                    double sum_abs = 0.0;
                    const float* xa = pr.x.data();
                    for (size_t i = 0; i < cols; i++) sum_abs += std::fabs(xa[i]);
                    no_mul_alpha = (cols > 0) ? (float)(sum_abs / cols) : 0.0f;
                    no_mul_thr = 0.5f * no_mul_alpha;
                }
                for (size_t o = 0; o < rows; o++) {
                    float e = cur.y[o] - pr.y[o];
                    if (e == 0.0f) continue;
                    const float* xp = pr.x.data();
                    float* gr = g.data() + o * cols;
                    if (no_mul) {
                        for (size_t i = 0; i < cols; i++) {
                            float a = xp[i];
                            if (a > no_mul_thr)      gr[i] += e;
                            else if (a < -no_mul_thr) gr[i] -= e;
                        }
                    } else {
                        for (size_t i = 0; i < cols; i++) gr[i] += e * xp[i];
                    }
                }
            }

            // Softmax of scaled logits (also used for CE and the embedding grad).
            {
                float mx = -1e30f;
                for (int i = 0; i < V; i++) { float s = logits[i] * scale; if (s > mx) mx = s; }
                double sum = 0.0;
                for (int i = 0; i < V; i++) {
                    softmax_buf[i] = (float)std::exp((double)logits[i] * scale - mx);
                    sum += softmax_buf[i];
                }
                if (sum > 0) for (int i = 0; i < V; i++) softmax_buf[i] /= (float)sum;
                block_loss += -std::log((double)(softmax_buf[target] + 1e-12));
                valid_positions++;
            }

            // Embedding gradient: g = softmax(logits*scale) - onehot(target).
            if (target >= 0 && target < V) {
                const float* h = cap.h.data();
                // Variant-2 rule (--no-mul): keep the embedding error gv in full
                // precision, quantize the hidden activation h to ternary with an
                // absmean threshold (computed once per position, not per v).
                float hthr = 0.0f;
                if (no_mul) {
                    double hsum = 0.0;
                    for (int i = 0; i < H; i++) hsum += std::fabs(h[i]);
                    hthr = 0.5f * (H > 0 ? (float)(hsum / H) : 0.0f);
                }
                for (int v = 0; v < V; v++) {
                    float gv = softmax_buf[v] - (v == target ? 1.0f : 0.0f);
                    if (gv == 0.0f) continue;
                    float* ge = gradE.data() + (size_t)v * H;
                    if (no_mul) {
                        for (int i = 0; i < H; i++) {
                            if (h[i] > hthr)      ge[i] += gv;
                            else if (h[i] < -hthr) ge[i] -= gv;
                        }
                    } else {
                        for (int i = 0; i < H; i++) ge[i] += gv * h[i];
                    }
                }
            }

            prev = std::move(cap);
        }

        // Feed deltas into accumulators (rule 'c'). The capture names carry no
        // ".latent_weights" suffix; the weight map keys do.
        if (!no_ternary) {
            for (size_t l = 0; l < names.size(); l++) {
                auto it = model.ternary_weights.find(names[l] + ".latent_weights");
                if (it == model.ternary_weights.end()) continue;
                sl_feed_predictive(it->second, grads[l], decay, energy, sparsity);
            }
        }

        // Whole-block ARS (--ars-block): baseline probe right after the block
        // forward, before any embedding/flip updates. The rollback snapshots
        // for this step live here (per-tensor staged floats + embedding copy).
        std::vector<std::vector<float>> all_before;
        std::vector<float> emb_old;
        std::vector<std::pair<size_t, size_t>> ars_accepted;
        double pre_ce = -1.0;
        bool have_flip_final = false;
        double L_flip_final = -1.0;
        if (ars_block) {
            const size_t blk = (size_t)cache.pos;
            if (blk >= (size_t)ars_probe && blk + (size_t)ars_probe <= (size_t)model.header.max_seq_len)
                pre_ce = ars_probe_block(model, tokens, start, blk, ars_probe, ars_windows,
                                         scale, softmax_buf, cache, V);
        }

        // Embedding local SGD: per-row clip (norm<1 -> normalize) + decoupled WD.
        // With --ars-emb, the top --ars-emb-max rows by |grad| are trialled
        // through the block probe like flip chunks: apply the full LR step,
        // probe CE and keep it only if CE stays within the margin, else
        // revert the row. WD applies to every row regardless.
        if (!flip_only) {
            std::vector<int> emb_rows;
            std::vector<float> emb_norm;
            for (int v = 0; v < V; v++) {
                const float* ge = gradE.data() + (size_t)v * H;
                float norm = 0.0f;
                for (int i = 0; i < H; i++) norm += ge[i] * ge[i];
                norm = sqrtf(norm);
                if (norm > 0.0f) { emb_rows.push_back(v); emb_norm.push_back(norm); }
            }
            std::vector<int> emb_order(emb_rows.size());
            for (size_t i = 0; i < emb_order.size(); i++) emb_order[i] = (int)i;
            std::sort(emb_order.begin(), emb_order.end(),
                      [&](int a, int b) { return emb_norm[(size_t)a] > emb_norm[(size_t)b]; });
            emb_old.resize((size_t)V * H);
            memcpy(emb_old.data(), emb, (size_t)V * H * sizeof(float));
            long long emb_tried = 0, emb_accepted = 0;
            double L_emb = 0.0;
            for (size_t oi = 0; oi < emb_order.size(); oi++) {
                int v = emb_rows[(size_t)emb_order[oi]];
                float* ge = gradE.data() + (size_t)v * H;
                float* er = emb + (size_t)v * H;
                float s = (emb_norm[(size_t)emb_order[oi]] < 1.0f)
                              ? 1.0f / emb_norm[(size_t)emb_order[oi]] : 1.0f;
                for (int i = 0; i < H; i++) er[i] -= lr_emb * (ge[i] * s);
                if (ars_emb && emb_tried < (long long)ars_emb_max) {
                    if (emb_tried == 0)
                        L_emb = ars_probe_block(model, tokens, start, (size_t)cache.pos,
                                                ars_probe, ars_windows, scale,
                                                softmax_buf, cache, V);
                    double L_new = ars_probe_block(model, tokens, start, (size_t)cache.pos,
                                                   ars_probe, ars_windows, scale,
                                                   softmax_buf, cache, V);
                    if (L_new <= L_emb + ars_margin_eff(ars_margin, ars_margin_rel, L_emb)) {
                        L_emb = L_new;
                        emb_accepted++;
                    } else {
                        memcpy(er, emb_old.data() + (size_t)v * H, H * sizeof(float));
                    }
                    emb_tried++;
                }
            }
            if (ars_emb && emb_tried > 0)
                fprintf(stderr, "  ARS-emb: tried=%lld accepted=%lld probeCE=%.4f win=%d/%d (margin=%.4f)\n",
                        emb_tried, emb_accepted, L_emb, ars_windows, ars_probe,
                        (double)ars_margin_eff(ars_margin, ars_margin_rel, L_emb));
            // Decoupled weight decay over all rows (incl. reverted ones).
            for (int i = 0; i < V * H; i++) emb[i] *= (1.0f - lr_emb * wd_emb);
        }

        // Bit flips every N steps.
        long long total_flips = 0;
        long long n_changed = 0;
        long long real_changes = 0;
        bool flip_ran = false;
        long long ars_tried = 0, ars_ok = 0;
        if (!no_ternary && flip_every > 0 && (step + 1) % flip_every == 0) {
            flip_ran = true;
            // Exp 18: probe capacity scales with the proposal budget (k0/k_eff)
            // so the ARS gate is not the binding constraint on sustained churn.
            int trials_eff = ars_trials, chunk_eff = ars_chunk;
            if (churn_ramp > 0.0f) {
                double ratio = (double)adaptive_thr / (k_eff > 1e-6f ? k_eff : 1e-6f);
                trials_eff = (int)std::lround((double)ars_trials * ratio);
                if (trials_eff > churn_max_trials) trials_eff = churn_max_trials;
                if (trials_eff < 1) trials_eff = 1;
                chunk_eff = (int)std::lround((double)ars_chunk * ratio);
                if (chunk_eff > churn_max_chunk) chunk_eff = churn_max_chunk;
                if (chunk_eff < 1) chunk_eff = 1;
                flip_passes++;
            }
            long long acc_over20 = 0, acc_over15 = 0;
            double acc_max = 0;
            for (size_t wi = 0; wi < wlist.size(); wi++) {
                TernaryWeightXNOR& w = *wlist[wi];
                for (size_t i = 0; i < w.accumulator.size(); i++) {
                    float a = w.accumulator[i];
                    double aa = fabs((double)a);
                    if (aa > acc_max) acc_max = aa;
                    if (aa > 20.0) acc_over20++;
                    else if (aa > 15.0) acc_over15++;
                }
            }
            fprintf(stderr, "  [acc stats] max=%.4f >20=%lld in(15,20]=%lld\n",
                    acc_max, acc_over20, acc_over15);
            const float eff_thr = thr_anneal > 0.0f
                    ? thr + thr_anneal * ((float)((step + 1) / flip_every) - 1.0f)
                    : thr;
            const bool eff_toggle = toggle && (toggle_window <= 0 || (int)step + 1 <= toggle_window);
            all_before.resize(wlist.size());
            std::vector<std::vector<float>> all_after(wlist.size());
            std::vector<std::vector<size_t>> changed(wlist.size());
            for (size_t wi = 0; wi < wlist.size(); wi++) {
                TernaryWeightXNOR& w = *wlist[wi];
                const size_t n = w.floats.size();
                all_before[wi].resize(n);
                memcpy(all_before[wi].data(), w.floats.data(), n * sizeof(float));
                total_flips += apply_bit_flips(w, eff_thr, eff_toggle, model.sl_outlier_mult,
                                               k_eff);
                all_after[wi].resize(n);
                memcpy(all_after[wi].data(), w.floats.data(), n * sizeof(float));
                for (size_t i = 0; i < n; i++)
                    if (all_after[wi][i] != all_before[wi][i]) { changed[wi].push_back(i); n_changed++; }
            }
            if (ars && n_changed > 0
                    && (size_t)cache.pos >= (size_t)ars_probe
                    && (size_t)cache.pos + (size_t)ars_probe <= (size_t)model.header.max_seq_len) {
                // Roll every proposal back, then re-trial chunk by chunk on
                // probe windows replayed from the live cache.
const size_t blk = (size_t)cache.pos;
                for (size_t wi = 0; wi < wlist.size(); wi++)
                    for (size_t i : changed[wi]) wlist[wi]->floats[i] = all_before[wi][i];
                ars_repack(model);
                double L_cur = ars_probe_block(model, tokens, start, blk, ars_probe,
                                               ars_windows, scale, softmax_buf, cache, V);
                std::vector<std::pair<size_t, size_t>> flat;
                for (size_t wi = 0; wi < wlist.size(); wi++)
                    for (size_t i : changed[wi]) flat.emplace_back(wi, i);
                size_t tried = 0, ok_chunks = 0;
                real_changes = 0;
                for (size_t p = 0; p < flat.size(); p += (size_t)chunk_eff) {
                    size_t nc = (std::min)((size_t)chunk_eff, flat.size() - p);
                    if (tried >= (size_t)trials_eff) {
                        for (size_t k = 0; k < nc; k++)
                            wlist[flat[p + k].first]->floats[flat[p + k].second] =
                                all_before[flat[p + k].first][flat[p + k].second];
                    } else {
                        for (size_t k = 0; k < nc; k++)
                            wlist[flat[p + k].first]->floats[flat[p + k].second] =
                                all_after[flat[p + k].first][flat[p + k].second];
                        double L_new = ars_probe_block(model, tokens, start, blk, ars_probe,
                                                    ars_windows, scale, softmax_buf, cache, V);
                        float margin_eff = ars_margin_eff(ars_margin, ars_margin_rel, L_cur);
                        if (L_new <= L_cur + margin_eff) {
                            L_cur = L_new;
                            ok_chunks++;
                            for (size_t k = 0; k < nc; k++) {
                                hists[flat[p + k].first][flat[p + k].second]++;
                                real_changes++;
                                ars_accepted.emplace_back(flat[p + k].first, flat[p + k].second);
                            }
                        } else {
                            for (size_t k = 0; k < nc; k++)
                                wlist[flat[p + k].first]->floats[flat[p + k].second] =
                                    all_before[flat[p + k].first][flat[p + k].second];
                        }
                    }
                    tried++;
                }
                // The cache is rebuilt per block, so the K/V left by the last
                // probe dies with this step; restore the natural position
                // anyway so nothing depends on where the probe ended.
                cache.pos = (int)blk;
                ars_repack(model);
                have_flip_final = true;
                L_flip_final = L_cur;
                fprintf(stderr, "  ARS: trials=%zu accepted_chunks=%zu accepted_weights=%lld "
                                "probeCE=%.4f win=%d/%d (margin=%.4f)\n",
                        tried, ok_chunks, real_changes, L_cur, ars_windows,
                        ars_probe, (double)ars_margin_eff(ars_margin, ars_margin_rel, L_cur));
                ars_tried = (long long)((std::min)(tried, (size_t)trials_eff));
                ars_ok = (long long)ok_chunks;
            } else {
                for (size_t wi = 0; wi < wlist.size(); wi++) {
                    auto& hist = hists[wi];
                    for (size_t i : changed[wi]) {
                        hist[i]++; real_changes++;
                        ars_accepted.emplace_back(wi, i);
                    }
                }
            }
            // Rescue-then-freeze (finding #12): at the toggle-window boundary,
            // zero every accumulator so the mass-kick residual does not keep
            // driving real flips for hundreds of blocks after toggle turns off.
            if (toggle_window > 0 && (int)step + 1 == toggle_window) {
                size_t nacc = 0;
                for (size_t wi = 0; wi < wlist.size(); wi++)
                    nacc += wlist[wi]->accumulator.size();
                for (size_t wi = 0; wi < wlist.size(); wi++) {
                    auto& accs = wlist[wi]->accumulator;
                    std::fill(accs.begin(), accs.end(), 0.0f);
                }
                fprintf(stderr, "  >> toggle window ended at block %d: %zu accumulators zeroed\n",
                        toggle_window, nacc);
            }
        }

        // Whole-block ARS gate (--ars-block): keep this step's changes only if
        // the post-update probe CE stays within the margin of the pre-update
        // baseline; otherwise restore the staged float weights and the
        // embedding copy (full undo — even this step's weight decay reverts).
        bool block_reverted = false;
        if (ars_block && pre_ce >= 0.0) {
            const size_t blk = (size_t)cache.pos;
            double post_ce = L_flip_final;
            if (!have_flip_final) {
                if (blk >= (size_t)ars_probe && blk + (size_t)ars_probe <= (size_t)model.header.max_seq_len)
                    post_ce = ars_probe_block(model, tokens, start, blk, ars_probe, ars_windows,
                                              scale, softmax_buf, cache, V);
                else
                    post_ce = -1.0;
            }
            if (post_ce >= 0.0) {
                const double margin = ars_margin_eff(ars_margin, ars_margin_rel, pre_ce);
                if (post_ce > pre_ce + margin) {
                    for (size_t wi = 0; wi < wlist.size() && wi < all_before.size(); wi++)
                        if (!all_before[wi].empty())
                            memcpy(wlist[wi]->floats.data(), all_before[wi].data(),
                                   all_before[wi].size() * sizeof(float));
                    if (!all_before.empty()) ars_repack(model);
                    if (!flip_only && emb_old.size() == (size_t)V * H)
                        memcpy(emb, emb_old.data(), (size_t)V * H * sizeof(float));
                    real_changes = 0;
                    for (auto& p : ars_accepted)
                        if (hists[p.first][p.second] > 0) hists[p.first][p.second]--;
                    fprintf(stderr, "  ARS-block: pre=%.4f post=%.4f margin=%.4f -> REVERTED flips+emb\n",
                            pre_ce, post_ce, margin);
                    block_reverted = true;
                } else {
                    fprintf(stderr, "  ARS-block: pre=%.4f post=%.4f margin=%.4f -> kept\n",
                            pre_ce, post_ce, margin);
                }
            }
        }

        // Exp 18 churn-ramp reflex: after every flip pass, descend k (and the
        // probe capacity with it) while the gates keep accepting; on a
        // violation (chunk acceptance below --churn-min-accept, or the block
        // gate reverting the step) walk k back and halve the ramp rate so the
        // pipeline settles at the sustainable churn frontier.
        if (flip_ran && churn_ramp > 0.0f && ars && flip_passes >= churn_warmup) {
            const double accept = (ars_tried > 0) ? (double)ars_ok / (double)ars_tried : 1.0;
            bool violation = false;
            if (n_changed > 0 && ars_tried > 0 && accept < (double)churn_min_accept)
                violation = true;
            if (block_reverted) violation = true;
            if (n_changed == 0) {
                // No proposals at all: no signal to trade on, hold k.
                fprintf(stderr, "  [churn] k=%.3f no proposals — hold\n", k_eff);
            } else if (violation) {
                k_eff += 2.0f * ramp_rate;
                if (k_eff > adaptive_thr) k_eff = adaptive_thr;
                ramp_rate *= 0.5f;
                fprintf(stderr, "  [churn] k=%.3f accept=%.2f blockRevert=%d -> walk back, rate=%.3f\n",
                        k_eff, accept, block_reverted ? 1 : 0, ramp_rate);
            } else if (k_eff > churn_min_k) {
                k_eff -= ramp_rate;
                if (k_eff < churn_min_k) k_eff = churn_min_k;
                fprintf(stderr, "  [churn] k=%.3f accept=%.2f -> ramp down (%.3f/pass)\n",
                        k_eff, accept, ramp_rate);
            } else {
                fprintf(stderr,  "  [churn] k=%.3f accept=%.2f at floor — hold\n",
                        k_eff, accept);
            }
        }

        // Churn buckets: fraction of weights flipped at least 1/2/4/8 times cumulatively.
        long long ever = 0, m2 = 0, m4 = 0, m8 = 0;
        for (auto& h : hists)
            for (auto c : h) {
                if (c) ever++;
                if (c >= 2) m2++;
                if (c >= 4) m4++;
                if (c >= 8) m8++;
            }
        double tot = (double)total_w;
        const float eff_thr = thr_anneal > 0.0f
                ? thr + thr_anneal * ((float)((step + 1) / flip_every) - 1.0f)
                : thr;
        const std::string k_suffix = (churn_ramp > 0.0f)
                ? " k=" + std::to_string(k_eff) : "";
        fprintf(stderr, "  flips=%lld (real changes=%lld, %.1f%% no-op, eff_thr=%.1f%s) | "
                        "churn: ever=%.2f%% >=2=%.2f%% >=4=%.2f%% >=8=%.2f%%\n",
                total_flips, real_changes,
                total_flips > 0 ? 100.0 * (1.0 - (double)real_changes / total_flips) : 0.0,
                eff_thr, k_suffix.c_str(),
                ever / tot * 100.0, m2 / tot * 100.0, m4 / tot * 100.0, m8 / tot * 100.0);

        auto te = std::chrono::high_resolution_clock::now();
        ms_total += std::chrono::duration<double, std::milli>(te - ts).count();

        if (log_every > 0 && (step + 1) % log_every == 0) {
            double ce = valid_positions > 0 ? block_loss / valid_positions : 0.0;
            fprintf(stderr, "step %4d | block CE %.4f | %lld flips | %.1f ms/block\n",
                    step + 1, ce, total_flips, ms_total / (step + 1));
        }

        if (save_every > 0 && (step + 1) % save_every == 0) {
            save_model(model, out_path);
            fprintf(stderr, "Saved %s\n", out_path);
        }
    }

    save_model(model, out_path);
    double tot_ms = ms_total;
    fprintf(stderr, "Done. %d blocks in %.1f ms (%.2f ms/block). Wrote %s\n",
            steps, tot_ms, tot_ms / steps, out_path);
    return 0;
}
