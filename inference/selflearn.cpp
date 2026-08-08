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

int main(int argc, char** argv) {
    // Eval-only mode: --eval <model.bin> <tokens.bin> [max_positions]
    // Reports average next-token cross-entropy without any learning.
    if (argc >= 4 && strcmp(argv[1], "--eval") == 0) {
        Model model = load_model(argv[2]);
        std::vector<uint16_t> tokens = read_tokens(argv[3]);
        size_t limit = tokens.size() > 1 ? tokens.size() - 1 : 0;
        if (argc > 4) {
            long n = atol(argv[4]);
            if (n > 0 && (size_t)n < limit) limit = (size_t)n;
        }
        fprintf(stderr, "Eval: %zu tokens (%zu positions)\n", tokens.size(), limit);
        if (limit == 0) { fprintf(stderr, "No tokens\n"); return 1; }

        const int H = model.header.hidden_dim;
        const int V = model.header.vocab_size;
        const float scale = model.sl_logit_scale;
        KVCache cache;
        cache.init(model.header.num_layers, model.header.max_seq_len, H,
                   model.is_mla, model.kv_latent_dim, model.rope_dim);

        std::vector<float> softmax_buf(V);
        double loss = 0.0;
        size_t pos = 0;
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
        fprintf(stderr, "Eval done. avg CE %.4f | PPL %.4f | %zu positions\n",
                loss / pos, std::exp(loss / pos), pos);
        return 0;
    }

    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <model.bin> <tokens.bin> <out.bin> [steps] [log_every] [save_every] [thr] [decay] [flip_every] [toggle] [--toggle-window N] [--thr-anneal RATE] [--energy] [--adaptive-thr K]\n",
            argv[0]);
        fprintf(stderr,
            "  thr/decay/flip_every/toggle override the v6 metadata (0 = keep metadata value; -1 for toggle = keep metadata)\n");
        fprintf(stderr,
            "  --toggle-window N: only toggle for the first N blocks, then no-op flips (annealing, finding #12)\n"
            "  --thr-anneal RATE: raise the flip threshold by RATE per pass (finding #12 refinement)\n"
            "  --energy: feed -grad magnitude into accumulators (Exp 3, needed with --adaptive-thr)\n"
            "  --adaptive-thr K: per-channel flip threshold tau = K * RMS(acc) (Exp 3, scale-invariant)\n");
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
    // Exp 3 flip mechanics: "--energy" switches the accumulator feed from
    // -sign(grad) votes to -grad magnitude; "--adaptive-thr K" sets the
    // per-channel flip threshold tau = K * RMS(acc). Negative value = keep the
    // exported metadata value.
    bool energy_override = false;
    float adaptive_override = -1.0f;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--energy") == 0) energy_override = true;
    }
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--adaptive-thr") == 0) {
            adaptive_override = (float)atof(argv[i + 1]);
        }
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
    const float scale = model.sl_logit_scale;
    const float lr_emb = model.sl_lr_embedding;
    const float wd_emb = model.sl_wd_embedding;
    const bool energy = energy_override ? true : (model.sl_energy != 0);
    const float adaptive_thr = adaptive_override >= 0.0f ? adaptive_override : model.sl_adaptive_thr;

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
                    "scale=%.6f lrEmb=%.1e wdEmb=%.2f%s%s energy=%d adaptiveThr=%.3f\n",
            std::chrono::duration<double, std::milli>(t1 - t0).count(),
            block, thr, decay, flip_every, toggle ? 1 : 0,
            (toggle_window > 0 ? " (anneal off after block " + std::to_string(toggle_window) + ")" : "").c_str(),
            scale, lr_emb, wd_emb,
            (thr_anneal > 0.0f ? " | thr-anneal +" + std::to_string(thr_anneal) + "/pass" : "").c_str(),
            no_ternary ? " | embedding-only" : (flip_only ? " | flip-only" : ""),
            energy ? 1 : 0, adaptive_thr);

    double ms_total = 0.0;
    for (int step = 0; step < steps; step++) {
        auto ts = std::chrono::high_resolution_clock::now();
        KVCache cache;
        cache.init(model.header.num_layers, model.header.max_seq_len, H,
                   model.is_mla, model.kv_latent_dim, model.rope_dim);

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
                for (size_t o = 0; o < rows; o++) {
                    float e = cur.y[o] - pr.y[o];
                    if (e == 0.0f) continue;
                    const float* xp = pr.x.data();
                    float* gr = g.data() + o * cols;
                    for (size_t i = 0; i < cols; i++) gr[i] += e * xp[i];
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
                for (int v = 0; v < V; v++) {
                    float gv = softmax_buf[v] - (v == target ? 1.0f : 0.0f);
                    if (gv == 0.0f) continue;
                    float* ge = gradE.data() + (size_t)v * H;
                    for (int i = 0; i < H; i++) ge[i] += gv * h[i];
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
                sl_feed_predictive(it->second, grads[l], decay, energy);
            }
        }

        // Embedding local SGD: per-row clip (norm<1 -> normalize) + decoupled WD.
        if (!flip_only) {
        for (int v = 0; v < V; v++) {
            float* ge = gradE.data() + (size_t)v * H;
            float norm = 0.0f;
            for (int i = 0; i < H; i++) norm += ge[i] * ge[i];
            norm = sqrtf(norm);
            float s = 1.0f;
            if (norm > 0.0f && norm < 1.0f) s = 1.0f / norm;
            float* er = emb + (size_t)v * H;
            for (int i = 0; i < H; i++) {
                float upd = lr_emb * (ge[i] * s);
                er[i] -= upd;
                er[i] *= (1.0f - lr_emb * wd_emb);
            }
        }
        }

        // Bit flips every N steps.
        long long total_flips = 0;
        long long real_changes = 0;
        if (!no_ternary && flip_every > 0 && (step + 1) % flip_every == 0) {
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
            for (size_t wi = 0; wi < wlist.size(); wi++) {
                TernaryWeightXNOR& w = *wlist[wi];
                auto& hist = hists[wi];
                const size_t n = w.floats.size();
                std::vector<float> before(n);
                memcpy(before.data(), w.floats.data(), n * sizeof(float));
                total_flips += apply_bit_flips(w, eff_thr, eff_toggle, model.sl_outlier_mult,
                                               adaptive_thr);
                const float* f = w.floats.data();
                for (size_t i = 0; i < n; i++)
                    if (f[i] != before[i]) { hist[i]++; real_changes++; }
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
            fprintf(stderr, "  flips=%lld (real changes=%lld, %.1f%% no-op, eff_thr=%.1f) | "
                            "churn: ever=%.2f%% >=2=%.2f%% >=4=%.2f%% >=8=%.2f%%\n",
                    total_flips, real_changes,
                    total_flips > 0 ? 100.0 * (1.0 - (double)real_changes / total_flips) : 0.0,
                    eff_thr,
                    ever / tot * 100.0, m2 / tot * 100.0, m4 / tot * 100.0, m8 / tot * 100.0);

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
