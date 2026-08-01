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
            "Usage: %s <model.bin> <tokens.bin> <out.bin> [steps] [log_every] [save_every]\n",
            argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* data_path  = argv[2];
    const char* out_path   = argv[3];
    int steps      = (argc > 4) ? atoi(argv[4]) : 200;
    int log_every  = (argc > 5) ? atoi(argv[5]) : 50;
    int save_every = (argc > 6) ? atoi(argv[6]) : 100;

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
    const float thr = model.sl_threshold;
    const float decay = model.sl_acc_decay;
    const int flip_every = model.sl_flip_every_n;
    const float scale = model.sl_logit_scale;
    const float lr_emb = model.sl_lr_embedding;
    const float wd_emb = model.sl_wd_embedding;

    // Per-block gradient buffers, keyed by layer name (filled from first capture).
    std::vector<std::string> names;
    std::vector<std::vector<float>> grads;

    // Embedding gradient buffer (V x H) and its flattened view.
    std::vector<float> gradE((size_t)V * H, 0.0f);
    float* emb = model.fp32_weights.at("token_embedding.weight").data.data();

    std::vector<float> softmax_buf(V);
    auto t1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "Init in %.1f ms | block=%d thr=%.1f decay=%.3f flipEvery=%d "
                    "scale=%.6f lrEmb=%.1e wdEmb=%.2f\n",
            std::chrono::duration<double, std::milli>(t1 - t0).count(),
            block, thr, decay, flip_every, scale, lr_emb, wd_emb);

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
        for (size_t l = 0; l < names.size(); l++) {
            auto it = model.ternary_weights.find(names[l] + ".latent_weights");
            if (it == model.ternary_weights.end()) continue;
            sl_feed_predictive(it->second, grads[l], decay);
        }

        // Embedding local SGD: per-row clip (norm<1 -> normalize) + decoupled WD.
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

        // Bit flips every N steps.
        long long total_flips = 0;
        if (flip_every > 0 && (step + 1) % flip_every == 0) {
            for (auto& kv : model.ternary_weights) {
                total_flips += apply_bit_flips(kv.second, thr);
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
