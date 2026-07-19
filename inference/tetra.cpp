// Tetra — C++ Inference Runner
// Loads tetra_model.bin and generates text.
//
// Usage:
//   tetra.exe <model.bin> <token_ids> [max_new_tokens] [temperature] [top_k] [top_p]
//   tetra.exe tetra_model.bin "373,378,67,338" 100 0.8 50 0.9
//
// Special tokens: 1=BOS, 2=EOS (auto-stops generation on EOS)
// Streaming: token IDs are printed as generated (stderr for timing)

#include "tetra.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>

static std::vector<int> parse_tokens(const char* str) {
    std::vector<int> tokens;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        tokens.push_back(std::stoi(item));
    }
    return tokens;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model.bin> <token_ids> [max_new_tokens] [temperature] [top_k] [top_p]\n", argv[0]);
        fprintf(stderr, "  Special tokens: 1=BOS, 2=EOS (auto-stops on EOS)\n");
        return 1;
    }

    const char* model_path = argv[1];
    std::vector<int> tokens = parse_tokens(argv[2]);
    int max_new = (argc > 3) ? atoi(argv[3]) : 100;
    float temp = (argc > 4) ? atof(argv[4]) : 0.8f;
    int top_k = (argc > 5) ? atoi(argv[5]) : 50;
    float top_p = (argc > 6) ? atof(argv[6]) : 0.9f;

    auto t0 = std::chrono::high_resolution_clock::now();
    tetra::Model model = tetra::load_model(model_path);
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "Model loaded in %.1f ms\n", load_ms);

    tetra::KVCache cache;
    cache.init(model.header.num_layers, model.header.max_seq_len, model.header.hidden_dim);

    // Model was trained without BOS — do NOT auto-prepend
    // User provides raw token IDs directly

    fprintf(stderr, "Prompt tokens: ");
    for (int t : tokens) fprintf(stderr, "%d ", t);
    fprintf(stderr, "\n");

    // Prefill: feed each prompt token one at a time (position comes from cache.pos)
    // Save logits from the last step — these predict the first new token
    auto t2 = std::chrono::high_resolution_clock::now();
    std::vector<float> logits;
    for (size_t i = 0; i < tokens.size(); i++) {
        std::vector<int> single = {tokens[i]};
        logits = tetra::forward(model, single, cache);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    fprintf(stderr, "Prefill: %.1f ms (%d tokens)\n", prefill_ms, (int)tokens.size());

    // Generate: feed only newly generated tokens (logits already from prefill for step 0)
    srand(42);
    auto t4 = std::chrono::high_resolution_clock::now();
    int generated = 0;
    bool stopped = false;
    for (int i = 0; i < max_new && !stopped; i++) {
        int next_token = tetra::sample(logits, temp, top_k, top_p);
        if (next_token >= (int)model.header.vocab_size) {
            next_token = model.header.vocab_size - 1;
        }

        // Stream: print token ID immediately
        if (i > 0) printf(" ");
        printf("%d", next_token);
        fflush(stdout);

        tokens.push_back(next_token);
        generated++;

        if (next_token == 2) stopped = true;  // EOS = 2

        // Forward only the new token to get logits for next step
        if (!stopped && i < max_new - 1) {
            std::vector<int> single = {next_token};
            logits = tetra::forward(model, single, cache);
        }
    }
    printf("\n");
    auto t5 = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
    double tok_per_sec = (double)generated / (gen_ms / 1000.0);
    fprintf(stderr, "Generate: %.1f ms (%d tokens, %.1f tok/s)%s\n",
            gen_ms, generated, tok_per_sec, stopped ? " [EOS]" : "");

    // Print all token IDs for Python detokenization
    printf("Output token IDs: ");
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i > 0) printf(",");
        printf("%d", tokens[i]);
    }
    printf("\n");
    fflush(stdout);

    return 0;
}
