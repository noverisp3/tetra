// Tetra — C++ Inference Runner
// Loads tetra_model.bin and generates text.
//
// Usage:
//   tetra.exe <model.bin> <token_ids> [max_new_tokens] [temperature]
//   tetra.exe tetra_model.bin "7454,2402,257,640" 100 0.8
//
// Or via Python wrapper:
//   python inference/run_inference.py tetra_model.bin "Once upon a time"

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
        fprintf(stderr, "Usage: %s <model.bin> <token_ids> [max_new_tokens] [temperature]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    std::vector<int> tokens = parse_tokens(argv[2]);
    int max_new = (argc > 3) ? atoi(argv[3]) : 100;
    float temp = (argc > 4) ? atof(argv[4]) : 0.8f;

    auto t0 = std::chrono::high_resolution_clock::now();
    tetra::Model model = tetra::load_model(model_path);
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "Model loaded in %.1f ms\n", load_ms);

    tetra::KVCache cache;
    cache.init(model.header.num_layers, model.header.max_seq_len, model.header.hidden_dim);

    fprintf(stderr, "Prompt tokens: ");
    for (int t : tokens) fprintf(stderr, "%d ", t);
    fprintf(stderr, "\n");

    // Prefill
    auto t2 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < tokens.size(); i++) {
        std::vector<int> prefix(tokens.begin(), tokens.begin() + i + 1);
        auto logits = tetra::forward(model, prefix, cache);
        (void)logits;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    fprintf(stderr, "Prefill: %.1f ms (%d tokens)\n", prefill_ms, (int)tokens.size());

    // Generate
    srand(42);
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < max_new; i++) {
        auto logits = tetra::forward(model, tokens, cache);
        int next_token = tetra::sample(logits, temp);
        if (next_token >= (int)model.header.vocab_size) {
            next_token = model.header.vocab_size - 1;
        }
        printf("%d ", next_token);
        tokens.push_back(next_token);
    }
    printf("\n");
    auto t5 = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
    double tok_per_sec = (double)max_new / (gen_ms / 1000.0);
    fprintf(stderr, "Generate: %.1f ms (%d tokens, %.1f tok/s)\n", gen_ms, max_new, tok_per_sec);

    // Print token IDs for Python detokenization
    printf("Output token IDs: ");
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i > 0) printf(",");
        printf("%d", tokens[i]);
    }
    printf("\n");
    fflush(stdout);

    return 0;
}
