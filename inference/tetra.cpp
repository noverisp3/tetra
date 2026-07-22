// Usage: tetra.exe <model.bin> <token_ids> [max_new_tokens] [temperature] [top_k] [top_p]
//   tetra.exe tetra_model.bin "373,378,67,338" 100 0.8 50 0.9
// Special tokens: 1=BOS, 2=EOS (auto-stops on EOS)

#include "tetra.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <ctime>

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

    fprintf(stderr, "Prompt tokens: ");
    for (int t : tokens) fprintf(stderr, "%d ", t);
    fprintf(stderr, "\n");

    // Batch prefill: process all prompt tokens in a single forward pass
    auto t2 = std::chrono::high_resolution_clock::now();
    std::vector<float> logits = tetra::forward(model, tokens, cache);
    auto t3 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    fprintf(stderr, "Prefill: %.1f ms (%d tokens)\n", prefill_ms, (int)tokens.size());

    srand((unsigned int)time(NULL));
    auto t4 = std::chrono::high_resolution_clock::now();
    int generated = 0;
    bool stopped = false;
    for (int i = 0; i < max_new && !stopped; i++) {
        int next_token = tetra::sample(logits, temp, top_k, top_p);
        if (next_token >= (int)model.header.vocab_size) {
            next_token = model.header.vocab_size - 1;
        }

        printf("%d\n", next_token);
        fflush(stdout);

        tokens.push_back(next_token);
        generated++;

        if (next_token == 2) stopped = true;

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

    printf("Output token IDs: ");
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i > 0) printf(",");
        printf("%d", tokens[i]);
    }
    printf("\n");
    fflush(stdout);

    return 0;
}
