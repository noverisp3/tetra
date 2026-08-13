// apply_flip_driver.cpp — parity harness for the apply_bit_flips kernel.
//
// Exposes the C++ on-device kernel (tetra.h::apply_bit_flips) as a small
// deterministic CLI so tests/test_apply_flip_parity.py can compare it
// bit-for-bit against ternary_llm.quantization.apply_bit_flips on identical
// inputs.
//
// Protocol (all little-endian, file-based; no stdout cosmetics):
//   argv:
//     <rows> <cols> <version(6|7|8)> <threshold> <toggle(0|1)>
//     <outlier_mult> <adaptive_thr> <in_dir> <out_dir>
//   in_dir/ (inputs, produced by the Python side):
//     packed.bin   uint8  ceil(rows*cols/4) bytes  (2-bit ternary codes)
//     acc.bin      f32    rows*cols                 (accumulator state)
//     blob.bin     uint8  optional                  (v7 sign bits / v8 magnitudes)
//   out_dir/ (outputs, consumed by the Python side):
//     packed.bin   uint8  repacked code-11 positions
//     acc.bin      f32    accumulator after the pass (parked / zeroed entries)
//     blob.bin     uint8  rebuilt dense sign blob (v7) / untouched (v6, v8)
//     flips.txt    int    value returned by apply_bit_flips
//
// Build (from inference/): g++ -O2 -std=c++17 -o apply_flip_driver apply_flip_driver.cpp

#include "tetra.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

using namespace tetra;

static bool read_file(const std::string& path, std::vector<uint8_t>& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return false; }
    out.resize((size_t)sz);
    if (sz > 0 && fread(out.data(), 1, (size_t)sz, f) != (size_t)sz) { fclose(f); return false; }
    fclose(f);
    return true;
}

static bool write_file(const std::string& path, const void* data, size_t n) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;
    if (n > 0 && fwrite(data, 1, n, f) != n) { fclose(f); return false; }
    fclose(f);
    return true;
}

int main(int argc, char** argv) {
    if (argc != 10) {
        fprintf(stderr, "usage: %s <rows> <cols> <ver 6|7|8> <thr> <toggle 0|1> "
                        "<outlier_mult> <adaptive_thr> <in_dir> <out_dir>\n", argv[0]);
        return 2;
    }
    const int rows = atoi(argv[1]);
    const int cols = atoi(argv[2]);
    const int version = atoi(argv[3]);
    const float thr = (float)atof(argv[4]);
    const bool toggle = atoi(argv[5]) != 0;
    const float outlier_mult = (float)atof(argv[6]);
    const float adaptive_thr = (float)atof(argv[7]);
    const std::string in_dir = argv[8];
    const std::string out_dir = argv[9];

    const size_t n = (size_t)rows * cols;
    const size_t n_packed = (n + 3) / 4;

    TernaryWeightXNOR w;
    w.rows = rows;
    w.cols = cols;
    w.is_v7 = (version == 7);
    w.is_v8 = (version == 8);

    std::vector<uint8_t> packed, acc_bytes, blob;
    if (!read_file(in_dir + "/packed.bin", packed) || packed.size() != n_packed) {
        fprintf(stderr, "bad packed.bin (%zu bytes, want %zu)\n", packed.size(), n_packed);
        return 2;
    }
    if (!read_file(in_dir + "/acc.bin", acc_bytes) || acc_bytes.size() != n * 4) {
        fprintf(stderr, "bad acc.bin (%zu bytes, want %zu)\n", acc_bytes.size(), n * 4);
        return 2;
    }
    if (!read_file(in_dir + "/blob.bin", blob)) {
        fprintf(stderr, "missing blob.bin\n");
        return 2;
    }
    w.packed = packed;
    w.accumulator.resize(n);
    memcpy(w.accumulator.data(), acc_bytes.data(), n * 4);
    w.outlier_blob = blob;

    // Dequantize packed+blob into w.floats (mirrors load_model / precompute_floats).
    w.floats.resize(n);
    const int row_bytes = (cols + 3) / 4;
    size_t outlier_idx = 0;
    for (int r = 0; r < rows; r++) {
        dequantize_row(w.packed.data(), r * row_bytes, cols,
                       w.floats.data() + (size_t)r * cols,
                       w.outlier_blob.empty() ? nullptr : w.outlier_blob.data(),
                       w.is_v8, outlier_idx);
    }

    const int flips = apply_bit_flips(w, thr, toggle, outlier_mult, adaptive_thr);

    char buf[64];
    snprintf(buf, sizeof buf, "%d", flips);
    if (!write_file(out_dir + "/flips.txt", buf, strlen(buf))) { fprintf(stderr, "write fail flips\n"); return 2; }
    if (!write_file(out_dir + "/packed.bin", w.packed.data(), w.packed.size())) { fprintf(stderr, "write fail packed\n"); return 2; }
    if (!write_file(out_dir + "/acc.bin", w.accumulator.data(), w.accumulator.size() * 4)) { fprintf(stderr, "write fail acc\n"); return 2; }
    if (!write_file(out_dir + "/blob.bin", w.outlier_blob.data(), w.outlier_blob.size())) { fprintf(stderr, "write fail blob\n"); return 2; }
    return 0;
}