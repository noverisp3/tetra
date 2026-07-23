// Ternary Operations for Tetra LLM - AVX-512 variant
// Uses 512-bit SIMD (16 float32/instruction) for 2x throughput vs AVX2
// Compiled separately with /arch:AVX512, loaded at runtime if CPU supports it

#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

// Pack 16 ternary floats at once using AVX-512
static void pack_16_floats(const float* data, uint8_t* out, int64_t base, int64_t n) {
    __m512 v = _mm512_loadu_ps(data + base);
    // Compare thresholds: val >= 0.5f -> +1=3 (but actually 2), val <= -0.5f -> -1=0, else 0=1
    // We'll generate uint8 indices then combine
    // Actually, just use the scalar approach for correctness (pack is not hot path)
    for (int i = 0; i < 16 && base + i < n; i++) {
        float f = data[base + i];
        int v = (f >= 0.5f) ? 2 : (f <= -0.5f) ? 0 : 1;
        out[base / 4 + i / 4] |= v << (6 - 2 * (i % 4));
    }
}

at::Tensor pack_ternary_avx512(at::Tensor w) {
    TORCH_CHECK(w.dtype() == torch::kFloat32 || w.dtype() == torch::kFloat16);
    TORCH_CHECK(w.is_contiguous());

    int64_t n = w.numel();
    int64_t padded = (n + 3) / 4 * 4;
    auto w_f = w.to(torch::kFloat32).contiguous();
    const float* data = w_f.data_ptr<float>();

    auto packed = torch::zeros(padded / 4, torch::kUInt8);
    auto* out = packed.data_ptr<uint8_t>();

    // Process in blocks of 16
    for (int64_t i = 0; i < padded; i += 16) {
        // Zero the 4 output bytes for this block
        for (int j = 0; j < 4; j++) out[i / 4 + j] = 0;
        pack_16_floats(data, out, i, n);
    }

    return packed;
}

at::Tensor unpack_ternary_avx512(at::Tensor packed, std::vector<int64_t> shape) {
    TORCH_CHECK(packed.dtype() == torch::kUInt8);
    TORCH_CHECK(packed.is_contiguous());

    int64_t total = 1;
    for (auto d : shape) total *= d;
    int64_t n_packed = (total + 3) / 4;
    TORCH_CHECK(packed.numel() >= n_packed);

    const auto* data = packed.data_ptr<uint8_t>();
    auto result = torch::empty(total, torch::kFloat32);
    auto* out = result.data_ptr<float>();

    // LUT for 4 possible 2-bit values
    static const float lut[4] = {-1.0f, 0.0f, 1.0f, 0.0f};
    // AVX-512 broadcast LUT into a register for vectorized lookup
    __m512 lut_v = _mm512_setr_ps(-1.0f, 0.0f, 1.0f, 0.0f,
                                   -1.0f, 0.0f, 1.0f, 0.0f,
                                   -1.0f, 0.0f, 1.0f, 0.0f,
                                   -1.0f, 0.0f, 1.0f, 0.0f);

    int64_t i = 0;
    // Process in blocks of 16 (4 packed bytes = 16 weights)
    for (; i + 15 < total; i += 16) {
        int byte_idx = i / 4;
        uint32_t w4;
        memcpy(&w4, data + byte_idx, 4);
        // Extract 16 2-bit values
        uint16_t vals[16];
        for (int j = 0; j < 16; j++) {
            vals[j] = (w4 >> (6 - 2 * (j % 4) + (j / 4) * 8)) & 3;
        }
        // Scatter to float array then load (simplest path, keep for correctness)
        float block[16];
        for (int j = 0; j < 16; j++) block[j] = lut[vals[j]];
        _mm512_storeu_ps(out + i, _mm512_loadu_ps(block));
    }

    // Remainder
    for (; i < total; i++) {
        int byte_idx = i / 4;
        int shift = 6 - 2 * (i % 4);
        out[i] = lut[(data[byte_idx] >> shift) & 3];
    }

    return result.reshape(shape);
}

// AVX-512 Fused Ternary Matmul
// TILE_OF = 16 (vs 8 for AVX2), 2x compute throughput
at::Tensor ternary_matmul_avx512(at::Tensor x, at::Tensor packed_weights,
                                  int64_t out_features, int64_t in_features, float scale) {
    TORCH_CHECK(x.dtype() == torch::kFloat32);
    TORCH_CHECK(packed_weights.dtype() == torch::kUInt8);
    TORCH_CHECK(packed_weights.is_contiguous());

    const auto* w = packed_weights.data_ptr<uint8_t>();
    auto x_sizes = x.sizes();
    TORCH_CHECK(x_sizes.size() == 3 || x_sizes.size() == 2);
    int64_t batch = (x_sizes.size() == 3) ? x_sizes[0] : 1;
    int64_t seq = (x_sizes.size() == 3) ? x_sizes[1] : x_sizes[0];

    auto result = torch::empty({batch, seq, out_features}, torch::kFloat32);
    auto* out = result.data_ptr<float>();
    const auto* x_data = x.data_ptr<float>();
    int64_t tokens = batch * seq;

    static const float lut[4] = {-1.0f, 0.0f, 1.0f, 0.0f};
    const int64_t TILE_OF = 16;

    for (int64_t t = 0; t < tokens; t++) {
        const float* row_x = x_data + t * in_features;
        float* row_out = out + t * out_features;

        for (int64_t of0 = 0; of0 < out_features; of0 += TILE_OF) {
            int64_t of_end = std::min(of0 + TILE_OF, out_features);
            int64_t n_of = of_end - of0;

            __m512 acc[TILE_OF];
            for (int64_t of = 0; of < n_of; of++) {
                acc[of] = _mm512_setzero_ps();
            }

            const uint8_t* w_row[TILE_OF];
            for (int64_t of = 0; of < n_of; of++) {
                w_row[of] = w + ((of0 + of) * in_features) / 4;
            }

            int64_t if_ = 0;
            // Process 16 input features at a time
            for (; if_ + 15 < in_features; if_ += 16) {
                __m512 xv = _mm512_loadu_ps(row_x + if_);

                for (int64_t of = 0; of < n_of; of++) {
                    // Load 4 packed bytes (16 weights)
                    uint32_t w4;
                    memcpy(&w4, w_row[of] + if_ / 4, 4);

                    float wf[16] = {
                        lut[(w4 >> 6) & 3], lut[(w4 >> 4) & 3],
                        lut[(w4 >> 2) & 3], lut[w4 & 3],
                        lut[(w4 >> 14) & 3], lut[(w4 >> 12) & 3],
                        lut[(w4 >> 10) & 3], lut[(w4 >> 8) & 3],
                        lut[(w4 >> 22) & 3], lut[(w4 >> 20) & 3],
                        lut[(w4 >> 18) & 3], lut[(w4 >> 16) & 3],
                        lut[(w4 >> 30) & 3], lut[(w4 >> 28) & 3],
                        lut[(w4 >> 26) & 3], lut[(w4 >> 24) & 3],
                    };
                    __m512 wv = _mm512_loadu_ps(wf);
                    acc[of] = _mm512_fmadd_ps(xv, wv, acc[of]);
                }
            }

            for (int64_t of = 0; of < n_of; of++) {
                float sum = _mm512_reduce_add_ps(acc[of]);

                for (int64_t if2 = if_; if2 < in_features; if2++) {
                    int byte_idx = ((of0 + of) * in_features + if2) / 4;
                    int shift = 6 - 2 * (((of0 + of) * in_features + if2) % 4);
                    int val = (w[byte_idx] >> shift) & 3;
                    if (val == 0) sum -= row_x[if2];
                    else if (val == 2) sum += row_x[if2];
                }

                row_out[of0 + of] = sum * scale;
            }
        }
    }

    return result;
}

// INT8 ternary matmul: int8 activations x packed ternary -> float32 output
// (scalar implementation, AVX-512 optimization can be added later)

at::Tensor ternary_matmul_int8_avx512(at::Tensor x_int8, at::Tensor packed_weights,
                                       int64_t out_features, int64_t in_features) {
    TORCH_CHECK(x_int8.dtype() == torch::kInt8);
    TORCH_CHECK(packed_weights.dtype() == torch::kUInt8);
    TORCH_CHECK(x_int8.is_contiguous());
    TORCH_CHECK(packed_weights.is_contiguous());

    auto sizes = x_int8.sizes();
    int64_t batch = (sizes.size() == 3) ? sizes[0] : 1;
    int64_t seq = (sizes.size() == 3) ? sizes[1] : sizes[0];
    int64_t tokens = batch * seq;

    const auto* x_data = x_int8.data_ptr<int8_t>();
    const auto* w = packed_weights.data_ptr<uint8_t>();

    auto result = torch::empty({batch, seq, out_features}, torch::kFloat32);
    auto* out = result.data_ptr<float>();

    for (int64_t t = 0; t < tokens; t++) {
        const int8_t* x_row = x_data + t * in_features;
        float* out_row = out + t * out_features;

        for (int64_t of = 0; of < out_features; of++) {
            const uint8_t* w_row = w + of * in_features / 4;
            int32_t acc = 0;

            int64_t if_ = 0;
            for (; if_ + 3 < in_features; if_ += 4) {
                uint8_t byte = w_row[if_ / 4];
                int v0 = (int)((byte >> 6) & 3) - 1;
                int v1 = (int)((byte >> 4) & 3) - 1;
                int v2 = (int)((byte >> 2) & 3) - 1;
                int v3 = (int)(byte & 3) - 1;
                if (v0 == 1)      acc += (int32_t)x_row[if_];
                else if (v0 == -1) acc -= (int32_t)x_row[if_];
                if (v1 == 1)      acc += (int32_t)x_row[if_+1];
                else if (v1 == -1) acc -= (int32_t)x_row[if_+1];
                if (v2 == 1)      acc += (int32_t)x_row[if_+2];
                else if (v2 == -1) acc -= (int32_t)x_row[if_+2];
                if (v3 == 1)      acc += (int32_t)x_row[if_+3];
                else if (v3 == -1) acc -= (int32_t)x_row[if_+3];
            }
            for (; if_ < in_features; if_++) {
                int byte_idx = (of * in_features + if_) / 4;
                int shift = 6 - 2 * ((of * in_features + if_) % 4);
                int code = (w[byte_idx] >> shift) & 3;
                if (code == 2)       acc += (int32_t)x_row[if_];
                else if (code == 0)  acc -= (int32_t)x_row[if_];
            }
            out_row[of] = (float)acc;
        }
    }
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Tetra ternary operations (AVX-512)";
    m.def("pack_ternary", &pack_ternary_avx512);
    m.def("unpack_ternary", &unpack_ternary_avx512);
    m.def("ternary_matmul", &ternary_matmul_avx512);
    m.def("ternary_matmul_int8", &ternary_matmul_int8_avx512);
}
