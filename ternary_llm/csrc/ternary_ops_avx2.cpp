// Ternary Operations for Tetra LLM
// SIMD-accelerated pack/unpack/matmul for 2-bit ternary weights {-1, 0, +1}
//
// Build: See build.py
// Requires: AVX2 (Intel Haswell+, Intel Iris Xe)

#include <torch/extension.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <immintrin.h>

// Pack: float {-1, 0, +1} -> packed uint8 (4 weights/byte)
// Encoding: -1->0, 0->1, +1->2  (2 bits per weight)

at::Tensor pack_ternary(at::Tensor w) {
    TORCH_CHECK(w.dtype() == torch::kFloat32 || w.dtype() == torch::kFloat16,
                "pack_ternary: expected float32 or float16 input");
    TORCH_CHECK(w.is_contiguous(), "pack_ternary: input must be contiguous");

    int64_t n = w.numel();
    int64_t padded = (n + 3) / 4 * 4;
    auto w_f = w.to(torch::kFloat32).contiguous();
    const float* data = w_f.data_ptr<float>();

    auto packed = torch::empty(padded / 4, torch::kUInt8);
    auto* out = packed.data_ptr<uint8_t>();

    for (int64_t i = 0; i < padded; i += 4) {
        int v0 = (i+0 < n) ? ((data[i+0] >= 0.5f) ? 2 : (data[i+0] <= -0.5f) ? 0 : 1) : 1;
        int v1 = (i+1 < n) ? ((data[i+1] >= 0.5f) ? 2 : (data[i+1] <= -0.5f) ? 0 : 1) : 1;
        int v2 = (i+2 < n) ? ((data[i+2] >= 0.5f) ? 2 : (data[i+2] <= -0.5f) ? 0 : 1) : 1;
        int v3 = (i+3 < n) ? ((data[i+3] >= 0.5f) ? 2 : (data[i+3] <= -0.5f) ? 0 : 1) : 1;
        out[i/4] = (uint8_t)(v0 * 64 + v1 * 16 + v2 * 4 + v3);
    }

    return packed;
}

// Unpack: packed uint8 -> float {-1, 0, +1}

at::Tensor unpack_ternary(at::Tensor packed, std::vector<int64_t> shape) {
    TORCH_CHECK(packed.dtype() == torch::kUInt8, "unpack_ternary: expected uint8 packed input");
    TORCH_CHECK(packed.is_contiguous(), "unpack_ternary: input must be contiguous");

    int64_t total = 1;
    for (auto d : shape) total *= d;
    int64_t n_packed = (total + 3) / 4;
    TORCH_CHECK(packed.numel() >= n_packed,
                "unpack_ternary: packed tensor too small");

    const auto* data = packed.data_ptr<uint8_t>();
    auto result = torch::empty(total, torch::kFloat32);
    auto* out = result.data_ptr<float>();

    static const float lut[4] = {-1.0f, 0.0f, 1.0f, 0.0f};

    for (int64_t i = 0; i < total; i++) {
        int byte_idx = i / 4;
        int shift = 6 - 2 * (i % 4);
        int val = (data[byte_idx] >> shift) & 3;
        out[i] = lut[val];
    }

    return result.reshape(shape);
}

// Fused Ternary Matmul: y = x @ (unpack(W) * scale)
// Unpacks weights on-the-fly, no intermediate float weight tensor

at::Tensor ternary_matmul(at::Tensor x, at::Tensor packed_weights,
                          int64_t out_features, int64_t in_features, float scale) {
    TORCH_CHECK(x.dtype() == torch::kFloat32, "ternary_matmul: x must be float32");
    TORCH_CHECK(packed_weights.dtype() == torch::kUInt8,
                "ternary_matmul: packed_weights must be uint8");
    TORCH_CHECK(packed_weights.is_contiguous(),
                "ternary_matmul: packed_weights must be contiguous");

    const auto* w = packed_weights.data_ptr<uint8_t>();
    auto x_sizes = x.sizes();
    TORCH_CHECK(x_sizes.size() == 3 || x_sizes.size() == 2,
                "ternary_matmul: x must be 2D or 3D");
    int64_t batch = (x_sizes.size() == 3) ? x_sizes[0] : 1;
    int64_t seq = (x_sizes.size() == 3) ? x_sizes[1] : x_sizes[0];

    auto result = torch::empty({batch, seq, out_features}, torch::kFloat32);
    auto* out = result.data_ptr<float>();
    const auto* x_data = x.data_ptr<float>();

    int64_t tokens = batch * seq;
    static const float lut[4] = {-1.0f, 0.0f, 1.0f, 0.0f};

    // Tile over output features for better cache reuse of x
    const int64_t TILE_OF = 8;  // process 8 output features at a time

    for (int64_t t = 0; t < tokens; t++) {
        const float* row_x = x_data + t * in_features;
        float* row_out = out + t * out_features;

        for (int64_t of0 = 0; of0 < out_features; of0 += TILE_OF) {
            int64_t of_end = std::min(of0 + TILE_OF, out_features);
            int64_t n_of = of_end - of0;

            // Accumulators for the tile
            __m256 acc[TILE_OF];
            for (int64_t of = 0; of < n_of; of++) {
                acc[of] = _mm256_setzero_ps();
            }

            // Process input channels in blocks of 8
            const uint8_t* w_row[TILE_OF];
            for (int64_t of = 0; of < n_of; of++) {
                w_row[of] = w + ((of0 + of) * in_features) / 4;
            }

            int64_t if_ = 0;
            for (; if_ + 7 < in_features; if_ += 8) {
                __m256 xv = _mm256_loadu_ps(row_x + if_);

                // For each output feature in the tile
                for (int64_t of = 0; of < n_of; of++) {
                    // Load 2 packed bytes (8 weights) for this row
                    uint16_t w2;
                    memcpy(&w2, w_row[of] + if_ / 4, 2);

                    float wf[8] = {
                        lut[(w2 >> 6) & 3], lut[(w2 >> 4) & 3],
                        lut[(w2 >> 2) & 3], lut[w2 & 3],
                        lut[(w2 >> 14) & 3], lut[(w2 >> 12) & 3],
                        lut[(w2 >> 10) & 3], lut[(w2 >> 8) & 3],
                    };
                    __m256 wv = _mm256_loadu_ps(wf);
                    acc[of] = _mm256_fmadd_ps(xv, wv, acc[of]);
                }
            }

            // Write tile results
            for (int64_t of = 0; of < n_of; of++) {
                __m128 hi = _mm_add_ps(_mm256_castps256_ps128(acc[of]),
                                      _mm256_extractf128_ps(acc[of], 1));
                hi = _mm_hadd_ps(hi, hi);
                hi = _mm_hadd_ps(hi, hi);
                float sum = _mm_cvtss_f32(hi);

                // Scalar remainder
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

// Fused Stochastic Backward: returns grad_x and updates accumulator in-place

std::vector<at::Tensor> stochastic_backward(
    at::Tensor grad_output,
    at::Tensor x,
    at::Tensor packed_weights,
    int64_t out_features,
    int64_t in_features,
    float scale,
    at::Tensor accumulator
) {
    TORCH_CHECK(x.dtype() == torch::kFloat32, "stochastic_backward: x must be float32");
    TORCH_CHECK(grad_output.dtype() == torch::kFloat32,
                "stochastic_backward: grad_output must be float32");
    TORCH_CHECK(accumulator.dtype() == torch::kFloat32,
                "stochastic_backward: accumulator must be float32");
    TORCH_CHECK(packed_weights.dtype() == torch::kUInt8,
                "stochastic_backward: packed_weights must be uint8");
    TORCH_CHECK(grad_output.is_contiguous(),
                "stochastic_backward: grad_output must be contiguous");

    const auto* w = packed_weights.data_ptr<uint8_t>();
    auto x_sizes = x.sizes();
    TORCH_CHECK(x_sizes.size() == 3 || x_sizes.size() == 2,
                "stochastic_backward: x must be 2D or 3D");
    int64_t total_tokens = (x_sizes.size() == 3) ? x_sizes[0] * x_sizes[1] : x_sizes[0];

    const float* x_data = x.data_ptr<float>();
    const float* grad_data = grad_output.data_ptr<float>();
    float* acc_data = accumulator.data_ptr<float>();

    // grad_x = grad_output @ (unpack(W) * scale)^T
    auto grad_x = torch::empty({total_tokens, in_features}, torch::kFloat32);
    float* grad_x_data = grad_x.data_ptr<float>();
    memset(grad_x_data, 0, total_tokens * in_features * sizeof(float));

    for (int64_t t = 0; t < total_tokens; t++) {
        const float* g_row = grad_data + t * out_features;
        float* gx_row = grad_x_data + t * in_features;

        for (int64_t of = 0; of < out_features; of++) {
            float g = g_row[of];
            for (int64_t if_ = 0; if_ < in_features; if_++) {
                int byte_idx = (of * in_features + if_) / 4;
                int shift = 6 - 2 * ((of * in_features + if_) % 4);
                int val = (w[byte_idx] >> shift) & 3;
                if (val == 0) gx_row[if_] -= g * scale;
                else if (val == 2) gx_row[if_] += g * scale;
            }
        }
    }

    // accumulator += sign(grad_output^T @ x) * (-1) / scale  (negate for SGD)
    float inv_scale = 1.0f / scale;
    for (int64_t of = 0; of < out_features; of++) {
        for (int64_t if_ = 0; if_ < in_features; if_++) {
            float grad_w = 0.0f;
            for (int64_t t = 0; t < total_tokens; t++) {
                grad_w += grad_data[t * out_features + of] * x_data[t * in_features + if_];
            }

            // sign(-grad_w) / scale = -sign(grad_w) / scale
            if (grad_w > 0) acc_data[of * in_features + if_] -= inv_scale;
            else if (grad_w < 0) acc_data[of * in_features + if_] += inv_scale;
        }
    }

    if (x_sizes.size() == 3) {
        return {grad_x.view({x_sizes[0], x_sizes[1], in_features})};
    }
    return {grad_x};
}

// Apply bit flips: check accumulator, flip bits where threshold exceeded

int64_t apply_bit_flips(at::Tensor packed_weights, at::Tensor accumulator,
                          float threshold, std::vector<int64_t> shape_w) {
    TORCH_CHECK(packed_weights.dtype() == torch::kUInt8);
    TORCH_CHECK(accumulator.dtype() == torch::kFloat32);

    int64_t total = 1;
    for (auto d : shape_w) total *= d;
    int64_t n_packed = (total + 3) / 4;

    auto* packed = packed_weights.data_ptr<uint8_t>();
    const auto* acc = accumulator.data_ptr<float>();

    // Unpack -> flip -> repack
    auto w = unpack_ternary(packed_weights, shape_w);
    auto* w_data = w.data_ptr<float>();

    int64_t flips = 0;
    for (int64_t i = 0; i < total; i++) {
        if (acc[i] > threshold) {
            w_data[i] = std::min(w_data[i] + 1.0f, 1.0f);
            flips++;
        } else if (acc[i] < -threshold) {
            w_data[i] = std::max(w_data[i] - 1.0f, -1.0f);
            flips++;
        }
    }

    if (flips > 0) {
        auto new_packed = pack_ternary(w);
        memcpy(packed, new_packed.data_ptr<uint8_t>(), n_packed);
        accumulator.zero_();
    }

    return flips;
}

// INT8 ternary matmul: int8 activations x packed ternary weights -> float32 output
// Pure integer in forward: conditional add/sub, int32 accumulation, final float scale
// Returns float32 for seamless integration with float norms/attention

at::Tensor ternary_matmul_int8(at::Tensor x_int8, at::Tensor packed_weights,
                                int64_t out_features, int64_t in_features) {
    TORCH_CHECK(x_int8.dtype() == torch::kInt8, "x_int8 must be int8");
    TORCH_CHECK(packed_weights.dtype() == torch::kUInt8,
                "packed_weights must be uint8");
    TORCH_CHECK(x_int8.is_contiguous(), "x_int8 must be contiguous");
    TORCH_CHECK(packed_weights.is_contiguous(), "packed_weights must be contiguous");

    auto sizes = x_int8.sizes();
    TORCH_CHECK(sizes.size() == 3 || sizes.size() == 2, "x_int8 must be 2D or 3D");
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
                // 00=-1, 01=0, 10=1, 11=0
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

            // Scalar remainder
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

// PyTorch bindings

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Tetra ternary operations";
    m.def("pack_ternary", &pack_ternary, "Pack float {-1,0,+1} to packed uint8 (4 weights/byte)");
    m.def("unpack_ternary", &unpack_ternary, "Unpack uint8 to float {-1,0,+1}");
    m.def("ternary_matmul", &ternary_matmul,
          "Fused ternary matmul (unpack-on-the-fly, no intermediate tensor)");
    m.def("stochastic_backward", &stochastic_backward,
          "Fused backward: compute grad_x + accumulate sign(-grad_w)/scale");
    m.def("apply_bit_flips", &apply_bit_flips,
          "Check accumulators and flip bits where threshold exceeded");
    m.def("ternary_matmul_int8", &ternary_matmul_int8,
          "INT8 ternary matmul: int8 activations x packed ternary -> int32 accum -> float");
}
