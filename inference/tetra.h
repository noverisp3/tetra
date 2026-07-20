#pragma once
// Tetra — Pure Ternary LLM Inference Engine
// Ternary weights {-1, 0, +1} with SIMD-accelerated matmul.
//
// Matmul Strategy:
//   1. At load time, dequantize 2-bit packed ternary → float {-1, 0, +1}
//   2. At inference, compute dot product using SIMD (AVX10 / AVX-512 / AVX2+FMA / scalar)
//   3. This gives EXACT quality (matches PyTorch F.linear) at SIMD speed
//
// SIMD dispatch priority:
//   AVX10 → AVX-512 → AVX2+FMA → scalar (detected at compile time)
//
// Build:
//   build.bat          → scalar fallback
//   build.bat avx2     → AVX2+FMA
//   build.bat avx512   → AVX-512
//   build.bat avx10    → AVX10
//
// Binary Format v2:
//   Header (64B): magic "TETR", version, model dims, param counts
//   Ternary weights: name, shape, alpha (absmean), 2-bit packed data
//   FP32 weights: embeddings, norms (lm_head tied to embedding)

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cstdlib>

#ifdef _MSC_VER
#include <intrin.h>
#define TETRA_POPCOUNT64(x) __popcnt64(x)
#else
#define TETRA_POPCOUNT64(x) __builtin_popcountll(x)
#endif

#include <immintrin.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#define TETRA_HAS_NEON 1
#endif

namespace tetra {

// Portable popcount fallback
#ifndef _MSC_VER
static inline uint64_t TETRA_POPCOUNT64(uint64_t x) {
    return __builtin_popcountll(x);
}
#endif

// Ternary weight format for XNOR+popcount
// Each row stores two bitmasks:
//   pos_mask[word]: bit i = 1 iff weight[row][i] == +1
//   neg_mask[word]: bit i = 1 iff weight[row][i] == -1
// Both zero → weight is 0 (skip).
struct TernaryWeightXNOR {
    std::vector<uint64_t> pos_masks;  // [rows * words_per_row]
    std::vector<uint64_t> neg_masks;  // [rows * words_per_row]
    std::vector<uint8_t> packed;      // original 2-bit packed ternary data
    std::vector<float> floats;        // pre-dequantized {-1, 0, +1} as float — SIMD-ready
    int rows, cols;
    int words_per_row;  // ceil(cols / 64)
    float alpha;        // v2: per-matrix absmean scale factor
};

// Decode-optimized LUT for 2-bit → float conversion
// Single table lookup per 4 weights — eliminates bit shifts at decode time.
// 256 entries × 16 bytes = 4 KB (fits L1 cache).
struct ByteDecoded { float v[4]; };
static ByteDecoded BYTE_LUT[256];

static void init_byte_lut() {
    static bool initialized = false;
    if (initialized) return;
    const float vals[4] = {-1.0f, 0.0f, 1.0f, 0.0f};
    for (int b = 0; b < 256; b++) {
        BYTE_LUT[b].v[0] = vals[(b >> 6) & 3];
        BYTE_LUT[b].v[1] = vals[(b >> 4) & 3];
        BYTE_LUT[b].v[2] = vals[(b >> 2) & 3];
        BYTE_LUT[b].v[3] = vals[b & 3];
    }
    initialized = true;
}

struct FP32Weight {
    std::vector<float> data;
    std::vector<int> shape;
};

struct ModelHeader {
    char magic[4];
    uint32_t version, vocab_size, hidden_dim, num_layers, num_heads;
    uint32_t ffn_dim, max_seq_len;
    uint64_t ternary_params, fp32_params;
};

// Convert 2-bit packed ternary to XNOR bitmasks
static TernaryWeightXNOR convert_to_xnor(
    const uint8_t* packed, int rows, int cols
) {
    TernaryWeightXNOR w;
    w.rows = rows;
    w.cols = cols;
    w.words_per_row = (cols + 63) / 64;
    w.pos_masks.resize(rows * w.words_per_row, 0);
    w.neg_masks.resize(rows * w.words_per_row, 0);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int flat = r * cols + c;
            int byte_idx = flat / 4;
            int bit_pos  = flat % 4;
            int shift = 6 - bit_pos * 2;
            uint8_t code = (packed[byte_idx] >> shift) & 0b11;

            int word = c / 64;
            int bit  = c % 64;

            if (code == 0b10) {       // +1
                w.pos_masks[r * w.words_per_row + word] |= (1ULL << bit);
            } else if (code == 0b00) { // -1
                w.neg_masks[r * w.words_per_row + word] |= (1ULL << bit);
            }
            // code == 01 or 11 → zero weight, both masks stay 0
        }
    }
    return w;
}

// XNOR+popcount: extract activation signs, pack into bitmask
// For a vector of floats, extract sign bit of each float and pack
// them into uint64 words. sign_mask[word] bit i = 1 iff x[word*64+i] > 0.
static void extract_sign_mask(const float* x, int n, uint64_t* sign_mask) {
    int words = (n + 63) / 64;
    for (int w = 0; w < words; w++) sign_mask[w] = 0;

#if defined(__AVX10_1__) || defined(__AVX10__)
    // AVX10: _mm256_movepi32_mask processes 8 floats per iteration (needs 8× for 64)
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512 v0 = _mm512_loadu_ps(x + i);
        __m512 v1 = _mm512_loadu_ps(x + i + 16);
        __m512 v2 = _mm512_loadu_ps(x + i + 32);
        __m512 v3 = _mm512_loadu_ps(x + i + 48);
        uint64_t s0 = (uint64_t)_mm512_movepi32_mask(_mm512_castps_si512(v0));
        uint64_t s1 = (uint64_t)_mm512_movepi32_mask(_mm512_castps_si512(v1));
        uint64_t s2 = (uint64_t)_mm512_movepi32_mask(_mm512_castps_si512(v2));
        uint64_t s3 = (uint64_t)_mm512_movepi32_mask(_mm512_castps_si512(v3));
        sign_mask[i / 64] = s0 | (s1 << 16) | (s2 << 32) | (s3 << 48);
    }
    for (; i < n; i++) {
        if (x[i] > 0.0f) sign_mask[i / 64] |= (1ULL << (i % 64));
    }
#elif defined(__AVX512F__)
    // Process 64 floats at a time (4 × __m512)
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        __m512 v0 = _mm512_loadu_ps(x + i);
        __m512 v1 = _mm512_loadu_ps(x + i + 16);
        __m512 v2 = _mm512_loadu_ps(x + i + 32);
        __m512 v3 = _mm512_loadu_ps(x + i + 48);

        // _mm512_movepi32_mask: extracts sign bit (bit 31) of each 32-bit int
        uint64_t s0 = (uint64_t)_mm512_movepi32_mask(_mm512_castps_si512(v0));
        uint64_t s1 = (uint64_t)_mm512_movepi32_mask(_mm512_castps_si512(v1));
        uint64_t s2 = (uint64_t)_mm512_movepi32_mask(_mm512_castps_si512(v2));
        uint64_t s3 = (uint64_t)_mm512_movepi32_mask(_mm512_castps_si512(v3));

        sign_mask[i / 64] = s0 | (s1 << 16) | (s2 << 32) | (s3 << 48);
    }
    // Handle remainder
    for (; i < n; i++) {
        if (x[i] > 0.0f) sign_mask[i / 64] |= (1ULL << (i % 64));
    }
#elif defined(__AVX2__)
    // AVX2: _mm256_movemask_ps extracts 8 sign bits at a time
    int i = 0;
    for (; i + 64 <= n; i += 64) {
        uint64_t mask = 0;
        for (int j = 0; j < 8; j++) {
            __m256 v = _mm256_loadu_ps(x + i + j * 8);
            int bits = _mm256_movemask_ps(v);
            mask |= (uint64_t)(uint8_t)bits << (j * 8);
        }
        sign_mask[i / 64] = mask;
    }
    for (; i < n; i++) {
        if (x[i] > 0.0f) sign_mask[i / 64] |= (1ULL << (i % 64));
    }
#else
    // Scalar fallback: extract sign bits one by one
    for (int i = 0; i < n; i++) {
        if (x[i] > 0.0f) sign_mask[i / 64] |= (1ULL << (i % 64));
    }
#endif
}

// Core XNOR+popcount dot product
// Compute dot product of activation signs with one ternary weight row.
// Returns integer score in [-cols, cols].
static inline int64_t xnor_popcount_row(
    const uint64_t* sign_mask,   // packed sign bits of activations
    const uint64_t* pos_mask,    // positive weight bitmask
    const uint64_t* neg_mask,    // negative weight bitmask
    int words_per_row
) {
    int64_t pos_count = 0, neg_count = 0;
    for (int w = 0; w < words_per_row; w++) {
        // XNOR = NOT(XOR): bit is 1 when both inputs match
        uint64_t xnor_pos = ~(sign_mask[w] ^ pos_mask[w]);
        uint64_t xnor_neg = ~(sign_mask[w] ^ neg_mask[w]);
        pos_count += TETRA_POPCOUNT64(xnor_pos);
        neg_count += TETRA_POPCOUNT64(xnor_neg);
    }
    return pos_count - neg_count;
}

// Full ternary matmul via XNOR+popcount
// out[r] = Σ sign(x[c]) * w[r][c] for all c
// This approximates the real dot product using only sign information.
// The scale factor compensates for lost magnitude information.
static void ternary_matmul_xnor(
    const float* x,                        // [cols] activations
    const TernaryWeightXNOR& w,            // precomputed XNOR weights
    float* out,                            // [rows] output
    float x_absmean,                       // mean(|x|) for activation scale recovery
    float alpha = 1.0f                     // v2: per-matrix weight scale α = mean(|W_latent|)
) {
    int words = w.words_per_row;

    // Precompute sign mask once (shared across all rows)
    std::vector<uint64_t> sign_mask(words, 0);
    extract_sign_mask(x, w.cols, sign_mask.data());

    // For each row: XNOR+popcount
    for (int r = 0; r < w.rows; r++) {
        const uint64_t* row_pos = w.pos_masks.data() + r * words;
        const uint64_t* row_neg = w.neg_masks.data() + r * words;

        int64_t score = xnor_popcount_row(sign_mask.data(), row_pos, row_neg, words);

        // Convert integer score back to float scale
        // score = Σ sign(x_i) * w_i ∈ [-cols, cols]
        // Actual dot product ≈ score × E[|x|]
        // Note: alpha (weight-side scale) is stored but NOT multiplied here because
        // training forward is x @ w_ternary (w_ternary ∈ {-1,0,+1}), not alpha * (x @ w_ternary).
        out[r] = (float)score * x_absmean;
    }
}

// Dequantize one row of 2-bit packed ternary → float array
// Unpacks 4 weights per byte (MSB first) into temp_w[0..cols-1].
// Each weight becomes -1.0f, 0.0f, or +1.0f.
// Encoding: 00=-1, 01=0, 10=+1
static inline void dequantize_row(
    const uint8_t* packed, int row_offset, int cols, float* temp_w
) {
    int base = row_offset;
    int c = 0;

    // Process 4 weights per byte
    int num_bytes = (cols + 3) / 4;
    for (int b = 0; b < num_bytes; b++) {
        uint8_t byte = packed[base + b];
        int remaining = (cols - c < 4) ? (cols - c) : 4;

        for (int i = 0; i < remaining; i++) {
            int shift = 6 - i * 2;
            uint8_t code = (byte >> shift) & 0b11;
            // 00=-1, 01=0, 10=+1
            temp_w[c] = (code == 0b00) ? -1.0f : (code == 0b10) ? 1.0f : 0.0f;
            c++;
        }
    }
}

// SIMD dot product: sum(x[i] * w[i]) for i in [0, cols)
// Detection priority: AVX10 → AVX-512 → AVX2+FMA → scalar
#if defined(__AVX10_1__) || defined(__AVX10__)
static inline float dot_product_simd(const float* a, const float* b, int n) {
    __m512 vsum = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        vsum = _mm512_fmadd_ps(va, vb, vsum);
    }
    __m256 hi256 = _mm512_extractf32x8_ps(vsum, 1);
    __m256 lo256 = _mm512_castps512_ps256(vsum);
    __m256 sum256 = _mm256_add_ps(lo256, hi256);
    __m128 hi128 = _mm256_extractf128_ps(sum256, 1);
    __m128 lo128 = _mm256_castps256_ps128(sum256);
    __m128 s = _mm_add_ps(lo128, hi128);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}
#elif defined(__AVX512F__)
static inline float dot_product_simd(const float* a, const float* b, int n) {
    __m512 vsum = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        vsum = _mm512_fmadd_ps(va, vb, vsum);
    }
    __m256 hi256 = _mm512_extractf32x8_ps(vsum, 1);
    __m256 lo256 = _mm512_castps512_ps256(vsum);
    __m256 sum256 = _mm256_add_ps(lo256, hi256);
    __m128 hi128 = _mm256_extractf128_ps(sum256, 1);
    __m128 lo128 = _mm256_castps256_ps128(sum256);
    __m128 s = _mm_add_ps(lo128, hi128);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}
#elif defined(__AVX2__)
static inline float dot_product_simd(const float* a, const float* b, int n) {
    __m256 vsum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        vsum = _mm256_fmadd_ps(va, vb, vsum);
    }
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}
#else
static inline float dot_product_simd(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}
#endif

// Dequantize-On-The-Fly ternary matmul
// Exact quality: dequantize 2-bit → {-1,0,+1} float, then SIMD dot product.
// hidden_dim=128 → AVX-512 needs 8 FMA per row, AVX2 needs 16 FMA per row.
static void ternary_matmul_dequant(
    const float* x, const uint8_t* packed, float* out, int rows, int cols
) {
    // Stack buffer for dequantized weights (safe up to cols=512)
    float temp_w[512];

    for (int r = 0; r < rows; r++) {
        // Step 1: Dequantize this row (fast bit unpack)
        dequantize_row(packed, r * cols / 4, cols, temp_w);

        // Step 2: SIMD dot product (x · temp_w)
        out[r] = dot_product_simd(x, temp_w, cols);
    }
}

// Precompute: dequantize entire matrix into float array
static void precompute_floats(TernaryWeightXNOR& w) {
    w.floats.resize(w.rows * w.cols);
    for (int r = 0; r < w.rows; r++) {
        dequantize_row(w.packed.data(), r * w.cols / 4, w.cols, w.floats.data() + r * w.cols);
    }
}

// Fast matmul using precomputed float weights
// Best performance: dequantize once at load, then pure SIMD dot products.
static void ternary_matmul_precomputed(
    const float* x, const TernaryWeightXNOR& w, float* out
) {
    for (int r = 0; r < w.rows; r++) {
        out[r] = dot_product_simd(x, w.floats.data() + r * w.cols, w.cols);
    }
}

// Dispatch
// Exact quality + SIMD speed via precomputed dequantized weights.
static void ternary_matmul(
    const float* x,
    const TernaryWeightXNOR& w,
    float* out,
    float x_absmean
) {
    ternary_matmul_precomputed(x, w, out);
}

// Prefetch helper
#ifdef _MSC_VER
#define TETRA_PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
#define TETRA_PREFETCH(addr) __builtin_prefetch(addr, 0, 3)
#endif

// Decode ternary matmul: LUT unpack + SIMD dot product
// Processes 256 weights (64 bytes = 1 cache line) per iteration.
// LUT eliminates bit shifts: single table lookup per byte → 4 floats.
// Prefetches next cache line while computing current one.
//
// Memory layout per iteration:
//   packed:  64 bytes  (256 weights × 2 bits) — 1 cache line, sequential read
//   unpacked: 256 floats × 4 bytes = 1 KB     — fits in L1
//   activation x[c..c+256]: 1 KB               — stays in L1 across iterations
static void ternary_matmul_decode(
    const float* x, const uint8_t* packed, float* out, int rows, int cols
) {
    init_byte_lut();

    alignas(64) float chunk[256];
    const int CHUNK = 256;
    const int CHUNK_BYTES = CHUNK / 4;  // 64 bytes = 1 cache line

    for (int r = 0; r < rows; r++) {
        const uint8_t* row = packed + (r * cols) / 4;
        float sum = 0.0f;
        int c = 0;

        // Prefetch first cache line of this row
        TETRA_PREFETCH(row);

        // Main loop: 256 weights per iteration
        // Each iteration reads exactly 1 cache line of packed data.
        for (; c + CHUNK <= cols; c += CHUNK) {
            const uint8_t* src = row + c / 4;

            // Prefetch next cache line while we unpack current
            TETRA_PREFETCH(src + CHUNK_BYTES);

            // Unpack: 64 bytes → 256 floats via LUT (no bit shifts)
            // 16 unrolled iterations × 4 weights = 64 weights per inner loop
            // 4 inner loops × 64 = 256 weights total
            #pragma loop(hint_parallel(4))
            for (int j = 0; j < CHUNK_BYTES; j += 4) {
                // Read 4 bytes at once (32-bit load, single cache access)
                uint32_t four;
                memcpy(&four, src + j, 4);

                // Extract 4 bytes and lookup (compiler-friendly: constant shifts)
                const ByteDecoded& d0 = BYTE_LUT[four & 0xFF];
                const ByteDecoded& d1 = BYTE_LUT[(four >> 8) & 0xFF];
                const ByteDecoded& d2 = BYTE_LUT[(four >> 16) & 0xFF];
                const ByteDecoded& d3 = BYTE_LUT[four >> 24];

                int base = j * 4;
                chunk[base]    = d0.v[0]; chunk[base+1]  = d0.v[1];
                chunk[base+2]  = d0.v[2]; chunk[base+3]  = d0.v[3];
                chunk[base+4]  = d1.v[0]; chunk[base+5]  = d1.v[1];
                chunk[base+6]  = d1.v[2]; chunk[base+7]  = d1.v[3];
                chunk[base+8]  = d2.v[0]; chunk[base+9]  = d2.v[1];
                chunk[base+10] = d2.v[2]; chunk[base+11] = d2.v[3];
                chunk[base+12] = d3.v[0]; chunk[base+13] = d3.v[1];
                chunk[base+14] = d3.v[2]; chunk[base+15] = d3.v[3];
            }

            // SIMD dot product on unpacked chunk
            sum += dot_product_simd(x + c, chunk, CHUNK);
        }

        // Remainder: handle remaining weights
        if (c < cols) {
            int rem = cols - c;
            int rem_bytes = (rem + 3) / 4;
            const uint8_t* src = row + c / 4;

            for (int j = 0; j < rem_bytes; j++) {
                const ByteDecoded& d = BYTE_LUT[src[j]];
                int base = j * 4;
                for (int k = 0; k < 4 && base + k < rem; k++) {
                    chunk[base + k] = d.v[k];
                }
            }
            sum += dot_product_simd(x + c, chunk, rem);
        }

        out[r] = sum;
    }
}

// Decode FP32 matmul with prefetch
// Used for LM head (vocab projection) during single-token decode.
static void matmul_fp32_decode(const float* x, const float* w, float* out,
                                int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        // Prefetch 3 rows ahead (≈3 × cols × 4 bytes)
        if (r + 3 < rows) {
            TETRA_PREFETCH(w + (r + 3) * cols);
        }
        out[r] = dot_product_simd(x, w + r * cols, cols);
    }
}

// Precomputed decode: SIMD dot product + prefetch
// Uses precomputed floats (dequantized at load time) with prefetching.
// Fastest path when weights fit in L2/L3 cache.
static void ternary_matmul_precomputed_decode(
    const float* x, const TernaryWeightXNOR& w, float* out
) {
    const int rows = w.rows;
    const int cols = w.cols;
    const float* data = w.floats.data();

    for (int r = 0; r < rows; r++) {
        // Prefetch next row (2 rows ahead to hide memory latency)
        if (r + 2 < rows) {
            TETRA_PREFETCH(data + (r + 2) * cols);
        }
        out[r] = dot_product_simd(x, data + r * cols, cols);
    }
}

// Decode dispatch
// Default: precomputed with prefetch (fastest for current model sizes).
// LUT path available for memory-constrained / very large models.
static void ternary_matmul_auto(
    const float* x, const TernaryWeightXNOR& w, float* out,
    float x_absmean, bool decode
) {
    if (decode) {
        ternary_matmul_precomputed_decode(x, w, out);
    } else {
        ternary_matmul_precomputed(x, w, out);
    }
}

// FP32 matmul (embeddings, norms)
static void matmul_fp32(const float* x, const float* w, float* out,
                         int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) sum += x[c] * w[r * cols + c];
        out[r] = sum;
    }
}

// RMSNorm
static void rmsnorm(float* x, const float* weight, int dim, float eps=1e-6f) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float rms = sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) x[i] = (x[i] / rms) * weight[i];
}

// SiLU
static float silu(float x) { return x / (1.0f + expf(-x)); }

// Softmax
static void softmax(float* x, int n) {
    float mx = *std::max_element(x, x + n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// Mean absolute value (for XNOR scale factor)
static float absmean(const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += fabsf(x[i]);
    return sum / n;
}

// Model
struct Model {
    ModelHeader header;
    std::unordered_map<std::string, TernaryWeightXNOR> ternary_weights;
    std::unordered_map<std::string, FP32Weight> fp32_weights;

    const TernaryWeightXNOR& tw(const std::string& name) const {
        return ternary_weights.at(name);
    }
    const FP32Weight& fw(const std::string& name) const {
        return fp32_weights.at(name);
    }
    const float* fw_ptr(const std::string& name) const {
        return fp32_weights.at(name).data.data();
    }
    int head_dim() const { return header.hidden_dim / header.num_heads; }
};

static Model load_model(const char* path) {
    Model model;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }

    uint8_t header_buf[64];
    fread(header_buf, 1, 64, f);
    auto& h = model.header;
    memcpy(h.magic, header_buf, 4);
    memcpy(&h.version,        header_buf + 4,  4);
    memcpy(&h.vocab_size,     header_buf + 8,  4);
    memcpy(&h.hidden_dim,     header_buf + 12, 4);
    memcpy(&h.num_layers,     header_buf + 16, 4);
    memcpy(&h.num_heads,      header_buf + 20, 4);
    memcpy(&h.ffn_dim,        header_buf + 24, 4);
    memcpy(&h.max_seq_len,    header_buf + 28, 4);
    memcpy(&h.ternary_params, header_buf + 32, 8);
    memcpy(&h.fp32_params,    header_buf + 40, 8);

    fprintf(stderr, "Tetra: %d layers, hidden=%d, heads=%d, ffn=%d, vocab=%d, seq=%d\n",
            h.num_layers, h.hidden_dim, h.num_heads, h.ffn_dim, h.vocab_size, h.max_seq_len);

    // Read ternary weights → convert to XNOR bitmasks
    for (uint32_t layer = 0; layer < h.num_layers; layer++) {
        for (int t = 0; t < 7; t++) {
            uint32_t name_len;
            fread(&name_len, 4, 1, f);
            std::string name(name_len, '\0');
            fread(&name[0], 1, name_len, f);

            uint16_t rows, cols;
            fread(&rows, 2, 1, f);
            fread(&cols, 2, 1, f);

            // v2: read per-matrix alpha scale factor
            float alpha = 1.0f;
            if (h.version >= 2) {
                fread(&alpha, 4, 1, f);
            }

            int packed_size = (rows * cols + 3) / 4;
            std::vector<uint8_t> packed(packed_size);
            fread(packed.data(), 1, packed_size, f);

            // Convert to XNOR bitmasks (one-time cost at load)
            TernaryWeightXNOR w = convert_to_xnor(packed.data(), rows, cols);
            w.alpha = alpha;
            w.packed = std::move(packed);
            precompute_floats(w);
            model.ternary_weights[name] = std::move(w);
        }
    }

    // Read FP32 weights
    while (true) {
        uint32_t name_len;
        if (fread(&name_len, 4, 1, f) != 1) break;
        if (name_len > 1024) break;

        std::string name(name_len, '\0');
        fread(&name[0], 1, name_len, f);

        uint8_t ndim;
        fread(&ndim, 1, 1, f);

        uint32_t dims[4] = {1,1,1,1};
        fread(dims, 4, 4, f);

        int n_elements = 1;
        std::vector<int> shape(ndim);
        for (int i = 0; i < ndim; i++) { shape[i] = dims[i]; n_elements *= dims[i]; }

        FP32Weight fw;
        fw.shape = shape;
        fw.data.resize(n_elements);
        fread(fw.data.data(), 4, n_elements, f);
        model.fp32_weights[name] = std::move(fw);
    }

    fclose(f);
    fprintf(stderr, "Loaded %zu ternary + %zu fp32 tensors\n",
            model.ternary_weights.size(), model.fp32_weights.size());
    return model;
}

// KV Cache
struct KVCache {
    std::vector<std::vector<float>> k_cache;
    std::vector<std::vector<float>> v_cache;
    int pos = 0;

    void init(int num_layers, int max_seq_len, int dim) {
        k_cache.resize(num_layers, std::vector<float>(max_seq_len * dim, 0.0f));
        v_cache.resize(num_layers, std::vector<float>(max_seq_len * dim, 0.0f));
        pos = 0;
    }
};

// Forward pass
static std::vector<float> forward(
    const Model& model,
    const std::vector<int>& tokens,
    KVCache& cache
) {
    int H  = model.header.hidden_dim;
    int L  = model.header.num_layers;
    int NH = model.header.num_heads;
    int HD = model.head_dim();
    int FFN = model.header.ffn_dim;
    int V  = model.header.vocab_size;
    int seq_len = (int)tokens.size();
    bool decode = (seq_len == 1);  // Single-token = decode phase

    std::vector<float> x(H), q(H), k(H), v(H);
    std::vector<float> attn_scores(model.header.max_seq_len);
    std::vector<float> attn_out(H);
    std::vector<float> gate(FFN), up(FFN), hidden(FFN), ffn_out(H);

    const float* tok_emb = model.fw_ptr("token_embedding.weight");
    const float* pos_emb = model.fw_ptr("pos_embedding.weight");

    // Token + Position Embedding
    // Use cache.pos for position (not tokens.size()-1) because after prefill,
    // cache.pos may exceed the tokens vector size.
    int pos = cache.pos;
    for (int i = 0; i < H; i++) {
        x[i] = tok_emb[tokens.back() * H + i]
              + pos_emb[pos * H + i];
    }

    // Transformer layers
    for (int l = 0; l < L; l++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "layers.%d.", l);

        // Attention (pre-norm)
        std::vector<float> normed = x;
        rmsnorm(normed.data(), model.fw_ptr(std::string(prefix) + "attn_norm.weight"), H);

        // Compute absmean of normed input (scale factor for XNOR)
        float x_scale = absmean(normed.data(), H);

        std::string q_name = std::string(prefix) + "attn.q_proj.latent_weights";
        std::string k_name = std::string(prefix) + "attn.k_proj.latent_weights";
        std::string v_name = std::string(prefix) + "attn.v_proj.latent_weights";
        std::string o_name = std::string(prefix) + "attn.o_proj.latent_weights";

        // XNOR+popcount ternary projections
        ternary_matmul_auto(normed.data(), model.tw(q_name), q.data(), x_scale, decode);
        ternary_matmul_auto(normed.data(), model.tw(k_name), k.data(), x_scale, decode);
        ternary_matmul_auto(normed.data(), model.tw(v_name), v.data(), x_scale, decode);

        // Store K,V in cache
        for (int i = 0; i < H; i++) {
            cache.k_cache[l][pos * H + i] = k[i];
            cache.v_cache[l][pos * H + i] = v[i];
        }

        // Multi-head attention (causal)
        for (int head = 0; head < NH; head++) {
            float scale = 1.0f / sqrtf((float)HD);
            int actual_len = pos + 1;

            // Attention scores: Q · K for each time step
            const float* q_head = q.data() + head * HD;
            for (int t = 0; t < actual_len; t++) {
                attn_scores[t] = dot_product_simd(
                    q_head,
                    cache.k_cache[l].data() + t * H + head * HD,
                    HD
                ) * scale;
            }
            softmax(attn_scores.data(), actual_len);

            // Attention output: weighted sum of V
            for (int d = 0; d < HD; d++) {
                float sum = 0.0f;
                for (int t = 0; t < actual_len; t++) {
                    sum += attn_scores[t] * cache.v_cache[l][t * H + head * HD + d];
                }
                attn_out[head * HD + d] = sum;
            }
        }

        // Output projection
        std::vector<float> proj_out(H);
        float o_scale = absmean(attn_out.data(), H);
        ternary_matmul_auto(attn_out.data(), model.tw(o_name), proj_out.data(), o_scale, decode);

        for (int i = 0; i < H; i++) x[i] += proj_out[i];

        // FFN (pre-norm)
        std::vector<float> ffn_normed = x;
        rmsnorm(ffn_normed.data(), model.fw_ptr(std::string(prefix) + "ffn_norm.weight"), H);

        float ffn_scale = absmean(ffn_normed.data(), H);

        std::string gate_name = std::string(prefix) + "ffn.gate_proj.latent_weights";
        std::string up_name   = std::string(prefix) + "ffn.up_proj.latent_weights";
        std::string down_name = std::string(prefix) + "ffn.down_proj.latent_weights";

        // SwiGLU: gate and up projections
        ternary_matmul_auto(ffn_normed.data(), model.tw(gate_name), gate.data(), ffn_scale, decode);
        ternary_matmul_auto(ffn_normed.data(), model.tw(up_name), up.data(), ffn_scale, decode);

        // SiLU(gate) * up
        for (int i = 0; i < FFN; i++) hidden[i] = silu(gate[i]) * up[i];

        // Down projection
        float h_scale = absmean(hidden.data(), FFN);
        ternary_matmul_auto(hidden.data(), model.tw(down_name), ffn_out.data(), h_scale, decode);

        for (int i = 0; i < H; i++) x[i] += ffn_out[i];
    }

    // Final norm + LM head (tied to token_embedding)
    rmsnorm(x.data(), model.fw_ptr("norm.weight"), H);

    std::vector<float> logits(V);
    if (decode) {
        // Decode: SIMD dot product with prefetch for each vocab entry
        matmul_fp32_decode(x.data(), tok_emb, logits.data(), V, H);
    } else {
        // Prefill: scalar (ok for small seq)
        for (int vi = 0; vi < V; vi++) {
            float dot = 0.0f;
            for (int i = 0; i < H; i++) dot += x[i] * tok_emb[vi * H + i];
            logits[vi] = dot;
        }
    }

    cache.pos++;
    return logits;
}

// Sampling
// Top-k: keep only top k tokens, then top-p: keep smallest set with cumprob >= p.
static int sample(const std::vector<float>& logits, float temperature, int top_k, float top_p) {
    int n = (int)logits.size();
    float mx = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probs(n);
    for (int i = 0; i < n; i++) probs[i] = expf((logits[i] - mx) / temperature);

    // Top-k filtering: zero out tokens outside top k
    if (top_k > 0 && top_k < n) {
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
            [&](int a, int b) { return probs[a] > probs[b]; });
        float z = 0.0f;
        for (int i = 0; i < top_k; i++) z += probs[indices[i]];
        // Save top-k values before zeroing
        std::vector<float> top_vals(top_k);
        for (int i = 0; i < top_k; i++) top_vals[i] = probs[indices[i]];
        for (int i = 0; i < n; i++) probs[i] = 0.0f;
        for (int i = 0; i < top_k; i++) probs[indices[i]] = z > 0 ? top_vals[i] / z : 1.0f / top_k;
    }

    // Top-p (nucleus) filtering: keep smallest set with cumprob >= top_p
    if (top_p < 1.0f && top_p > 0.0f) {
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
            [&](int a, int b) { return probs[a] > probs[b]; });
        float cum = 0.0f;
        int cutoff = n;
        for (int i = 0; i < n; i++) {
            cum += probs[indices[i]];
            if (cum >= top_p) { cutoff = i + 1; break; }
        }
        float z = 0.0f;
        for (int i = 0; i < cutoff; i++) z += probs[indices[i]];
        // Save values before zeroing
        std::vector<float> top_vals(cutoff);
        for (int i = 0; i < cutoff; i++) top_vals[i] = probs[indices[i]];
        for (int i = 0; i < n; i++) probs[i] = 0.0f;
        for (int i = 0; i < cutoff; i++) probs[indices[i]] = z > 0 ? top_vals[i] / z : 1.0f / cutoff;
    }

    // Renormalize and sample
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += probs[i];
    if (sum > 0) for (int i = 0; i < n; i++) probs[i] /= sum;

    float r = (float)rand() / RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < n; i++) { cum += probs[i]; if (r < cum) return i; }
    return n - 1;
}

}  // namespace tetra
