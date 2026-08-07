#pragma once
// Tetra - C++ inference engine
// Build: build.bat [avx2|avx10|avx512] (scalar if no arg)
// Binary Format v2:
//   Header (64B): magic, version, dims, param counts
//   Ternary weights: name, shape, alpha, 2-bit packed data
//   FP32 weights: embeddings, norms (lm_head tied to embedding)

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <random>
#include <numeric>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <immintrin.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#define TETRA_HAS_NEON 1
#endif

namespace tetra {

static constexpr int TETRA_MAX_COLS = 8192;

struct TernaryWeightXNOR {
    std::vector<uint8_t> packed;
    std::vector<float> floats;
    int rows, cols;
    int group_size;            // 0 = scalar/per-channel, >0 = per-group block
    float alpha;               // scalar alpha (default 1.0)
    std::vector<float> alphas; // flat array: per-group = rows*num_groups, per-channel = rows, empty = scalar
    std::vector<float> accumulator; // FP32 learning state (rows*cols), v6 only
    bool is_v7 = false;        // v7 entry: code 11 = ±2 outlier, sign in outlier_blob
    std::vector<uint8_t> outlier_blob; // dense sign bits (MSB-first, 1=positive), only
                                       // ceil(n_outliers/8) bytes, dense-first in scan order
};

// SIMD dot product: sum(x[i] * w[i]) for i in [0, cols)
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

// Prefetch helper
#ifdef _MSC_VER
#define TETRA_PREFETCH(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#else
#define TETRA_PREFETCH(addr) __builtin_prefetch(addr, 0, 3)
#endif

// Dequantize one row of 2-bit packed ternary -> float array.
// blob: dense sign bits for code-11 outliers (dense-first in row-major scan
// order, MSB-first, 1=positive). ``outlier_idx`` is the running counter of
// code-11 weights already scanned across all rows. blob == nullptr -> v6.
static inline void dequantize_row(const uint8_t* packed, int row_offset, int cols, float* out,
                                  const uint8_t* blob, size_t& outlier_idx) {
    static const float lut[4] = {-1.0f, 0.0f, 1.0f, 0.0f};
    int c = 0, num_bytes = (cols + 3) / 4;
    for (int b = 0; b < num_bytes; b++) {
        uint8_t byte = packed[row_offset + b];
        int rem = (cols - c < 4) ? (cols - c) : 4;
        for (int i = 0; i < rem; i++, c++) {
            int code = (byte >> (6 - i * 2)) & 3;
            if (code == 3) {
                if (blob) {
                    int bit = (int)((blob[outlier_idx >> 3] >> (7 - (outlier_idx & 7))) & 1);
                    out[c] = bit ? 2.0f : -2.0f;
                } else {
                    out[c] = 0.0f;
                }
                outlier_idx++;
            } else {
                out[c] = lut[code];
            }
        }
    }
}

// Precompute: dequantize all weights to float at load time
static void precompute_floats(TernaryWeightXNOR& w) {
    int row_bytes = (w.cols + 3) / 4;
    w.floats.resize((size_t)w.rows * w.cols);
    size_t outlier_idx = 0;
    for (int r = 0; r < w.rows; r++) {
        dequantize_row(w.packed.data(), r * row_bytes, w.cols,
                       w.floats.data() + (size_t)r * w.cols,
                       w.outlier_blob.empty() ? nullptr : w.outlier_blob.data(), outlier_idx);
    }
}

// Precomputed float matmul with prefetch + alpha scaling
static void ternary_matmul_precomputed(
    const float* x, const TernaryWeightXNOR& w, float* out
) {
    const float alpha0 = w.alpha;
    const float* data = w.floats.data();
    if (w.group_size > 0) {
        int num_groups = (w.cols + w.group_size - 1) / w.group_size;
        for (int r = 0; r < w.rows; r++) {
            float sum = 0;
            for (int g = 0; g < num_groups; g++) {
                int offset = g * w.group_size;
                int gs = (std::min)(w.group_size, w.cols - offset);
                sum += w.alphas[r * num_groups + g] * dot_product_simd(x + offset, data + r * w.cols + offset, gs);
            }
            out[r] = sum;
        }
    } else if (!w.alphas.empty()) {
        for (int r = 0; r < w.rows; r++) {
            out[r] = dot_product_simd(x, data + r * w.cols, w.cols) * w.alphas[r];
        }
    } else {
        for (int r = 0; r < w.rows; r++) {
            out[r] = dot_product_simd(x, data + r * w.cols, w.cols) * alpha0;
        }
    }
}

// Precomputed decode path: same but with row prefetch
static void ternary_matmul_precomputed_decode(
    const float* x, const TernaryWeightXNOR& w, float* out
) {
    const int rows = w.rows;
    const int cols = w.cols;
    const float alpha0 = w.alpha;
    const float* data = w.floats.data();
    if (w.group_size > 0) {
        int num_groups = (cols + w.group_size - 1) / w.group_size;
        for (int r = 0; r < rows; r++) {
            if (r + 2 < rows) TETRA_PREFETCH(data + (r + 2) * cols);
            float sum = 0;
            for (int g = 0; g < num_groups; g++) {
                int offset = g * w.group_size;
                int gs = (std::min)(w.group_size, cols - offset);
                sum += w.alphas[r * num_groups + g] * dot_product_simd(x + offset, data + r * cols + offset, gs);
            }
            out[r] = sum;
        }
    } else if (!w.alphas.empty()) {
        for (int r = 0; r < rows; r++) {
            if (r + 2 < rows) TETRA_PREFETCH(data + (r + 2) * cols);
            out[r] = dot_product_simd(x, data + r * cols, cols) * w.alphas[r];
        }
    } else {
        for (int r = 0; r < rows; r++) {
            if (r + 2 < rows) TETRA_PREFETCH(data + (r + 2) * cols);
            out[r] = dot_product_simd(x, data + r * cols, cols) * alpha0;
        }
    }
}

// Dispatch
static void ternary_matmul_auto(
    const float* x, const TernaryWeightXNOR& w, float* out,
    float x_absmean, bool decode
) {
    (void)x_absmean;
    if (decode) ternary_matmul_precomputed_decode(x, w, out);
    else        ternary_matmul_precomputed(x, w, out);
}

struct FP32Weight {
    std::vector<float> data;
    std::vector<int> shape;
    std::vector<int8_t> int8_data;  // raw INT8 (for LM head speed)
    float int8_scale = 0.0f;
};

struct ModelHeader {
    char magic[4];
    uint32_t version, vocab_size, hidden_dim, num_layers, num_heads;
    uint32_t ffn_dim, max_seq_len;
    uint64_t ternary_params, fp32_params;
    // v5 fields
    uint16_t flags;         // bit0=is_mla, bit1=int8_embeddings
    uint16_t kv_latent_dim;
    uint16_t rope_per_head;
    uint16_t group_size;
};

// Decode FP32 matmul with prefetch
static void matmul_fp32_decode(const float* x, const float* w, float* out,
                                int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        if (r + 3 < rows)
            TETRA_PREFETCH(w + (r + 3) * cols);
        out[r] = dot_product_simd(x, w + r * cols, cols);
    }
}

// INT8 matmul for LM head: reads 4x less memory bandwidth
static void matmul_int8_decode(const float* x, const int8_t* w, float* out,
                                int rows, int cols, float scale) {
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        int c = 0;
#if defined(__AVX2__)
        __m256 vsum = _mm256_setzero_ps();
        for (; c + 8 <= cols; c += 8) {
            __m256i vi8 = _mm256_loadu_si256((const __m256i*)(w + r * cols + c));
            __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm256_castsi256_si128(vi8)));
            __m256 vx = _mm256_loadu_ps(x + c);
            vsum = _mm256_fmadd_ps(vx, vf, vsum);
        }
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        sum = _mm_cvtss_f32(s);
#endif
        for (; c < cols; c++)
            sum += x[c] * (float)w[r * cols + c];
        out[r] = sum * scale;
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
    if (sum > 0.0f) for (int i = 0; i < n; i++) x[i] /= sum;
}

// Mean absolute value (for XNOR scale factor)
static float absmean(const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += fabsf(x[i]);
    return sum / n;
}

// === Rotary Position Embedding (RoPE) ===
static std::vector<float> precompute_rope_freqs(int rope_dim, int max_seq_len) {
    int num_pairs = rope_dim / 2;
    std::vector<float> freqs(max_seq_len * num_pairs * 2, 0.0f);
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < num_pairs; i++) {
            float freq = 1.0f / powf(10000.0f, (float)(i * 2) / (float)rope_dim);
            float angle = (float)pos * freq;
            int idx = (pos * num_pairs + i) * 2;
            freqs[idx + 0] = cosf(angle);
            freqs[idx + 1] = sinf(angle);
        }
    }
    return freqs;
}

static void apply_rope(float* x, int rope_per_head, int pos, const float* freqs) {
    int num_pairs = rope_per_head / 2;
    const float* f = freqs + (pos * num_pairs) * 2;
    for (int i = 0; i < num_pairs; i++) {
        float a = x[i * 2], b = x[i * 2 + 1];
        float c = f[i * 2], s = f[i * 2 + 1];
        x[i * 2 + 0] = a * c - b * s;
        x[i * 2 + 1] = a * s + b * c;
    }
}

// MappedFile: cross-platform memory-mapped file
struct MappedFile {
#ifdef _WIN32
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMap = nullptr;
#else
    int fd = -1;
#endif
    const uint8_t* data = nullptr;
    size_t size = 0;

    bool open(const char* path) {
#ifdef _WIN32
        hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) return false;
        LARGE_INTEGER li;
        GetFileSizeEx(hFile, &li);
        size = (size_t)li.QuadPart;
        hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!hMap) { CloseHandle(hFile); return false; }
        data = (const uint8_t*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!data) { CloseHandle(hMap); CloseHandle(hFile); return false; }
#else
        fd = ::open(path, O_RDONLY);
        if (fd < 0) return false;
        struct stat st;
        fstat(fd, &st);
        size = st.st_size;
        data = (const uint8_t*)mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
        if (data == MAP_FAILED) { ::close(fd); return false; }
#endif
        return true;
    }

    void close() {
#ifdef _WIN32
        if (data) UnmapViewOfFile(data);
        if (hMap) CloseHandle(hMap);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#else
        if (data) munmap((void*)data, size);
        if (fd >= 0) ::close(fd);
#endif
        data = nullptr;
    }

    ~MappedFile() { close(); }
    MappedFile() = default;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept { *this = std::move(other); }
    MappedFile& operator=(MappedFile&& other) noexcept {
        close();
        data = other.data; other.data = nullptr;
        size = other.size; other.size = 0;
#ifdef _WIN32
        hFile = other.hFile; other.hFile = INVALID_HANDLE_VALUE;
        hMap = other.hMap; other.hMap = NULL;
#else
        fd = other.fd; other.fd = -1;
#endif
        return *this;
    }
};

// Cursor-based reader over mmap region
struct Reader {
    const uint8_t* pos;
    const uint8_t* end;
    Reader(const uint8_t* p, const uint8_t* e) : pos(p), end(e) {}
    template<typename T> void read(T& val) {
        if (pos + sizeof(T) > end) { fprintf(stderr, "Read past end\n"); exit(1); }
        memcpy(&val, pos, sizeof(T)); pos += sizeof(T);
    }
    void read_bytes(void* buf, size_t n) {
        if (pos + n > end) { fprintf(stderr, "Read past end\n"); exit(1); }
        memcpy(buf, pos, n); pos += n;
    }
    std::string read_str(size_t len) {
        std::string s(len, '\0');
        read_bytes(&s[0], len);
        return s;
    }
    void skip(size_t n) { pos += n; }
};

// Model
struct Model {
    ModelHeader header;
    MappedFile mapped;
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
    const int8_t* int8_ptr(const std::string& name) const {
        return fp32_weights.at(name).int8_data.data();
    }
    float int8_scale_val(const std::string& name) const {
        return fp32_weights.at(name).int8_scale;
    }
    int head_dim() const { return header.hidden_dim / header.num_heads; }

    // MLA fields: prefer header v5, fallback to tensor name inference
    bool is_mla = (header.version >= 5) ? (header.flags & 1) : false;
    int kv_latent_dim = (header.version >= 5) ? header.kv_latent_dim : 0;
    int rope_per_head = (header.version >= 5) ? header.rope_per_head : 0;
    int rope_dim = 0;
    std::vector<float> freqs_cis;

    // Self-learning config (v6, from metadata JSON)
    int sl_rule = 0;              // 0=c predictive coding, 1=b FF, 2=p target PC
    float sl_threshold = 20.0f;
    float sl_acc_decay = 0.99f;
    int sl_flip_every_n = 5;
    float sl_logit_scale = 0.0625f;
    float sl_lr_embedding = 1e-4f;
    float sl_wd_embedding = 0.1f;
    int sl_block_size = 128;
    int sl_toggle = 0;            // anti-stiction toggle kicks
    float sl_outlier_mult = 3.0f; // v7 promotion threshold multiplier
    bool sl_enabled = false;      // true when a v6 self-learning model was loaded
};

// ──────────────────────────────────────────────────────────────
// Minimal JSON field extractors (for the metadata section)
// ──────────────────────────────────────────────────────────────

static std::string json_get_str(const std::string& j, const std::string& key) {
    std::string pat = "\"" + key + "\"";
    size_t p = j.find(pat);
    if (p == std::string::npos) return "";
    p = j.find(':', p + pat.size());
    if (p == std::string::npos) return "";
    while (p + 1 < j.size() && (j[p+1] == ' ' || j[p+1] == '\t' || j[p+1] == '\n' || j[p+1] == '\r')) p++;
    if (p + 1 >= j.size() || j[p+1] != '"') return "";
    size_t q = p + 2;
    size_t end = j.find('"', q);
    if (end == std::string::npos) return "";
    return j.substr(q, end - q);
}

static double json_get_num(const std::string& j, const std::string& key, double dflt) {
    std::string pat = "\"" + key + "\"";
    size_t p = j.find(pat);
    if (p == std::string::npos) return dflt;
    p = j.find(':', p + pat.size());
    if (p == std::string::npos) return dflt;
    p++;
    while (p < j.size() && (j[p] == ' ' || j[p] == '\t' || j[p] == '\n' || j[p] == '\r')) p++;
    size_t start = p;
    while (p < j.size() && (j[p] == '-' || j[p] == '+' || j[p] == '.' ||
                            (j[p] >= '0' && j[p] <= '9') || j[p] == 'e' || j[p] == 'E')) p++;
    if (p == start) return dflt;
    try { return std::stod(j.substr(start, p - start)); }
    catch (...) { return dflt; }
}

static Model load_model(const char* path) {
    Model model;
    if (!model.mapped.open(path)) {
        fprintf(stderr, "Cannot open %s\n", path); exit(1);
    }
    Reader r(model.mapped.data, model.mapped.data + model.mapped.size);

    uint8_t header_buf[64];
    r.read_bytes(header_buf, 64);
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

    // v5 fields (bytes 48-55)
    h.flags = 0; h.kv_latent_dim = 0; h.rope_per_head = 0; h.group_size = 0;
    if (h.version >= 5) {
        memcpy(&h.flags,         header_buf + 48, 2);
        memcpy(&h.kv_latent_dim, header_buf + 50, 2);
        memcpy(&h.rope_per_head, header_buf + 52, 2);
        memcpy(&h.group_size,    header_buf + 54, 2);
    }

    fprintf(stderr, "Tetra: %d layers, hidden=%d, heads=%d, ffn=%d, vocab=%d, seq=%d",
            h.num_layers, h.hidden_dim, h.num_heads, h.ffn_dim, h.vocab_size, h.max_seq_len);
    if (h.version >= 5 && (h.flags & 1)) {
        fprintf(stderr, " [MLA kv_lat=%d rope_per_head=%d]", h.kv_latent_dim, h.rope_per_head);
    }
    if (h.version >= 5 && h.group_size > 0) {
        fprintf(stderr, " [group=%d]", h.group_size);
    }
    fprintf(stderr, "\n");

    // Read ternary weights: read until we hit FP32 section
    // (ternary names end with ".latent_weights", FP32 names don't)
    int ternary_count = 0;
    while (r.pos < r.end - 4) {
        // Peek name length to detect FP32 section
        uint32_t peek_len;
        memcpy(&peek_len, r.pos, 4);
        if (peek_len > 1024 || peek_len == 0) break;

        const uint8_t* name_start = r.pos + 4;
        // Check if name ends with "latent_weights"
        bool is_ternary = false;
        if (peek_len > 13) {
            const char* name_cstr = (const char*)name_start;
            const char* suffix = "latent_weights";
            int slen = 14;
            is_ternary = (peek_len >= (uint32_t)slen &&
                         memcmp(name_cstr + peek_len - slen, suffix, slen) == 0);
        }

        if (!is_ternary) break;  // FP32 section starts

        uint32_t name_len;
        r.read(name_len);
        std::string name = r.read_str(name_len);

        uint16_t rows, cols;
        r.read(rows);
        r.read(cols);

        int group_size = 0;
        std::vector<float> alphas;
        if (h.version >= 4) {
            uint16_t gs, num_alphas;
            r.read(gs);
            r.read(num_alphas);
            group_size = gs;
            if (num_alphas > 0) {
                alphas.resize(num_alphas);
                r.read_bytes(alphas.data(), num_alphas * sizeof(float));
            }
        } else if (h.version >= 3) {
            uint16_t num_alphas;
            r.read(num_alphas);
            if (num_alphas > 0) {
                alphas.resize(num_alphas);
                r.read_bytes(alphas.data(), num_alphas * sizeof(float));
            }
        } else if (h.version >= 2) {
            float alpha;
            r.read(alpha);
            alphas.push_back(alpha);
        }

        int packed_size = (rows * cols + 3) / 4;
        std::vector<uint8_t> packed(packed_size);
        r.read_bytes(packed.data(), packed_size);

        TernaryWeightXNOR w;
        w.rows = rows;
        w.cols = cols;
        w.group_size = group_size;
        w.alpha = alphas.empty() ? 1.0f : alphas[0];
        w.alphas = std::move(alphas);
        w.packed = std::move(packed);

        if (h.version >= 6) {
            w.accumulator.resize((size_t)rows * cols, 0.0f);
            r.read_bytes(w.accumulator.data(), w.accumulator.size() * sizeof(float));
        }

        // v7: trailing uint32 outlier count + dense sign blob
        // (ceil(n_outliers/8) bytes, MSB-first, 1=positive, dense-first scan order).
        if (h.version >= 7) {
            uint32_t n_outliers = 0;
            r.read(n_outliers);
            w.is_v7 = true;
            size_t nb = ((size_t)n_outliers + 7) / 8;
            w.outlier_blob.resize(nb);
            if (nb) r.read_bytes(w.outlier_blob.data(), nb);
        }

        precompute_floats(w);
        model.ternary_weights[name] = std::move(w);
        ternary_count++;

        // Detect MLA from tensor names (fallback for v4 and earlier)
        if (name.find("kv_down_proj") != std::string::npos)
            model.is_mla = true;
    }

    // Extract MLA dimensions: prefer header, fallback to tensor shapes
    if (model.is_mla) {
        if (model.rope_per_head == 0 || model.kv_latent_dim == 0) {
            // Fallback: infer from tensor dimensions
            for (auto& [name, w] : model.ternary_weights) {
                if (name.find("kv_down_proj") != std::string::npos && model.kv_latent_dim == 0) {
                    model.kv_latent_dim = w.rows;
                }
                if (name.find("q_rope_proj") != std::string::npos && model.rope_per_head == 0) {
                    model.rope_dim = w.rows;
                    model.rope_per_head = w.rows / h.num_heads;
                }
            }
        } else {
            // Compute rope_dim from header values
            model.rope_dim = model.rope_per_head * h.num_heads;
        }
        model.freqs_cis = precompute_rope_freqs(model.rope_per_head, h.max_seq_len);
    }

    // Read FP32/INT8 weights
    while (r.pos < r.end - 4) {
        uint32_t name_len;
        r.read(name_len);
        if (name_len > 1024) break;

        std::string name = r.read_str(name_len);

        uint8_t ndim, dtype;
        r.read(ndim);
        r.read(dtype);

        uint32_t dims[4] = {1,1,1,1};
        r.read_bytes(dims, 16);

        int n_elements = 1;
        std::vector<int> shape(ndim);
        for (int i = 0; i < ndim; i++) { shape[i] = (int)dims[i]; n_elements *= dims[i]; }

        FP32Weight fw;
        fw.shape = shape;
        fw.data.resize(n_elements);

        if (dtype == 1) {  // INT8 -> dequantize to FP32, keep raw INT8 for LM head
            float scale;
            r.read(scale);
            fw.int8_scale = scale;
            const int8_t* src = (const int8_t*)r.pos;
            for (int i = 0; i < n_elements; i++)
                fw.data[i] = (float)src[i] * scale;
            fw.int8_data.assign(src, src + n_elements);
            r.skip(n_elements);
        } else {  // FP32
            r.read_bytes(fw.data.data(), 4 * n_elements);
        }
        model.fp32_weights[name] = std::move(fw);
    }

    fprintf(stderr, "Loaded %zu ternary + %zu fp32 tensors\n",
            model.ternary_weights.size(), model.fp32_weights.size());

    // v6: parse the metadata JSON section for the self-learning config.
    // Note: the fp32 loop already consumed the 4-byte "META" magic as a
    // (rejected) name-length field, so the magic sits at r.pos - 4.
    if (r.pos + 4 <= r.end && memcmp(r.pos - 4, "META", 4) == 0) {
        uint32_t meta_len;
        r.read(meta_len);
        std::string meta = (meta_len <= (uint32_t)(r.end - r.pos))
            ? r.read_str(meta_len) : "";
        std::string rule = json_get_str(meta, "sl_rule");
        if (rule == "c")      model.sl_rule = 0;
        else if (rule == "b") model.sl_rule = 1;
        else if (rule == "p") model.sl_rule = 2;
        else if (rule == "h") model.sl_rule = 3;
        else if (rule == "e") model.sl_rule = 4;
        model.sl_threshold   = (float)json_get_num(meta, "sl_threshold",   20.0);
        model.sl_acc_decay   = (float)json_get_num(meta, "sl_acc_decay",   0.99);
        model.sl_flip_every_n = (int)json_get_num(meta, "sl_flip_every_n", 5);
        model.sl_logit_scale = (float)json_get_num(meta, "sl_logit_scale", 0.0625);
        model.sl_lr_embedding = (float)json_get_num(meta, "sl_lr_embedding", 1e-4);
        model.sl_wd_embedding = (float)json_get_num(meta, "sl_wd_embedding", 0.1);
        model.sl_block_size  = (int)json_get_num(meta, "sl_block_size",    128);
        model.sl_toggle      = (int)json_get_num(meta, "sl_toggle",        0);
        model.sl_outlier_mult = (float)json_get_num(meta, "sl_outlier_mult", 3.0);
        model.sl_enabled = true;
        fprintf(stderr, "Self-learning: rule=%s thr=%.1f decay=%.3f flipEvery=%d toggle=%d scale=%.6f embLR=%.1e embWD=%.2f block=%d outlierMult=%.2f\n",
                rule.c_str(), model.sl_threshold, model.sl_acc_decay, model.sl_flip_every_n,
                model.sl_toggle, model.sl_logit_scale, model.sl_lr_embedding, model.sl_wd_embedding, model.sl_block_size, model.sl_outlier_mult);
    }
    return model;
}

// KV Cache
struct KVCache {
    std::vector<std::vector<float>> k_cache;
    std::vector<std::vector<float>> v_cache;
    // MLA latent + RoPE cache
    std::vector<std::vector<float>> latent_cache;
    std::vector<std::vector<float>> k_rope_cache;
    // MLA expanded K/V cache (avoids per-step reconstruction from latent)
    std::vector<std::vector<float>> k_full_cache;
    std::vector<std::vector<float>> v_full_cache;
    int pos = 0;

    void init(int num_layers, int max_seq_len, int dim,
              bool is_mla = false, int kv_latent_dim = 0, int rope_dim = 0) {
        if (is_mla) {
            latent_cache.resize(num_layers, std::vector<float>(max_seq_len * kv_latent_dim, 0.0f));
            k_rope_cache.resize(num_layers, std::vector<float>(max_seq_len * rope_dim, 0.0f));
            k_full_cache.resize(num_layers, std::vector<float>(max_seq_len * dim, 0.0f));
            v_full_cache.resize(num_layers, std::vector<float>(max_seq_len * dim, 0.0f));
        } else {
            k_cache.resize(num_layers, std::vector<float>(max_seq_len * dim, 0.0f));
            v_cache.resize(num_layers, std::vector<float>(max_seq_len * dim, 0.0f));
        }
        pos = 0;
    }

    void clear() {
        for (auto& v : k_cache) std::fill(v.begin(), v.end(), 0.0f);
        for (auto& v : v_cache) std::fill(v.begin(), v.end(), 0.0f);
        for (auto& v : latent_cache) std::fill(v.begin(), v.end(), 0.0f);
        for (auto& v : k_rope_cache) std::fill(v.begin(), v.end(), 0.0f);
        for (auto& v : k_full_cache) std::fill(v.begin(), v.end(), 0.0f);
        for (auto& v : v_full_cache) std::fill(v.begin(), v.end(), 0.0f);
        pos = 0;
    }
};

// ──────────────────────────────────────────────────────────────
// Activation capture (for the self-learning runtime)
// ──────────────────────────────────────────────────────────────

struct LinearCapture {
    std::string name;
    int rows, cols;
    std::vector<float> x;  // layer input  (cols)
    std::vector<float> y;  // layer output (rows)
};

struct Capture {
    std::vector<LinearCapture> layers;  // current position
    std::vector<float> h;               // final normed hidden (H)
};

static void cap_fill(Capture* cap, const std::string& name,
                     const float* x, int cols, const float* y, int rows) {
    if (!cap) return;
    LinearCapture lc;
    lc.name = name;
    lc.rows = rows;
    lc.cols = cols;
    lc.x.assign(x, x + cols);
    lc.y.assign(y, y + rows);
    cap->layers.push_back(std::move(lc));
}

// Forward pass - supports both decode (single token) and batch prefill (multiple tokens)
static std::vector<float> forward(
    const Model& model,
    const std::vector<int>& tokens,
    KVCache& cache,
    Capture* cap = nullptr
) {
    int H  = model.header.hidden_dim;
    int L  = model.header.num_layers;
    int NH = model.header.num_heads;
    int HD = model.head_dim();
    int FFN = model.header.ffn_dim;
    int V  = model.header.vocab_size;
    int seq_len = (int)tokens.size();
    bool decode = (seq_len == 1);

    const float* tok_emb = model.fw_ptr("token_embedding.weight");
    int pos = cache.pos;

    if (decode) {
        if (cap && model.is_mla) {
            fprintf(stderr, "Capture not supported for MLA models\n");
            cap = nullptr;
        }
        std::vector<float> x(H), q(H), k(H), v(H);
        std::vector<float> attn_scores(model.header.max_seq_len);
        std::vector<float> attn_out(H);
        std::vector<float> gate(FFN), up(FFN), hidden(FFN), ffn_out(H);

        if (model.is_mla) {
            // MLA decode: no learned pos_emb, uses RoPE
            for (int i = 0; i < H; i++)
                x[i] = tok_emb[tokens.back() * H + i];
        } else {
            const float* pos_emb = model.fw_ptr("pos_embedding.weight");
            for (int i = 0; i < H; i++)
                x[i] = tok_emb[tokens.back() * H + i] + pos_emb[pos * H + i];
        }

        for (int l = 0; l < L; l++) {
            char pfx[64];
            snprintf(pfx, sizeof(pfx), "layers.%d.", l);

            // Attention (pre-norm)
            std::vector<float> normed = x;
            rmsnorm(normed.data(), model.fw_ptr(std::string(pfx) + "attn_norm.weight"), H);
            float x_scale = absmean(normed.data(), H);

            if (model.is_mla) {
                const int KV = model.kv_latent_dim;
                const int RP = model.rope_per_head;
                const int RD = model.rope_dim;
                const int EHD = HD + RP;

                std::string pfx_s(pfx);
                std::string qn = pfx_s + "attn.q_proj.latent_weights";
                std::string kdn = pfx_s + "attn.kv_down_proj.latent_weights";
                std::string kun = pfx_s + "attn.k_up_proj.latent_weights";
                std::string vun = pfx_s + "attn.v_up_proj.latent_weights";
                std::string qrn = pfx_s + "attn.q_rope_proj.latent_weights";
                std::string krn = pfx_s + "attn.k_rope_proj.latent_weights";
                std::string on = pfx_s + "attn.o_proj.latent_weights";

                // Q main
                ternary_matmul_auto(normed.data(), model.tw(qn), q.data(), x_scale, decode);

                // KV latent + expand to full K/V for this position
                std::vector<float> kv_latent(KV);
                ternary_matmul_auto(normed.data(), model.tw(kdn), kv_latent.data(), x_scale, decode);

                const float* kup_d = model.tw(kun).floats.data();
                const float* vup_d = model.tw(vun).floats.data();
                float* kc = cache.k_full_cache[l].data() + pos * H;
                float* vc = cache.v_full_cache[l].data() + pos * H;
                for (int r = 0; r < H; r++) {
                    kc[r] = dot_product_simd(kv_latent.data(), kup_d + r * KV, KV);
                    vc[r] = dot_product_simd(kv_latent.data(), vup_d + r * KV, KV);
                }

                // Q/K RoPE
                std::vector<float> q_rope(RD), k_rope(RD);
                ternary_matmul_auto(normed.data(), model.tw(qrn), q_rope.data(), x_scale, decode);
                ternary_matmul_auto(normed.data(), model.tw(krn), k_rope.data(), x_scale, decode);

                const float* fc = model.freqs_cis.data();
                for (int h = 0; h < NH; h++) {
                    apply_rope(q_rope.data() + h * RP, RP, pos, fc);
                    apply_rope(k_rope.data() + h * RP, RP, pos, fc);
                }

                // Store latent + k_rope to cache (for future MLA re-compression if needed)
                int lc_pos = pos * KV;
                for (int i = 0; i < KV; i++)
                    cache.latent_cache[l][lc_pos + i] = kv_latent[i];
                int krc_pos = pos * RD;
                for (int i = 0; i < RD; i++)
                    cache.k_rope_cache[l][krc_pos + i] = k_rope[i];

                // Attention: read from cached K_full/V_full + K_rope
                int actual_len = pos + 1;
                float eff_scale = 1.0f / (float)EHD;
                for (int head = 0; head < NH; head++) {
                    float* qh = q.data() + head * HD;
                    float* qrh = q_rope.data() + head * RP;

                    for (int t = 0; t < actual_len; t++) {
                        float* kh = cache.k_full_cache[l].data() + t * H + head * HD;
                        float* krh = cache.k_rope_cache[l].data() + t * RD + head * RP;
                        float s = dot_product_simd(qh, kh, HD) * eff_scale;
                        s += dot_product_simd(qrh, krh, RP) * eff_scale;
                        if (s > 80.0f) s = 80.0f;
                        else if (s < -80.0f) s = -80.0f;
                        attn_scores[t] = s;
                    }
                    softmax(attn_scores.data(), actual_len);
                    for (int d = 0; d < HD; d++) {
                        float sum = 0.0f;
                        for (int t = 0; t < actual_len; t++)
                            sum += attn_scores[t] * cache.v_full_cache[l][t * H + head * HD + d];
                        attn_out[head * HD + d] = sum;
                    }
                }
                std::vector<float> proj_out(H);
                float o_scale = absmean(attn_out.data(), H);
                ternary_matmul_auto(attn_out.data(), model.tw(on), proj_out.data(), o_scale, decode);
                for (int i = 0; i < H; i++) x[i] += proj_out[i];
            } else {
                std::string qn = std::string(pfx) + "attn.q_proj.latent_weights";
                std::string kn = std::string(pfx) + "attn.k_proj.latent_weights";
                std::string vn = std::string(pfx) + "attn.v_proj.latent_weights";
                std::string on = std::string(pfx) + "attn.o_proj.latent_weights";

                ternary_matmul_auto(normed.data(), model.tw(qn), q.data(), x_scale, decode);
                ternary_matmul_auto(normed.data(), model.tw(kn), k.data(), x_scale, decode);
                ternary_matmul_auto(normed.data(), model.tw(vn), v.data(), x_scale, decode);
                cap_fill(cap, std::string(pfx) + "attn.q_proj", normed.data(), H, q.data(), H);
                cap_fill(cap, std::string(pfx) + "attn.k_proj", normed.data(), H, k.data(), H);
                cap_fill(cap, std::string(pfx) + "attn.v_proj", normed.data(), H, v.data(), H);

                for (int i = 0; i < H; i++) {
                    cache.k_cache[l][pos * H + i] = k[i];
                    cache.v_cache[l][pos * H + i] = v[i];
                }

                for (int head = 0; head < NH; head++) {
                    float scale = 1.0f / sqrtf((float)HD);
                    int actual_len = pos + 1;
                    const float* q_head = q.data() + head * HD;
                    for (int t = 0; t < actual_len; t++) {
                        float s = dot_product_simd(q_head,
                            cache.k_cache[l].data() + t * H + head * HD, HD) * scale;
                        if (s > 80.0f) s = 80.0f;
                        else if (s < -80.0f) s = -80.0f;
                        attn_scores[t] = s;
                    }
                    softmax(attn_scores.data(), actual_len);
                    for (int d = 0; d < HD; d++) {
                        float sum = 0.0f;
                        for (int t = 0; t < actual_len; t++)
                            sum += attn_scores[t] * cache.v_cache[l][t * H + head * HD + d];
                        attn_out[head * HD + d] = sum;
                    }
                }

                std::vector<float> proj_out(H);
                float o_scale = absmean(attn_out.data(), H);
                ternary_matmul_auto(attn_out.data(), model.tw(on), proj_out.data(), o_scale, decode);
                cap_fill(cap, std::string(pfx) + "attn.o_proj", attn_out.data(), H, proj_out.data(), H);
                for (int i = 0; i < H; i++) x[i] += proj_out[i];
            }

            // FFN (pre-norm) — shared between MLA and standard
            std::vector<float> ffn_normed = x;
            rmsnorm(ffn_normed.data(), model.fw_ptr(std::string(pfx) + "ffn_norm.weight"), H);
            float ffn_scale = absmean(ffn_normed.data(), H);

            std::string fused_n = std::string(pfx) + "ffn.gate_up_proj.latent_weights";
            std::string down_n = std::string(pfx) + "ffn.down_proj.latent_weights";

            std::vector<float> fused(2 * FFN);
            ternary_matmul_auto(ffn_normed.data(), model.tw(fused_n), fused.data(), ffn_scale, decode);
            cap_fill(cap, std::string(pfx) + "ffn.gate_up_proj", ffn_normed.data(), H, fused.data(), 2 * FFN);
            for (int i = 0; i < FFN; i++) { gate[i] = fused[i]; up[i] = fused[FFN + i]; }
            for (int i = 0; i < FFN; i++) hidden[i] = silu(gate[i]) * up[i];

            float h_scale = absmean(hidden.data(), FFN);
            ternary_matmul_auto(hidden.data(), model.tw(down_n), ffn_out.data(), h_scale, decode);
            cap_fill(cap, std::string(pfx) + "ffn.down_proj", hidden.data(), FFN, ffn_out.data(), H);
            for (int i = 0; i < H; i++) x[i] += ffn_out[i];
        }

        rmsnorm(x.data(), model.fw_ptr("norm.weight"), H);
        if (cap) cap->h.assign(x.data(), x.data() + H);
        std::vector<float> logits(V);
        {
            const auto& emb = model.fp32_weights.at("token_embedding.weight");
            if (!emb.int8_data.empty()) {
                matmul_int8_decode(x.data(), emb.int8_data.data(), logits.data(), V, H, emb.int8_scale);
            } else {
                matmul_fp32_decode(x.data(), tok_emb, logits.data(), V, H);
            }
        }
        cache.pos++;
        return logits;
    }

    // Prefill path (seq_len > 1)
    std::vector<float> x(seq_len * H);

    if (model.is_mla) {
        // MLA prefill: no learned pos_emb
        for (int j = 0; j < seq_len; j++)
            for (int i = 0; i < H; i++)
                x[j * H + i] = tok_emb[tokens[j] * H + i];
    } else {
        const float* pos_emb = model.fw_ptr("pos_embedding.weight");
        for (int j = 0; j < seq_len; j++)
            for (int i = 0; i < H; i++)
                x[j * H + i] = tok_emb[tokens[j] * H + i] + pos_emb[(pos + j) * H + i];
    }

    std::vector<float> q(seq_len * H), k(seq_len * H), v(seq_len * H);
    std::vector<float> attn_out(seq_len * H);

    std::vector<float> gate(seq_len * FFN), up(seq_len * FFN);
    std::vector<float> hidden(seq_len * FFN), ffn_out(seq_len * H);

    for (int l = 0; l < L; l++) {
        char pfx[64];
        snprintf(pfx, sizeof(pfx), "layers.%d.", l);
        std::string pfx_s(pfx);

        if (model.is_mla) {
            const int KV = model.kv_latent_dim;
            const int RP = model.rope_per_head;
            const int RD = model.rope_dim;
            const int EHD = HD + RP;
            float eff_scale = 1.0f / (float)EHD;

            std::string qn = pfx_s + "attn.q_proj.latent_weights";
            std::string kdn = pfx_s + "attn.kv_down_proj.latent_weights";
            std::string kun = pfx_s + "attn.k_up_proj.latent_weights";
            std::string vun = pfx_s + "attn.v_up_proj.latent_weights";
            std::string qrn = pfx_s + "attn.q_rope_proj.latent_weights";
            std::string krn = pfx_s + "attn.k_rope_proj.latent_weights";
            std::string on = pfx_s + "attn.o_proj.latent_weights";

            // Per-position compute: Q, kv_latent, Q/K RoPE, then expand K/V to cache
            const float* kup_d = model.tw(kun).floats.data();
            const float* vup_d = model.tw(vun).floats.data();
            std::vector<float> kv_latent_new(seq_len * KV);
            std::vector<float> q_rope(seq_len * RD);
            std::vector<float> k_rope_new(seq_len * RD);

            #pragma omp parallel for if(seq_len > 1)
            for (int j = 0; j < seq_len; j++) {
                float* xj = x.data() + j * H;
                std::vector<float> normed(H);
                memcpy(normed.data(), xj, H * sizeof(float));
                rmsnorm(normed.data(), model.fw_ptr(pfx_s + "attn_norm.weight"), H);
                float xs = absmean(normed.data(), H);

                ternary_matmul_auto(normed.data(), model.tw(qn), q.data() + j * H, xs, false);
                ternary_matmul_auto(normed.data(), model.tw(kdn), kv_latent_new.data() + j * KV, xs, false);
                ternary_matmul_auto(normed.data(), model.tw(qrn), q_rope.data() + j * RD, xs, false);
                ternary_matmul_auto(normed.data(), model.tw(krn), k_rope_new.data() + j * RD, xs, false);

                const float* fc = model.freqs_cis.data();
                for (int h = 0; h < NH; h++) {
                    int p = pos + j;
                    apply_rope(q_rope.data() + j * RD + h * RP, RP, p, fc);
                    apply_rope(k_rope_new.data() + j * RD + h * RP, RP, p, fc);
                }

                // Expand K/V to full and store to cache
                const float* lt = kv_latent_new.data() + j * KV;
                float* kc = cache.k_full_cache[l].data() + (pos + j) * H;
                float* vc = cache.v_full_cache[l].data() + (pos + j) * H;
                for (int r = 0; r < H; r++) {
                    kc[r] = dot_product_simd(lt, kup_d + r * KV, KV);
                    vc[r] = dot_product_simd(lt, vup_d + r * KV, KV);
                }
            }

            // Store latents + k_rope to cache
            for (int j = 0; j < seq_len; j++) {
                int ci = (pos + j) * KV;
                for (int i = 0; i < KV; i++)
                    cache.latent_cache[l][ci + i] = kv_latent_new[j * KV + i];
                ci = (pos + j) * RD;
                for (int i = 0; i < RD; i++)
                    cache.k_rope_cache[l][ci + i] = k_rope_new[j * RD + i];
            }

            // Attention + output projection: read K/V directly from cache
            #pragma omp parallel for if(seq_len > 1)
            for (int j = 0; j < seq_len; j++) {
                std::vector<float> attn_local(model.header.max_seq_len);
                int actual_len = pos + j + 1;
                float* out_j = attn_out.data() + j * H;

                for (int head = 0; head < NH; head++) {
                    float* qh = q.data() + j * H + head * HD;
                    float* qrh = q_rope.data() + j * RD + head * RP;
                    for (int t = 0; t < actual_len; t++) {
                        float* kh = cache.k_full_cache[l].data() + t * H + head * HD;
                        float* krh = cache.k_rope_cache[l].data() + t * RD + head * RP;
                        float s = dot_product_simd(qh, kh, HD) * eff_scale;
                        s += dot_product_simd(qrh, krh, RP) * eff_scale;
                        if (s > 80.0f) s = 80.0f;
                        else if (s < -80.0f) s = -80.0f;
                        attn_local[t] = s;
                    }
                    softmax(attn_local.data(), actual_len);
                    for (int d = 0; d < HD; d++) {
                        float sum = 0.0f;
                        for (int t = 0; t < actual_len; t++)
                            sum += attn_local[t] * cache.v_full_cache[l][t * H + head * HD + d];
                        out_j[head * HD + d] = sum;
                    }
                }

                float os = absmean(out_j, H);
                std::vector<float> proj_out(H);
                ternary_matmul_auto(out_j, model.tw(on), proj_out.data(), os, false);
                float* xj = x.data() + j * H;
                for (int i = 0; i < H; i++) xj[i] += proj_out[i];
            }
        } else {
            // Standard prefill (existing code)
            #pragma omp parallel for if(seq_len > 1)
            for (int j = 0; j < seq_len; j++) {
                float* xj = x.data() + j * H;
                std::vector<float> normed(H);
                memcpy(normed.data(), xj, H * sizeof(float));
                rmsnorm(normed.data(), model.fw_ptr(std::string(pfx) + "attn_norm.weight"), H);
                float xs = absmean(normed.data(), H);
                ternary_matmul_auto(normed.data(), model.tw(std::string(pfx) + "attn.q_proj.latent_weights"), q.data() + j * H, xs, false);
                ternary_matmul_auto(normed.data(), model.tw(std::string(pfx) + "attn.k_proj.latent_weights"), k.data() + j * H, xs, false);
                ternary_matmul_auto(normed.data(), model.tw(std::string(pfx) + "attn.v_proj.latent_weights"), v.data() + j * H, xs, false);
            }

            for (int j = 0; j < seq_len; j++)
                for (int i = 0; i < H; i++) {
                    cache.k_cache[l][(pos + j) * H + i] = k[j * H + i];
                    cache.v_cache[l][(pos + j) * H + i] = v[j * H + i];
                }

            #pragma omp parallel for if(seq_len > 1)
            for (int j = 0; j < seq_len; j++) {
                std::vector<float> attn_local(model.header.max_seq_len);
                int actual_len = pos + j + 1;
                for (int head = 0; head < NH; head++) {
                    float scale = 1.0f / sqrtf((float)HD);
                    const float* q_head = q.data() + j * H + head * HD;
                    for (int t = 0; t < actual_len; t++) {
                        float s = dot_product_simd(q_head,
                            cache.k_cache[l].data() + t * H + head * HD, HD) * scale;
                        if (s > 80.0f) s = 80.0f;
                        else if (s < -80.0f) s = -80.0f;
                        attn_local[t] = s;
                    }
                    softmax(attn_local.data(), actual_len);
                    for (int d = 0; d < HD; d++) {
                        float sum = 0.0f;
                        for (int t = 0; t < actual_len; t++)
                            sum += attn_local[t] * cache.v_cache[l][t * H + head * HD + d];
                        attn_out[j * H + head * HD + d] = sum;
                    }
                }
            }

            // Output projection
            std::vector<float> proj_out(seq_len * H);
            #pragma omp parallel for if(seq_len > 1)
            for (int j = 0; j < seq_len; j++) {
                float* xj = x.data() + j * H;
                float os = absmean(attn_out.data() + j * H, H);
                ternary_matmul_auto(attn_out.data() + j * H, model.tw(std::string(pfx) + "attn.o_proj.latent_weights"), proj_out.data() + j * H, os, false);
                for (int i = 0; i < H; i++) xj[i] += proj_out[j * H + i];
            }
        }

        // FFN — shared
        #pragma omp parallel for if(seq_len > 1)
        for (int j = 0; j < seq_len; j++) {
            float* xj = x.data() + j * H;
            std::vector<float> ffn_normed(H);
            memcpy(ffn_normed.data(), xj, H * sizeof(float));
            rmsnorm(ffn_normed.data(), model.fw_ptr(std::string(pfx) + "ffn_norm.weight"), H);
            float fs = absmean(ffn_normed.data(), H);

            std::string fused_n = std::string(pfx) + "ffn.gate_up_proj.latent_weights";
            std::string down_n = std::string(pfx) + "ffn.down_proj.latent_weights";
            std::vector<float> fused(2 * FFN);
            ternary_matmul_auto(ffn_normed.data(), model.tw(fused_n), fused.data(), fs, false);
            for (int i = 0; i < FFN; i++) {
                gate[j * FFN + i] = fused[i];
                up[j * FFN + i] = fused[FFN + i];
            }
            for (int i = 0; i < FFN; i++) hidden[j * FFN + i] = silu(gate[j * FFN + i]) * up[j * FFN + i];

            float hs = absmean(hidden.data() + j * FFN, FFN);
            ternary_matmul_auto(hidden.data() + j * FFN, model.tw(down_n), ffn_out.data() + j * H, hs, false);
            for (int i = 0; i < H; i++) xj[i] += ffn_out[j * H + i];
        }
    }

    // Final norm + LM head (only last position for prefill)
    int last = seq_len - 1;
    rmsnorm(x.data() + last * H, model.fw_ptr("norm.weight"), H);
    std::vector<float> logits(V);
    {
        const auto& emb = model.fp32_weights.at("token_embedding.weight");
        if (!emb.int8_data.empty()) {
            matmul_int8_decode(x.data() + last * H, emb.int8_data.data(), logits.data(), V, H, emb.int8_scale);
        } else {
            for (int vi = 0; vi < V; vi++) {
                float dot = 0.0f;
                for (int i = 0; i < H; i++) dot += x[last * H + i] * tok_emb[vi * H + i];
                logits[vi] = dot;
            }
        }
    }

    cache.pos += seq_len;
    return logits;
}

static std::mt19937& rng() {
    static thread_local std::mt19937 gen((std::random_device())());
    return gen;
}

static int sample(const std::vector<float>& logits, float temperature, int top_k, float top_p,
                  const std::vector<int>& context = {}, float repeat_penalty = 1.0f) {
    int n = (int)logits.size();

    // Greedy (temperature=0): argmax with optional repeat penalty
    if (temperature <= 0.0f) {
        std::vector<float> penalized(n);
        for (int i = 0; i < n; i++) penalized[i] = logits[i];
        if (repeat_penalty != 1.0f && !context.empty()) {
            for (int id : context) {
                if (id < 0 || id >= n) continue;
                if (penalized[id] > 0.0f) penalized[id] /= repeat_penalty;
                else penalized[id] *= repeat_penalty;
            }
        }
        int best = 0;
        for (int i = 1; i < n; i++)
            if (penalized[i] > penalized[best]) best = i;
        return best;
    }

    std::vector<float> scaled(n);
    for (int i = 0; i < n; i++) scaled[i] = logits[i];

    // Repetition penalty: penalize tokens already in context
    if (repeat_penalty != 1.0f && !context.empty()) {
        for (int id : context) {
            if (id < 0 || id >= n) continue;
            if (scaled[id] > 0.0f) scaled[id] /= repeat_penalty;
            else scaled[id] *= repeat_penalty;
        }
    }

    for (int i = 0; i < n; i++) scaled[i] /= temperature;

    if (top_k > 0 && top_k < n) {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(),
            [&](int a, int b) { return scaled[a] > scaled[b]; });
        float threshold = scaled[idx[top_k - 1]];
        for (int i = 0; i < n; i++)
            if (scaled[i] < threshold) scaled[i] = -INFINITY;
    }

    float mx = *std::max_element(scaled.begin(), scaled.end());
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        scaled[i] = expf(scaled[i] - mx);
        sum += scaled[i];
    }
    if (sum > 0) for (int i = 0; i < n; i++) scaled[i] /= sum;

    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<int> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
            [&](int a, int b) { return scaled[a] > scaled[b]; });
        float cum = 0.0f;
        for (int i = 0; i < n; i++) {
            if (cum >= top_p) scaled[idx[i]] = 0.0f;
            cum += scaled[idx[i]];
        }
        sum = 0.0f;
        for (int i = 0; i < n; i++) sum += scaled[i];
        if (sum > 0) for (int i = 0; i < n; i++) scaled[i] /= sum;
    }

    float r = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng());
    float cum = 0.0f;
    for (int i = 0; i < n; i++) { cum += scaled[i]; if (r < cum) return i; }
    return n - 1;
}

// ──────────────────────────────────────────────────────────────
// Self-learning kernels (v6)
// ──────────────────────────────────────────────────────────────

// Flip ternary bits where |accumulator| exceeds the threshold, in place.
// Updates both the packed 2-bit data and the precomputed float rows and
// resets flipped accumulator entries to zero (mirrors apply_bit_flips in the
// PyTorch trainer). Returns the number of bits flipped.
// toggle=true: anti-stiction mode — a weight already saturated in the push
// direction (+1 pushed up / -1 pushed down) is kicked to the opposite
// extreme instead of staying a no-op.
// v7 (w.is_v7): code 11 = ±2 outlier with the sign in the dense side-channel
// blob. Mirrors the Python dynamics:
//   - promote:  |acc| > outlier_mult * threshold and |w| <= 1 -> ±2 (sign of
//               acc); acc parked at ±threshold.
//   - demote:   outlier whose |acc| dropped below threshold -> ±1 toward acc.
//   - outliers pushed in their own direction stay put (acc reset).
//   - the sign blob is rebuilt from the packed data at the end.
static int apply_bit_flips(TernaryWeightXNOR& w, float threshold, bool toggle = false,
                           float outlier_mult = 3.0f) {
    const int rows = w.rows, cols = w.cols;
    const int row_bytes = (cols + 3) / 4;
    const size_t n = (size_t)rows * cols;
    const bool v7 = w.is_v7;
    const float prom_thr = threshold * outlier_mult;
    std::vector<int8_t> act(n, 0);  // 0=none, 1=flip up, -1=flip down, 2=promote
    bool any = false;
    for (size_t i = 0; i < n; i++) {
        float a = w.accumulator[i];
        int8_t f = (a > threshold) ? 1 : ((a < -threshold) ? (int8_t)-1 : (int8_t)0);
        if (v7 && (a > prom_thr || a < -prom_thr)) f = 2;
        act[i] = f;
        if (f) any = true;
    }
    if (!any) return 0;

    int flips = 0;
    for (int r = 0; r < rows; r++) {
        float* prow = w.floats.data() + (size_t)r * cols;
        uint8_t* packed_row = w.packed.data() + (size_t)r * row_bytes;
        float* acc_row = w.accumulator.data() + (size_t)r * cols;
        for (int c = 0; c < cols; c++) {
            size_t i = (size_t)r * cols + c;
            int8_t f = act[i];
            if (!f) continue;
            float wv = prow[c];
            bool is_std = (fabsf(wv) <= 1.5f);
            if (f == 2) {
                if (!is_std) {
                    // outlier with huge acc: fall back to normal flip (same-dir stays)
                    f = (acc_row[c] > 0) ? 1 : -1;
                } else {
                    prow[c] = (acc_row[c] > 0) ? 2.0f : -2.0f;
                    acc_row[c] = (acc_row[c] > 0) ? threshold : -threshold;
                    flips++;
                    continue;
                }
            }
            if (!is_std) {
                bool same_dir = (wv > 0 && f > 0) || (wv < 0 && f < 0);
                if (same_dir) { acc_row[c] = 0.0f; continue; }  // stays outlier
            }
            float nv;
            if (toggle && ((wv > 0.5f && f > 0) || (wv < -0.5f && f < 0))) {
                nv = -wv;  // kick to opposite extreme
            } else {
                nv = wv + (float)f;
            }
            if (nv > 1.0f) nv = 1.0f;
            else if (nv < -1.0f) nv = -1.0f;
            prow[c] = nv;
            acc_row[c] = 0.0f;
            flips++;
        }
    }

    // v7 demote: outliers whose accumulator relaxed below threshold -> ±1
    if (v7) {
        for (size_t i = 0; i < n; i++) {
            if (act[i]) continue;
            float wv = w.floats[i];
            if (fabsf(wv) <= 1.5f) continue;
            float a = w.accumulator[i];
            if (fabsf(a) < threshold) {
                w.floats[i] = (a > 0) ? 1.0f : ((a < 0) ? -1.0f : (wv > 0 ? 1.0f : -1.0f));
                w.accumulator[i] = 0.0f;
                flips++;
            }
        }
    }

    // Repack floats -> 2-bit codes and rebuild the dense sign blob.
    for (int r = 0; r < rows; r++) {
        float* prow = w.floats.data() + (size_t)r * cols;
        uint8_t* packed_row = w.packed.data() + (size_t)r * row_bytes;
        for (int c = 0; c < cols; c++) {
            float v = prow[c];
            int enc;  // {-2,-1,0,1,2} -> codes {3,0,1,2,3}; -2's sign lives in the blob
            if (v > 1.5f) enc = 3;
            else if (v < -1.5f) enc = 3;
            else if (v > 0.5f) enc = 2;
            else if (v < -0.5f) enc = 0;
            else enc = 1;
            int byte_idx = c >> 2;
            int shift = 6 - (c & 3) * 2;
            packed_row[byte_idx] = (uint8_t)(
                (packed_row[byte_idx] & ~(3 << shift)) | (enc << shift));
        }
    }
    if (v7) {
        size_t count = 0;
        for (size_t i = 0; i < n; i++) if (fabsf(w.floats[i]) > 1.5f) count++;
        w.outlier_blob.assign((count + 7) / 8, 0);
        size_t k = 0;
        for (size_t i = 0; i < n; i++) {
            float v = w.floats[i];
            if (fabsf(v) > 1.5f) {
                if (v > 0) w.outlier_blob[k >> 3] |= (uint8_t)(0x80 >> (k & 7));
                k++;
            }
        }
    }
    return flips;
}

// Rule 'c' (predictive coding): feed one block's gradient into the weights.
//   grad[o][i] += (y_t - y_{t-1})[o] * x_{t-1}[i]   over the block
//   acc = acc * decay + (-sign(grad))
static void sl_feed_predictive(TernaryWeightXNOR& w, const std::vector<float>& grad,
                               float acc_decay) {
    const size_t n = (size_t)w.rows * w.cols;
    for (size_t j = 0; j < n; j++) {
        float g = grad[j];
        float d = (g > 0.0f) ? -1.0f : ((g < 0.0f) ? 1.0f : 0.0f);
        w.accumulator[j] = w.accumulator[j] * acc_decay + d;
    }
}

// ──────────────────────────────────────────────────────────────
// Atomic write-back (v6)
// ──────────────────────────────────────────────────────────────

static std::string build_sl_metadata(const Model& m) {
    const char* rule = "c";
    if (m.sl_rule == 1) rule = "b";
    else if (m.sl_rule == 2) rule = "p";
    else if (m.sl_rule == 3) rule = "h";
    else if (m.sl_rule == 4) rule = "e";
    char buf[576];
    snprintf(buf, sizeof(buf),
        "{\"_export_version\":%u,\"sl_rule\":\"%s\",\"sl_threshold\":%.4f,"
        "\"sl_acc_decay\":%.4f,\"sl_flip_every_n\":%d,\"sl_toggle\":%d,"
        "\"sl_logit_scale\":%.8f,"
        "\"sl_lr_embedding\":%.8f,\"sl_wd_embedding\":%.4f,\"sl_block_size\":%d,"
        "\"sl_outlier_mult\":%.4f}",
        m.header.version, rule, m.sl_threshold, m.sl_acc_decay, m.sl_flip_every_n, m.sl_toggle,
        m.sl_logit_scale,
        m.sl_lr_embedding, m.sl_wd_embedding, m.sl_block_size, m.sl_outlier_mult);
    return std::string(buf);
}

// Serialize the (possibly mutated) model back to a binary (v6 or v7, matching
// the loaded header version). Writes to a .tmp sibling then atomically renames
// over the destination so a crash never leaves a truncated file.
static void save_model(Model& model, const char* path) {
    std::string tmp = std::string(path) + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) { fprintf(stderr, "save_model: cannot open %s\n", tmp.c_str()); exit(1); }

    const ModelHeader& H = model.header;
    uint8_t hdr[64];
    memset(hdr, 0, 64);
    memcpy(hdr, "TETR", 4);
    uint32_t ver = (H.version >= 7) ? 7u : 6u;
    memcpy(hdr + 4, &ver, 4);
    memcpy(hdr + 8,  &H.vocab_size, 4);
    memcpy(hdr + 12, &H.hidden_dim, 4);
    memcpy(hdr + 16, &H.num_layers, 4);
    memcpy(hdr + 20, &H.num_heads, 4);
    memcpy(hdr + 24, &H.ffn_dim, 4);
    memcpy(hdr + 28, &H.max_seq_len, 4);
    memcpy(hdr + 32, &H.ternary_params, 8);
    memcpy(hdr + 40, &H.fp32_params, 8);
    memcpy(hdr + 48, &H.flags, 2);
    memcpy(hdr + 50, &H.kv_latent_dim, 2);
    memcpy(hdr + 52, &H.rope_per_head, 2);
    memcpy(hdr + 54, &H.group_size, 2);
    fwrite(hdr, 1, 64, f);

    for (auto& kv : model.ternary_weights) {
        const std::string& name = kv.first;
        TernaryWeightXNOR& w = kv.second;
        uint32_t nl = (uint32_t)name.size();
        uint16_t rows = (uint16_t)w.rows, cols = (uint16_t)w.cols;
        uint16_t gs = (uint16_t)w.group_size, na = (uint16_t)w.alphas.size();
        fwrite(&nl, 4, 1, f);
        fwrite(name.data(), 1, nl, f);
        fwrite(&rows, 2, 1, f); fwrite(&cols, 2, 1, f);
        fwrite(&gs, 2, 1, f); fwrite(&na, 2, 1, f);
        if (na) fwrite(w.alphas.data(), 4, na, f);
        fwrite(w.packed.data(), 1, w.packed.size(), f);
        if (!w.accumulator.empty())
            fwrite(w.accumulator.data(), 4, w.accumulator.size(), f);
        if (ver >= 7) {
            // Rebuild the dense sign blob from the dequantized floats so it is
            // always consistent with the packed code-11 positions, then write
            // the trimmed ceil(n/8) bytes (matches the Python exporter).
            const size_t total = (size_t)w.rows * w.cols;
            size_t count = 0;
            for (size_t i = 0; i < total; i++)
                if (fabsf(w.floats[i]) > 1.5f) count++;
            std::vector<uint8_t> blob((count + 7) / 8, 0);
            size_t k = 0;
            for (size_t i = 0; i < total; i++) {
                float v = w.floats[i];
                if (fabsf(v) > 1.5f) {
                    if (v > 0) blob[k >> 3] |= (uint8_t)(0x80 >> (k & 7));
                    k++;
                }
            }
            uint32_t n_outliers = (uint32_t)count;
            fwrite(&n_outliers, 4, 1, f);
            if (!blob.empty()) fwrite(blob.data(), 1, blob.size(), f);
        }
    }

    for (auto& kv : model.fp32_weights) {
        const std::string& name = kv.first;
        FP32Weight& w = kv.second;
        uint32_t nl = (uint32_t)name.size();
        uint8_t ndim = (uint8_t)w.shape.size();
        uint8_t dtype = 0;  // write everything back as FP32
        uint32_t dims[4] = {1, 1, 1, 1};
        for (size_t i = 0; i < w.shape.size() && i < 4; i++) dims[i] = (uint32_t)w.shape[i];
        fwrite(&nl, 4, 1, f);
        fwrite(name.data(), 1, nl, f);
        fwrite(&ndim, 1, 1, f); fwrite(&dtype, 1, 1, f);
        fwrite(dims, 4, 4, f);
        fwrite(w.data.data(), 4, w.data.size(), f);
    }

    std::string meta = build_sl_metadata(model);
    fwrite("META", 1, 4, f);
    uint32_t ml = (uint32_t)meta.size();
    fwrite(&ml, 4, 1, f);
    fwrite(meta.data(), 1, ml, f);
    fclose(f);

#ifdef _WIN32
    MoveFileExA(tmp.c_str(), path, MOVEFILE_REPLACE_EXISTING);
#else
    ::remove(path);
    ::rename(tmp.c_str(), path);
#endif
}

}  // namespace tetra
