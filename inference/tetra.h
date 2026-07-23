#pragma once
// Tetra — C++ inference engine
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
    float alpha;
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

// Dequantize one row of 2-bit packed ternary → float array
static inline void dequantize_row(const uint8_t* packed, int row_offset, int cols, float* out) {
    static const float lut[4] = {-1.0f, 0.0f, 1.0f, 0.0f};
    int c = 0, num_bytes = (cols + 3) / 4;
    for (int b = 0; b < num_bytes; b++) {
        uint8_t byte = packed[row_offset + b];
        int rem = (cols - c < 4) ? (cols - c) : 4;
        for (int i = 0; i < rem; i++, c++) {
            out[c] = lut[(byte >> (6 - i * 2)) & 3];
        }
    }
}

// Precompute: dequantize all weights to float at load time
static void precompute_floats(TernaryWeightXNOR& w) {
    w.floats.resize(w.rows * w.cols);
    for (int r = 0; r < w.rows; r++) {
        dequantize_row(w.packed.data(), r * w.cols / 4, w.cols, w.floats.data() + r * w.cols);
    }
}

// Precomputed float matmul with prefetch
static void ternary_matmul_precomputed(
    const float* x, const TernaryWeightXNOR& w, float* out
) {
    const float* data = w.floats.data();
    for (int r = 0; r < w.rows; r++) {
        out[r] = dot_product_simd(x, data + r * w.cols, w.cols);
    }
}

// Precomputed decode path: same but with row prefetch
static void ternary_matmul_precomputed_decode(
    const float* x, const TernaryWeightXNOR& w, float* out
) {
    const int rows = w.rows;
    const int cols = w.cols;
    const float* data = w.floats.data();
    for (int r = 0; r < rows; r++) {
        if (r + 2 < rows) TETRA_PREFETCH(data + (r + 2) * cols);
        out[r] = dot_product_simd(x, data + r * cols, cols);
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
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// Mean absolute value (for XNOR scale factor)
static float absmean(const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += fabsf(x[i]);
    return sum / n;
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
};

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

    fprintf(stderr, "Tetra: %d layers, hidden=%d, heads=%d, ffn=%d, vocab=%d, seq=%d\n",
            h.num_layers, h.hidden_dim, h.num_heads, h.ffn_dim, h.vocab_size, h.max_seq_len);

    for (uint32_t layer = 0; layer < h.num_layers; layer++) {
        for (int t = 0; t < 6; t++) {
            uint32_t name_len;
            r.read(name_len);
            std::string name = r.read_str(name_len);

            uint16_t rows, cols;
            r.read(rows);
            r.read(cols);

            float alpha = 1.0f;
            if (h.version >= 2) r.read(alpha);

            int packed_size = (rows * cols + 3) / 4;
            std::vector<uint8_t> packed(packed_size);
            r.read_bytes(packed.data(), packed_size);

            TernaryWeightXNOR w;
            w.rows = rows;
            w.cols = cols;
            w.alpha = alpha;
            w.packed = std::move(packed);
            precompute_floats(w);
            model.ternary_weights[name] = std::move(w);
        }
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

// Forward pass — supports both decode (single token) and batch prefill (multiple tokens)
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
    bool decode = (seq_len == 1);

    const float* tok_emb = model.fw_ptr("token_embedding.weight");
    const float* pos_emb = model.fw_ptr("pos_embedding.weight");
    int pos = cache.pos;

    if (decode) {
        std::vector<float> x(H), q(H), k(H), v(H);
        std::vector<float> attn_scores(model.header.max_seq_len);
        std::vector<float> attn_out(H);
        std::vector<float> gate(FFN), up(FFN), hidden(FFN), ffn_out(H);

        for (int i = 0; i < H; i++)
            x[i] = tok_emb[tokens.back() * H + i] + pos_emb[pos * H + i];

        for (int l = 0; l < L; l++) {
            char pfx[64];
            snprintf(pfx, sizeof(pfx), "layers.%d.", l);

            // Attention (pre-norm)
            std::vector<float> normed = x;
            rmsnorm(normed.data(), model.fw_ptr(std::string(pfx) + "attn_norm.weight"), H);
            float x_scale = absmean(normed.data(), H);

            std::string qn = std::string(pfx) + "attn.q_proj.latent_weights";
            std::string kn = std::string(pfx) + "attn.k_proj.latent_weights";
            std::string vn = std::string(pfx) + "attn.v_proj.latent_weights";
            std::string on = std::string(pfx) + "attn.o_proj.latent_weights";

            ternary_matmul_auto(normed.data(), model.tw(qn), q.data(), x_scale, decode);
            ternary_matmul_auto(normed.data(), model.tw(kn), k.data(), x_scale, decode);
            ternary_matmul_auto(normed.data(), model.tw(vn), v.data(), x_scale, decode);

            for (int i = 0; i < H; i++) {
                cache.k_cache[l][pos * H + i] = k[i];
                cache.v_cache[l][pos * H + i] = v[i];
            }

            for (int head = 0; head < NH; head++) {
                float scale = 1.0f / sqrtf((float)HD);
                int actual_len = pos + 1;
                const float* q_head = q.data() + head * HD;
                for (int t = 0; t < actual_len; t++) {
                    attn_scores[t] = dot_product_simd(q_head,
                        cache.k_cache[l].data() + t * H + head * HD, HD) * scale;
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
            for (int i = 0; i < H; i++) x[i] += proj_out[i];

            // FFN (pre-norm)
            std::vector<float> ffn_normed = x;
            rmsnorm(ffn_normed.data(), model.fw_ptr(std::string(pfx) + "ffn_norm.weight"), H);
            float ffn_scale = absmean(ffn_normed.data(), H);

            std::string fused_n = std::string(pfx) + "ffn.gate_up_proj.latent_weights";
            std::string down_n = std::string(pfx) + "ffn.down_proj.latent_weights";

            std::vector<float> fused(2 * FFN);
            ternary_matmul_auto(ffn_normed.data(), model.tw(fused_n), fused.data(), ffn_scale, decode);
            for (int i = 0; i < FFN; i++) { gate[i] = fused[i]; up[i] = fused[FFN + i]; }
            for (int i = 0; i < FFN; i++) hidden[i] = silu(gate[i]) * up[i];

            float h_scale = absmean(hidden.data(), FFN);
            ternary_matmul_auto(hidden.data(), model.tw(down_n), ffn_out.data(), h_scale, decode);
            for (int i = 0; i < H; i++) x[i] += ffn_out[i];
        }

        rmsnorm(x.data(), model.fw_ptr("norm.weight"), H);
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

    std::vector<float> x(seq_len * H);

    // Embedding for all positions
    for (int j = 0; j < seq_len; j++)
        for (int i = 0; i < H; i++)
            x[j * H + i] = tok_emb[tokens[j] * H + i] + pos_emb[(pos + j) * H + i];

    std::vector<float> q(seq_len * H), k(seq_len * H), v(seq_len * H);
    std::vector<float> attn_scores(model.header.max_seq_len);
    std::vector<float> attn_out(seq_len * H);

    std::vector<float> gate(seq_len * FFN), up(seq_len * FFN);
    std::vector<float> hidden(seq_len * FFN), ffn_out(seq_len * H);

    for (int l = 0; l < L; l++) {
        char pfx[64];
        snprintf(pfx, sizeof(pfx), "layers.%d.", l);

        // Pre-norm + QKV for all positions
        for (int j = 0; j < seq_len; j++) {
            float* xj = x.data() + j * H;
            rmsnorm(xj, model.fw_ptr(std::string(pfx) + "attn_norm.weight"), H);
            float xs = absmean(xj, H);
            ternary_matmul_auto(xj, model.tw(std::string(pfx) + "attn.q_proj.latent_weights"), q.data() + j * H, xs, false);
            ternary_matmul_auto(xj, model.tw(std::string(pfx) + "attn.k_proj.latent_weights"), k.data() + j * H, xs, false);
            ternary_matmul_auto(xj, model.tw(std::string(pfx) + "attn.v_proj.latent_weights"), v.data() + j * H, xs, false);
        }

        // Store K, V for all new positions
        for (int j = 0; j < seq_len; j++)
            for (int i = 0; i < H; i++) {
                cache.k_cache[l][(pos + j) * H + i] = k[j * H + i];
                cache.v_cache[l][(pos + j) * H + i] = v[j * H + i];
            }

        // Causal multi-head attention for each position
        for (int j = 0; j < seq_len; j++) {
            int actual_len = pos + j + 1;
            for (int head = 0; head < NH; head++) {
                float scale = 1.0f / sqrtf((float)HD);
                const float* q_head = q.data() + j * H + head * HD;
                for (int t = 0; t < actual_len; t++)
                    attn_scores[t] = dot_product_simd(q_head,
                        cache.k_cache[l].data() + t * H + head * HD, HD) * scale;
                softmax(attn_scores.data(), actual_len);
                for (int d = 0; d < HD; d++) {
                    float sum = 0.0f;
                    for (int t = 0; t < actual_len; t++)
                        sum += attn_scores[t] * cache.v_cache[l][t * H + head * HD + d];
                    attn_out[j * H + head * HD + d] = sum;
                }
            }
        }

        // Output projection for all positions
        std::vector<float> proj_out(seq_len * H);
        for (int j = 0; j < seq_len; j++) {
            float* xj = x.data() + j * H;
            float os = absmean(attn_out.data() + j * H, H);
            ternary_matmul_auto(attn_out.data() + j * H, model.tw(std::string(pfx) + "attn.o_proj.latent_weights"), proj_out.data() + j * H, os, false);
            for (int i = 0; i < H; i++) xj[i] += proj_out[j * H + i];
        }

        // FFN for all positions
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

static int sample(const std::vector<float>& logits, float temperature, int top_k, float top_p) {
    int n = (int)logits.size();
    std::vector<float> scaled(n);

    for (int i = 0; i < n; i++) scaled[i] = logits[i] / temperature;

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

    float r = (float)rand() / RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < n; i++) { cum += scaled[i]; if (r < cum) return i; }
    return n - 1;
}

}  // namespace tetra
