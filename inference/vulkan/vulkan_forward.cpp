// vulkan_forward.cpp - Vulkan compute port of the Tetra forward pass.
//
// Loads a v6 self-learning export (same loader as selflearn.cpp), uploads the
// dequantized ternary weights + FP32 tensors to the iGPU, and runs the 128-token
// decode block with compute shaders (shaders/*.comp -> SPIR-V via glslc).
//
// Modes:
//   vulkan_forward.exe <model.bin> <tokens.bin> --eval [max_positions]
//       avg CE over the token slice, identical math to selflearn.exe --eval
//   vulkan_forward.exe <model.bin> <tokens.bin> --bench N
//       N fresh-cache 128-token blocks, reports ms/block + block CE
//   vulkan_forward.exe <model.bin> <tokens.bin> --sl <out.bin> [steps] [log_every]
//       [save_every] [thr] [decay] [flip_every] [toggle]
//       [--toggle-window N] [--thr-anneal RATE]
//       rule-'c' self-learning loop: GPU forward + activation capture + rule-c /
//       embedding gradients, host accumulator feed / bit flips / embedding SGD
//       (same update math as selflearn.exe)
//
// Build: build_vulkan.bat   (needs VULKAN_SDK env var)

#include "tetra.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <chrono>

#include <vulkan/vulkan.h>

using namespace tetra;

#define VK_CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "Vulkan error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); exit(1); } } while (0)

// Vulkan context
struct Device {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    uint32_t qidx = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkDescriptorSet dset = VK_NULL_HANDLE;
};

static void init_vulkan(Device& d) {
    VkApplicationInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "tetra-vulkan";
    ai.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &ai;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &d.inst));

    uint32_t nphys = 0;
    vkEnumeratePhysicalDevices(d.inst, &nphys, nullptr);
    std::vector<VkPhysicalDevice> phys(nphys);
    vkEnumeratePhysicalDevices(d.inst, &nphys, phys.data());
    if (nphys == 0) { fprintf(stderr, "No Vulkan device\n"); exit(1); }
    d.phys = phys[0];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(d.phys, &props);
    fprintf(stderr, "GPU: %s (api %d.%d.%d)\n", props.deviceName,
            VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion),
            VK_VERSION_PATCH(props.apiVersion));

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d.phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qps(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(d.phys, &nq, qps.data());
    for (uint32_t i = 0; i < nq; i++) {
        if (qps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { d.qidx = i; break; }
    }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo dq{};
    dq.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    dq.queueFamilyIndex = d.qidx;
    dq.queueCount = 1;
    dq.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &dq;
    VK_CHECK(vkCreateDevice(d.phys, &dci, nullptr, &d.dev));
    vkGetDeviceQueue(d.dev, d.qidx, 0, &d.queue);

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpci.queueFamilyIndex = d.qidx;
    VK_CHECK(vkCreateCommandPool(d.dev, &cpci, nullptr, &d.pool));

    std::vector<VkDescriptorSetLayoutBinding> binds(8);
    for (int i = 0; i < 8; i++) {
        binds[i].binding = (uint32_t)i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci{};
    dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dlci.bindingCount = (uint32_t)binds.size();
    dlci.pBindings = binds.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d.dev, &dlci, nullptr, &d.dsl));

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &d.dsl;
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 128;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(d.dev, &plci, nullptr, &d.pl));

    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 8;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &ps;
    VK_CHECK(vkCreateDescriptorPool(d.dev, &dpci, nullptr, &d.dpool));
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = d.dpool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &d.dsl;
    VK_CHECK(vkAllocateDescriptorSets(d.dev, &dai, &d.dset));
}

// Buffers (host-visible; Iris Xe is UMA)
struct GPUBuffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    float* host = nullptr;
};

static void gpu_alloc(Device& d, GPUBuffer& b, VkDeviceSize size) {
    b.size = size;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(d.dev, &bci, nullptr, &b.buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(d.dev, b.buf, &mr);
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    VkPhysicalDeviceMemoryProperties mprops;
    vkGetPhysicalDeviceMemoryProperties(d.phys, &mprops);
    uint32_t memtype = ~0u;
    for (uint32_t i = 0; i < mprops.memoryTypeCount; i++) {
        if ((mprops.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            memtype = i; break;
        }
    }
    mai.memoryTypeIndex = memtype;
    VK_CHECK(vkAllocateMemory(d.dev, &mai, nullptr, &b.mem));
    VK_CHECK(vkBindBufferMemory(d.dev, b.buf, b.mem, 0));
    VK_CHECK(vkMapMemory(d.dev, b.mem, 0, size, 0, (void**)&b.host));
}

// Kernels
struct Kernels {
    VkPipeline embed, rmsnorm, mm_partial, mm_reduce, attn, silu, add, cstore;
    VkPipeline capture, rulec, embgrad;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
};

static VkShaderModule load_spv(Device& d, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> code((size_t)sz / 4);
    fread(code.data(), 4, code.size(), f);
    fclose(f);
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = code.size() * 4;
    smci.pCode = code.data();
    VkShaderModule m;
    VK_CHECK(vkCreateShaderModule(d.dev, &smci, nullptr, &m));
    return m;
}

static VkPipeline make_pipeline(Device& d, const char* spv) {
    VkShaderModule mod = load_spv(d, spv);
    VkPipelineShaderStageCreateInfo ss{};
    ss.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ss.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ss.module = mod;
    ss.pName = "main";
    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage = ss;
    cpci.layout = d.pl;
    VkPipeline p;
    VK_CHECK(vkCreateComputePipelines(d.dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p));
    vkDestroyShaderModule(d.dev, mod, nullptr);
    return p;
}

static void init_kernels(Device& d, Kernels& k) {
    k.embed = make_pipeline(d, "shaders/embed.spv");
    k.rmsnorm = make_pipeline(d, "shaders/rmsnorm.spv");
    k.mm_partial = make_pipeline(d, "shaders/mm_partial.spv");
    k.mm_reduce = make_pipeline(d, "shaders/mm_reduce.spv");
    k.attn = make_pipeline(d, "shaders/attention.spv");
    k.silu = make_pipeline(d, "shaders/silu.spv");
    k.add = make_pipeline(d, "shaders/add_residual.spv");
    k.cstore = make_pipeline(d, "shaders/cache_store.spv");
    k.capture = make_pipeline(d, "shaders/capture.spv");
    k.rulec = make_pipeline(d, "shaders/rulec.spv");
    k.embgrad = make_pipeline(d, "shaders/embgrad.spv");
    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = d.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(d.dev, &cai, &k.cmd));
}

// Weight staging
struct TensorMap {
    std::unordered_map<std::string, int> off, rows, cols;
    std::unordered_map<std::string, float> alpha;
    int total = 0;
};

static void stage_ternary(const Model& model, std::vector<float>& wt, TensorMap& tm) {
    std::vector<std::string> ks;
    for (auto& kv : model.ternary_weights) ks.push_back(kv.first);
    std::sort(ks.begin(), ks.end());
    for (auto& name : ks) {
        const TernaryWeightXNOR& w = model.ternary_weights.at(name);
        tm.off[name] = tm.total;
        tm.rows[name] = w.rows;
        tm.cols[name] = w.cols;
        tm.alpha[name] = w.alpha;
        if (!w.alphas.empty() || w.group_size > 0) {
            fprintf(stderr, "WARNING: %s uses per-row/group alphas (%zu, gs=%d) - not supported\n",
                    name.c_str(), w.alphas.size(), w.group_size);
        }
        wt.insert(wt.end(), w.floats.begin(), w.floats.end());
        tm.total += (int)w.floats.size();
    }
    fprintf(stderr, "ternary weights: %d floats (%.1f MB)\n", tm.total, tm.total * 4.0 / 1e6);
}

// ActBuf float layout (x at offset 0)
enum { XA = 0, XNORM = 256, QA = 512, KA = 768, VA = 1024, ATTNO = 1280,
       PROJ = 1536, FUSED = 1792, DOWNO = 3840, LOGITS = 4096, ACT_TOTAL = 4096 + 8192 };
// K/V cache: one big buffer, layer l slot = l*seq*H

// Global partial buffer for the two-stage matmul
static GPUBuffer g_partial;

struct Weights {
    int embOff = 0, posOff = 0;
    std::unordered_map<std::string, int> normOff;
    TensorMap tm;
    int lmRows = 0, lmCols = 0, lmOff = 0;   // lm_head = token_embedding (in wtBuf)
};

// SL capture layout: per captured ternary weight (sorted name order) a y slot
// (Tmax*rows) then an x slot (Tmax*cols) in the history buffer, followed by the
// softmax slot (Tmax*V), the final hidden slot (Tmax*H), and the embedding
// gradient (V*H) in the gradient buffer.
struct SLCapture {
    int Tmax = 0;
    std::unordered_map<std::string, std::pair<int, int>> capBase; // name -> (yBase, xBase)
    std::unordered_map<std::string, int> gradOff;                 // name -> grad offset
    int smBase = 0, hBase = 0, gradE_off = 0;
    int histFloats = 0, gradFloats = 0;
};

static void submit_and_wait(Device& d, Kernels& k);

// Record one token's forward into k.cmd. If caps != nullptr, capture points
// (act offsets + labels) are appended; the caller must read ba.host after submit.
// If dbg_layer >= 0, the command buffer is submitted right after that layer and
// its intermediates are printed (for parity debugging).
// If sl != nullptr, the per-linear x/y activations are copied into the history
// buffer slots (position `pos`), and the final normed hidden goes to sl->hBase.
static void record_forward(Device& d, Kernels& k, int tok, int pos,
                           const Weights& W, const Model& model,
                           std::vector<std::pair<int, const char*>>* caps = nullptr,
                           int dbg_layer = -1, GPUBuffer* ba = nullptr,
                           SLCapture* sl = nullptr) {
    const int H = model.header.hidden_dim;
    const int L = model.header.num_layers;
    const int NH = model.header.num_heads;
    const int HD = H / NH;
    const int FFN = model.header.ffn_dim;
    const int V = model.header.vocab_size;
    const int seq = model.header.max_seq_len;
    (void)V;

    VkCommandBuffer cb = k.cmd;
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cb, &cbi));

    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.pl, 0, 1, &d.dset, 0, nullptr);

    auto push = [&](const void* data, size_t size) {
        vkCmdPushConstants(cb, d.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)size, data);
    };
    auto bar = [&]() {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };
    auto store = [&](int src, int dst, int count) {
        struct { int src, dst, count; } pc = { src, dst, count };
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.capture);
        push(&pc, sizeof(pc));
        vkCmdDispatch(cb, (uint32_t)((count + 63) / 64), 1, 1);
        // No barrier: hist is only read by the grad kernels, which run in a
        // separate command buffer after submit_and_wait (queue-idle sync).
    };
    auto mm = [&](const char* name, int xoff, int ooff, float alpha) {
        int woff = W.tm.off.at(name);
        int rows = W.tm.rows.at(name), cols = W.tm.cols.at(name);
        struct { int xoff, woff, pbase, rows, cols; float alpha; } pc =
            { xoff, woff, 0, rows, cols, alpha * W.tm.alpha.at(name) };
        vkCmdFillBuffer(cb, g_partial.buf, 0, g_partial.size, 0);
        bar();
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.mm_partial);
        push(&pc, sizeof(pc));
        vkCmdDispatch(cb, 4, (uint32_t)rows, 1);
        bar();
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.mm_reduce);
        struct { int pbase, ooff, rows; } rpc = { 0, ooff, rows };
        push(&rpc, sizeof(rpc));
        vkCmdDispatch(cb, (uint32_t)((rows + 63) / 64), 1, 1);
        bar();
        if (sl) {
            auto it = sl->capBase.find(name);
            if (it != sl->capBase.end()) {
                store(xoff, it->second.second + pos * cols, cols);
                store(ooff, it->second.first + pos * rows, rows);
            }
        }
    };
    auto rmsnorm = [&](int xoff, int ooff, const char* normname) {
        struct { int xoff, ooff, woff, dim; float eps; } pc =
            { xoff, ooff, W.normOff.at(normname), H, 1e-6f };
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.rmsnorm);
        push(&pc, sizeof(pc));
        vkCmdDispatch(cb, 1, 1, 1);
        bar();
    };
    auto add = [&](int dst, int src) {
        struct { int dst, src, dim; } pc = { dst, src, H };
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.add);
        push(&pc, sizeof(pc));
        vkCmdDispatch(cb, (uint32_t)((H + 63) / 64), 1, 1);
        bar();
    };
    auto store_cache = [&](int src, int layer, int pos, int which) {
        struct { int src, dstBase, which, count; } pc =
            { src, layer * seq * H + pos * H, which, H };
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.cstore);
        push(&pc, sizeof(pc));
        vkCmdDispatch(cb, 1, 1, 1);
        bar();
    };
    auto cap = [&](int off, const char* label) {
        if (caps) caps->push_back({ off, label });
    };

    // Embedding + position embedding
    {
        struct { int tok, pos, H, embOff, posOff; } pc =
            { tok, pos, H, W.embOff, W.posOff };
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.embed);
        push(&pc, sizeof(pc));
        vkCmdDispatch(cb, (uint32_t)((H + 63) / 64), 1, 1);
        bar();
    }
    cap(XA, "embed");

    for (int l = 0; l < L; l++) {
        char pfx[64];
        snprintf(pfx, sizeof(pfx), "layers.%d.", l);
        std::string pfx_s(pfx);

        rmsnorm(XA, XNORM, (pfx_s + "attn_norm.weight").c_str());
        mm((pfx_s + "attn.q_proj.latent_weights").c_str(), XNORM, QA, 1.0f);
        mm((pfx_s + "attn.k_proj.latent_weights").c_str(), XNORM, KA, 1.0f);
        mm((pfx_s + "attn.v_proj.latent_weights").c_str(), XNORM, VA, 1.0f);
        store_cache(KA, l, pos, 0);
        store_cache(VA, l, pos, 1);
        {
            struct { int qoff, ooff, kcbase, vcbase, head, hd, H, seq, pos; float scale; } pc =
                { QA, ATTNO, l * seq * H, l * seq * H, 0, HD, H, seq, pos,
                  1.0f / sqrtf((float)HD) };
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.attn);
            push(&pc, sizeof(pc));
            vkCmdDispatch(cb, (uint32_t)NH, 1, 1);
            bar();
        }
        mm((pfx_s + "attn.o_proj.latent_weights").c_str(), ATTNO, PROJ, 1.0f);
        add(XA, PROJ);
        if (l == 0) { cap(ATTNO, "layer0_attn"); cap(XA, "layer0_attn_resid"); }

        rmsnorm(XA, XNORM, (pfx_s + "ffn_norm.weight").c_str());
        mm((pfx_s + "ffn.gate_up_proj.latent_weights").c_str(), XNORM, FUSED, 1.0f);
        {
            struct { int base, ffn; } pc = { FUSED, FFN };
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.silu);
            push(&pc, sizeof(pc));
            vkCmdDispatch(cb, (uint32_t)((FFN + 63) / 64), 1, 1);
            bar();
        }
        mm((pfx_s + "ffn.down_proj.latent_weights").c_str(), FUSED, DOWNO, 1.0f);
        add(XA, DOWNO);
        if (l == dbg_layer || (l == 0 && dbg_layer > 0)) {
            VK_CHECK(vkEndCommandBuffer(cb));
            submit_and_wait(d, k);
            if (ba) {
                fprintf(stderr, "  L%d q :", l); for (int i = 0; i < 4; i++) fprintf(stderr, " %.4f", ba->host[QA + i]); fprintf(stderr, "\n");
                fprintf(stderr, "  L%d v :", l); for (int i = 0; i < 4; i++) fprintf(stderr, " %.4f", ba->host[VA + i]); fprintf(stderr, "\n");
                fprintf(stderr, "  L%d attn:", l); for (int i = 0; i < 4; i++) fprintf(stderr, " %.4f", ba->host[ATTNO + i]); fprintf(stderr, "\n");
                fprintf(stderr, "  L%d proj:", l); for (int i = 0; i < 4; i++) fprintf(stderr, " %.4f", ba->host[PROJ + i]); fprintf(stderr, "\n");
                fprintf(stderr, "  L%d hid :", l); for (int i = 0; i < 4; i++) fprintf(stderr, " %.4f", ba->host[FUSED + i]); fprintf(stderr, "\n");
                fprintf(stderr, "  L%d x   :", l); for (int i = 0; i < 4; i++) fprintf(stderr, " %.4f", ba->host[XA + i]); fprintf(stderr, "\n");
                fprintf(stderr, "  L%d part:", l); for (int i = 0; i < 8; i++) fprintf(stderr, " %.4f", g_partial.host[i]); fprintf(stderr, "\n");
            }
            VK_CHECK(vkBeginCommandBuffer(cb, &cbi));
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.pl, 0, 1, &d.dset, 0, nullptr);
        }
        if (l == 0) cap(XA, "layer0_ffn_resid");
    }

    // Final norm + LM head (embedding as weights)
    rmsnorm(XA, XNORM, "norm.weight");
    if (sl) store(XNORM, sl->hBase + pos * H, H);
    {
        struct { int xoff, woff, pbase, rows, cols; float alpha; } pc =
            { XNORM, W.lmOff, 0, W.lmRows, W.lmCols, 1.0f };
        vkCmdFillBuffer(cb, g_partial.buf, 0, g_partial.size, 0);
        bar();
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.mm_partial);
        push(&pc, sizeof(pc));
        vkCmdDispatch(cb, 4, (uint32_t)W.lmRows, 1);
        bar();
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.mm_reduce);
        struct { int pbase, ooff, rows; } rpc = { 0, LOGITS, W.lmRows };
        push(&rpc, sizeof(rpc));
        vkCmdDispatch(cb, (uint32_t)((W.lmRows + 63) / 64), 1, 1);
    }

    VK_CHECK(vkEndCommandBuffer(cb));
}

static void submit_and_wait(Device& d, Kernels& k) {
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &k.cmd;
    VK_CHECK(vkQueueSubmit(d.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(d.queue));
}

// Record + submit a cache-clear command buffer.
static void clear_cache(Device& d, Kernels& k, const GPUBuffer& bk, const GPUBuffer& bv) {
    VkCommandBuffer cb = k.cmd;
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cb, &cbi));
    vkCmdFillBuffer(cb, bk.buf, 0, bk.size, 0);
    vkCmdFillBuffer(cb, bv.buf, 0, bv.size, 0);
    VK_CHECK(vkEndCommandBuffer(cb));
    submit_and_wait(d, k);
}

// Record + submit one kernel, then dump 8 floats. Used by --dbg.
struct OneKernel { Device* d; Kernels* k; const Model* model; GPUBuffer* ba; };

// Record all rule-c gradients + the embedding gradient into ONE command buffer
// and submit it once (was 37 separate submits before).
static void record_grads(Device& d, Kernels& k, const SLCapture& sl,
                         const TensorMap& tm, int V, int H, int T) {
    VkCommandBuffer cb = k.cmd;
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cb, &cbi));
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.pl, 0, 1, &d.dset, 0, nullptr);
    auto push = [&](const void* data, size_t size) {
        vkCmdPushConstants(cb, d.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)size, data);
    };
    for (auto& kv : sl.gradOff) {
        const std::string& name = kv.first;
        int rows = tm.rows.at(name), cols = tm.cols.at(name);
        auto cb2 = sl.capBase.at(name);
        struct { int yBase, xBase, gradOff, rows, cols, T; } pc =
            { cb2.first, cb2.second, kv.second, rows, cols, T };
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.rulec);
        push(&pc, sizeof(pc));
        vkCmdDispatch(cb, (uint32_t)((rows + 7) / 8), (uint32_t)((cols + 7) / 8), 1);
    }
    {
        struct { int smBase, hBase, gradOff, V, Hh, T; } pc =
            { sl.smBase, sl.hBase, sl.gradE_off, V, H, T };
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k.embgrad);
        push(&pc, sizeof(pc));
        vkCmdDispatch(cb, (uint32_t)((V + 7) / 8), (uint32_t)((H + 7) / 8), 1);
    }
    VK_CHECK(vkEndCommandBuffer(cb));
    submit_and_wait(d, k);
}

static void submit1(Device& d, Kernels& k, VkPipeline p, const void* pcdata, size_t pcs, int gx) {
    VkCommandBuffer cb = k.cmd;
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cb, &cbi));
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.pl, 0, 1, &d.dset, 0, nullptr);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    vkCmdPushConstants(cb, d.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pcs, pcdata);
    vkCmdDispatch(cb, (uint32_t)gx, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cb));
    submit_and_wait(d, k);
}
static void submit2(Device& d, Kernels& k, VkPipeline p, const void* pcdata, size_t pcs, int gx, int gy) {
    VkCommandBuffer cb = k.cmd;
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cb, &cbi));
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d.pl, 0, 1, &d.dset, 0, nullptr);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    vkCmdPushConstants(cb, d.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pcs, pcdata);
    vkCmdDispatch(cb, (uint32_t)gx, (uint32_t)gy, 1);
    VK_CHECK(vkEndCommandBuffer(cb));
    submit_and_wait(d, k);
}
static void dump_act(Device& d, Kernels& k, GPUBuffer& ba, int off, const char* label) {
    float* a = ba.host + off;
    printf("  %-16s:", label);
    for (int i = 0; i < 8; i++) printf(" %.4f", a[i]);
    printf("\n");
}

static void fill_partial(Device& d, Kernels& k) {
    VkCommandBuffer cb = k.cmd;
    VkCommandBufferBeginInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cb, &cbi));
    vkCmdFillBuffer(cb, g_partial.buf, 0, g_partial.size, 0);
    VK_CHECK(vkEndCommandBuffer(cb));
    submit_and_wait(d, k);
}

// Run layer 0 only, one kernel per submit, dumping every intermediate.
static void dbg_layer0(Device& d, Kernels& k, GPUBuffer& ba, GPUBuffer& bf,
                       int tok, int pos, const Weights& W, const Model& model) {
    const int H = model.header.hidden_dim, NH = model.header.num_heads, HD = H / NH;
    const int seq = model.header.max_seq_len;
    std::string p0 = "layers.0.";
    {
        struct { int tok, pos, H, embOff, posOff; } pc = { tok, pos, H, W.embOff, W.posOff };
        submit1(d, k, k.embed, &pc, sizeof(pc), (H + 63) / 64);
        dump_act(d, k, ba, XA, "embed");
    }
    {
        struct { int xoff, ooff, woff, dim; float eps; } pc =
            { XA, XNORM, W.normOff.at(p0 + "attn_norm.weight"), H, 1e-6f };
        submit1(d, k, k.rmsnorm, &pc, sizeof(pc), 1);
        dump_act(d, k, ba, XNORM, "normed");
    }
    auto mm1 = [&](const char* name, int xoff, int ooff) {
        int woff = W.tm.off.at(name), rows = W.tm.rows.at(name), cols = W.tm.cols.at(name);
        struct { int xoff, woff, pbase, rows, cols; float alpha; } pc =
            { xoff, woff, 0, rows, cols, W.tm.alpha.at(name) };
        fill_partial(d, k);
        submit2(d, k, k.mm_partial, &pc, sizeof(pc), 4, rows);
        struct { int pbase, ooff, rows; } rpc = { 0, ooff, rows };
        submit1(d, k, k.mm_reduce, &rpc, sizeof(rpc), (rows + 63) / 64);
    };
    mm1((p0 + "attn.q_proj.latent_weights").c_str(), XNORM, QA);
    dump_act(d, k, ba, QA, "q");
    mm1((p0 + "attn.k_proj.latent_weights").c_str(), XNORM, KA);
    dump_act(d, k, ba, KA, "k");
    mm1((p0 + "attn.v_proj.latent_weights").c_str(), XNORM, VA);
    dump_act(d, k, ba, VA, "v");
    {
        struct { int src, dstBase, which, count; } pc0 = { KA, 0, 0, H }, pc1 = { VA, 0, 1, H };
        submit1(d, k, k.cstore, &pc0, sizeof(pc0), 1);
        submit1(d, k, k.cstore, &pc1, sizeof(pc1), 1);
    }
    {
        struct { int qoff, ooff, kcbase, vcbase, head, hd, H, seq, pos; float scale; } pc =
            { QA, ATTNO, 0, 0, 0, HD, H, seq, pos, 1.0f / sqrtf((float)HD) };
        submit1(d, k, k.attn, &pc, sizeof(pc), NH);
        dump_act(d, k, ba, ATTNO, "attn");
    }
    mm1((p0 + "attn.o_proj.latent_weights").c_str(), ATTNO, PROJ);
    {
        struct { int dst, src, dim; } pc = { XA, PROJ, H };
        submit1(d, k, k.add, &pc, sizeof(pc), (H + 63) / 64);
        dump_act(d, k, ba, XA, "attn_resid");
    }
    {
        struct { int xoff, ooff, woff, dim; float eps; } pc =
            { XA, XNORM, W.normOff.at(p0 + "ffn_norm.weight"), H, 1e-6f };
        submit1(d, k, k.rmsnorm, &pc, sizeof(pc), 1);
        dump_act(d, k, ba, XNORM, "ffn_normed");
    }
    mm1((p0 + "ffn.gate_up_proj.latent_weights").c_str(), XNORM, FUSED);
    dump_act(d, k, ba, FUSED, "fused");
    {
        struct { int base, ffn; } pc = { FUSED, model.header.ffn_dim };
        submit1(d, k, k.silu, &pc, sizeof(pc), (model.header.ffn_dim + 63) / 64);
        dump_act(d, k, ba, FUSED, "hidden");
    }
    mm1((p0 + "ffn.down_proj.latent_weights").c_str(), FUSED, DOWNO);
    {
        struct { int dst, src, dim; } pc = { XA, DOWNO, H };
        submit1(d, k, k.add, &pc, sizeof(pc), (H + 63) / 64);
        dump_act(d, k, ba, XA, "ffn_resid");
    }
}

// Main
static double block_ce(float* logits, const std::vector<uint16_t>& tokens,
                       size_t t, float scale, std::vector<float>& softmax_buf) {
    const int V = (int)softmax_buf.size();
    float mx = -1e30f;
    for (int i = 0; i < V; i++) { float s = logits[i] * scale; if (s > mx) mx = s; }
    double sum = 0.0;
    for (int i = 0; i < V; i++) {
        softmax_buf[i] = (float)std::exp((double)logits[i] * scale - mx);
        sum += softmax_buf[i];
    }
    int target = tokens[t + 1];
    return -std::log((double)(softmax_buf[target] / (float)(sum + 1e-12)) + 1e-12);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: vulkan_forward.exe <model.bin> <tokens.bin> --eval [maxpos]\n"
                        "   or: vulkan_forward.exe <model.bin> <tokens.bin> --bench N\n"
                        "   or: vulkan_forward.exe <model.bin> <tokens.bin> --sl <out.bin> [steps] [log_every]\n"
                        "       [save_every] [thr] [decay] [flip_every] [toggle] [--toggle-window N] [--thr-anneal RATE]\n");
        return 1;
    }
    Model model = load_model(argv[1]);
    const int H = model.header.hidden_dim;
    const int V = model.header.vocab_size;
    const int seq = model.header.max_seq_len;
    const float scale = model.sl_logit_scale;
    std::vector<uint16_t> tokens;
    {
        FILE* f = fopen(argv[2], "rb");
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        tokens.resize(sz / 2);
        fread(tokens.data(), 2, tokens.size(), f);
        fclose(f);
    }
    std::string mode = argv[3];

    Device d;
    init_vulkan(d);
    Kernels k;
    init_kernels(d, k);

    // ---- Stage weights: ternary + lm_head embedding into the same buffer ----
    std::vector<float> wt;
    TensorMap tm;
    stage_ternary(model, wt, tm);
    Weights W;
    W.tm = tm;
    {
        const auto& emb = model.fp32_weights.at("token_embedding.weight");
        W.lmOff = tm.total;
        W.lmRows = (int)emb.shape[0];
        W.lmCols = (int)emb.shape[1];
        wt.insert(wt.end(), emb.data.begin(), emb.data.end());
        tm.total += (int)emb.data.size();
        fprintf(stderr, "lm_head via embedding: %d x %d\n", W.lmRows, W.lmCols);
    }

    // ---- Stage fp32 (embeddings + norms) ----
    W.embOff = 0;
    W.posOff = (int)model.fp32_weights.at("token_embedding.weight").data.size();
    int fp_total = W.posOff + (int)model.fp32_weights.at("pos_embedding.weight").data.size();
    std::vector<std::string> norm_names;
    for (auto& kv : model.fp32_weights) {
        if (kv.first.find("norm.weight") != std::string::npos) {
            norm_names.push_back(kv.first);
            W.normOff[kv.first] = fp_total;
            fp_total += (int)kv.second.data.size();
        }
    }
    std::sort(norm_names.begin(), norm_names.end());

    // ---- GPU buffers ----
    GPUBuffer bw, bf, ba, bk, bv, bp;
    gpu_alloc(d, bw, wt.size() * 4);
    memcpy(bw.host, wt.data(), wt.size() * 4);
    gpu_alloc(d, bf, (size_t)fp_total * 4);
    {
        float* f = bf.host;
        const auto& emb = model.fp32_weights.at("token_embedding.weight").data;
        memcpy(f + W.embOff, emb.data(), emb.size() * 4);
        const auto& pos = model.fp32_weights.at("pos_embedding.weight").data;
        memcpy(f + W.posOff, pos.data(), pos.size() * 4);
        for (auto& nm : norm_names)
            memcpy(f + W.normOff[nm], model.fp32_weights.at(nm).data.data(),
                   model.fp32_weights.at(nm).data.size() * 4);
    }
    gpu_alloc(d, ba, ACT_TOTAL * 4);
    gpu_alloc(d, bk, (size_t)model.header.num_layers * seq * H * 4);
    gpu_alloc(d, bv, (size_t)model.header.num_layers * seq * H * 4);
    gpu_alloc(d, bp, (size_t)(4 * 8192) * 4);   // max rows = lm_head 8192
    g_partial = bp;

    // ---- SL capture layout (allocated unconditionally; --sl uses it) ----
    SLCapture sl;
    sl.Tmax = model.sl_block_size;
    {
        std::vector<std::string> ks;
        for (auto& kv : model.ternary_weights) ks.push_back(kv.first);
        std::sort(ks.begin(), ks.end());
        for (auto& name : ks) {
            int rows = tm.rows.at(name), cols = tm.cols.at(name);
            sl.gradOff[name] = sl.gradFloats;
            sl.gradFloats += rows * cols;
            sl.capBase[name] = { sl.histFloats, 0 };
            sl.histFloats += sl.Tmax * rows;
            sl.capBase[name].second = sl.histFloats;
            sl.histFloats += sl.Tmax * cols;
        }
        sl.smBase = sl.histFloats;
        sl.histFloats += sl.Tmax * V;
        sl.hBase = sl.histFloats;
        sl.histFloats += sl.Tmax * H;
        sl.gradE_off = sl.gradFloats;
        sl.gradFloats += V * H;
        fprintf(stderr, "SL layout: hist=%d floats (%.1f MB), grad=%d floats (%.1f MB)\n",
                sl.histFloats, sl.histFloats * 4.0 / 1e6,
                sl.gradFloats, sl.gradFloats * 4.0 / 1e6);
    }
    GPUBuffer bh, bg;
    gpu_alloc(d, bh, (size_t)sl.histFloats * 4);
    gpu_alloc(d, bg, (size_t)sl.gradFloats * 4);

    // ---- Descriptor set ----
    VkDescriptorBufferInfo dbi[8] = {};
    VkBuffer bufs[8] = { bw.buf, bf.buf, ba.buf, bk.buf, bv.buf, bp.buf, bh.buf, bg.buf };
    for (int i = 0; i < 8; i++) {
        dbi[i].buffer = bufs[i];
        dbi[i].range = VK_WHOLE_SIZE;
    }
    VkWriteDescriptorSet wds[8] = {};
    for (int i = 0; i < 8; i++) {
        wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds[i].dstSet = d.dset;
        wds[i].dstBinding = (uint32_t)i;
        wds[i].descriptorCount = 1;
        wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(d.dev, 8, wds, 0, nullptr);

    std::vector<float> softmax_buf(V);

    if (mode == "--eval") {
        size_t limit = tokens.size() - 1;
        if (argc > 4) {
            long n = atol(argv[4]);
            if (n > 0 && (size_t)n < limit) limit = (size_t)n;
        }
        clear_cache(d, k, bk, bv);
        double loss = 0.0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t t = 0; t < limit; t++) {
            if (t > 0 && t % (size_t)seq == 0) clear_cache(d, k, bk, bv);
            record_forward(d, k, tokens[t], (int)(t % seq), W, model);
            submit_and_wait(d, k);
            loss += block_ce(ba.host + LOGITS, tokens, t, scale, softmax_buf);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fprintf(stderr, "Eval done. avg CE %.4f | PPL %.4f | %zu positions | %.2f ms/token\n",
                loss / limit, std::exp(loss / limit), limit, ms / limit);
    } else if (mode == "--bench") {
        int nblocks = atoi(argv[4]);
        double ms_total = 0.0;
        for (int b = 0; b < nblocks; b++) {
            size_t start = ((size_t)b * 128) % (tokens.size() - 1);
            size_t end = (std::min)(tokens.size() - 1, start + 128);
            clear_cache(d, k, bk, bv);
            double block_loss = 0.0;
            auto t0 = std::chrono::high_resolution_clock::now();
            for (size_t t = start; t < end; t++) {
                record_forward(d, k, tokens[t], (int)(t - start), W, model);
                submit_and_wait(d, k);
                block_loss += block_ce(ba.host + LOGITS, tokens, t, scale, softmax_buf);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            ms_total += ms;
            fprintf(stderr, "block %d | CE %.4f | %.1f ms\n", b + 1,
                    block_loss / 128.0, ms);
        }
        fprintf(stderr, "avg %.2f ms/block | %.3f ms/token\n",
                ms_total / nblocks, ms_total / nblocks / 128.0);
    } else if (mode == "--dbg") {
        size_t n = (argc > 4) ? (size_t)atoi(argv[4]) : 1;
        clear_cache(d, k, bk, bv);
        for (size_t t = 0; t < n; t++) {
            printf("pos %zu tok %d\n", t, (int)tokens[t]);
            if (t == 0) {
                dbg_layer0(d, k, ba, bf, (int)tokens[t], (int)(t % seq), W, model);
                printf("  --- full forward ---\n");
            }
            std::vector<std::pair<int, const char*>> caps;
            record_forward(d, k, tokens[t], (int)(t % seq), W, model, &caps, 1, &ba);
            submit_and_wait(d, k);
            dump_act(d, k, ba, QA, "L5_q");
            dump_act(d, k, ba, KA, "L5_k");
            dump_act(d, k, ba, VA, "L5_v");
            dump_act(d, k, ba, ATTNO, "L5_attn");
            dump_act(d, k, ba, PROJ, "L5_proj");
            dump_act(d, k, ba, FUSED, "L5_hidden");
            dump_act(d, k, ba, XA, "L5_final_x");
            dump_act(d, k, ba, XNORM, "L5_normed");
            float* lg = ba.host + LOGITS;
            printf("  logits:");
            for (int i = 0; i < 12; i++) printf(" %.4f", lg[i]);
            printf("\n");
        }
    } else if (mode == "--sl") {
        if (argc < 5) { fprintf(stderr, "--sl needs <out.bin>\n"); return 1; }
        const char* out_path = argv[4];
        int steps      = (argc > 5) ? atoi(argv[5]) : 200;
        int log_every  = (argc > 6) ? atoi(argv[6]) : 50;
        int save_every = (argc > 7) ? atoi(argv[7]) : 100;
        float thr_override   = (argc > 8) ? (float)atof(argv[8]) : 0.0f;
        float decay_override = (argc > 9) ? (float)atof(argv[9]) : 0.0f;
        int every_override   = (argc > 10) ? atoi(argv[10]) : 0;
        int toggle_override  = (argc > 11) ? atoi(argv[11]) : -1;
        int toggle_window = 0;
        float thr_anneal = 0.0f;
        for (int i = 1; i + 1 < argc; i++) {
            if (strcmp(argv[i], "--toggle-window") == 0) toggle_window = atoi(argv[i + 1]);
            if (strcmp(argv[i], "--thr-anneal") == 0) thr_anneal = (float)atof(argv[i + 1]);
        }
        if (model.is_mla) {
            fprintf(stderr, "ERROR: self-learning is only supported for standard attention.\n");
            return 1;
        }
        if (model.sl_rule != 0) {
            fprintf(stderr, "ERROR: only rule 'c' (predictive coding) is implemented so far (sl_rule=%d).\n",
                    model.sl_rule);
            return 1;
        }
        const int block = model.sl_block_size;
        const float thr = thr_override > 0.0f ? thr_override : model.sl_threshold;
        const float decay = decay_override > 0.0f ? decay_override : model.sl_acc_decay;
        const int flip_every = every_override > 0 ? every_override : model.sl_flip_every_n;
        const bool toggle = toggle_override >= 0 ? (toggle_override != 0) : (model.sl_toggle != 0);
        const float lr_emb = model.sl_lr_embedding;
        const float wd_emb = model.sl_wd_embedding;
        float* emb = model.fp32_weights.at("token_embedding.weight").data.data();

        // Churn tracking (same bookkeeping as selflearn.cpp).
        std::vector<TernaryWeightXNOR*> wlist;
        std::vector<std::vector<uint32_t>> hists;
        std::vector<std::string> wnames;
        long long total_w = 0;
        for (auto& kv : model.ternary_weights) {
            wlist.push_back(&kv.second);
            wnames.push_back(kv.first);
            hists.emplace_back(kv.second.floats.size(), 0u);
            total_w += (long long)kv.second.floats.size();
        }

        fprintf(stderr, "SL: block=%d thr=%.1f decay=%.3f flipEvery=%d toggle=%d "
                        "lrEmb=%.1e wdEmb=%.2f%s%s\n",
                block, thr, decay, flip_every, toggle ? 1 : 0, lr_emb, wd_emb,
                (toggle_window > 0 ? " | toggle-window " + std::to_string(toggle_window) : "").c_str(),
                (thr_anneal > 0.0f ? " | thr-anneal +" + std::to_string(thr_anneal) + "/pass" : "").c_str());

        double ms_total = 0.0;
        // Host staging copy of the gradient buffer: the GPU coherent read path
        // is ~100ns/element, so bulk-copy once (memcpy) instead of scattered
        // reads inside the feed/SGD loops.
        std::vector<float> grads_host((size_t)sl.gradFloats);
        for (int step = 0; step < steps; step++) {
            auto ts = std::chrono::high_resolution_clock::now();
            memset(bg.host, 0, (size_t)sl.gradFloats * 4);
            clear_cache(d, k, bk, bv);
            size_t start = ((size_t)step * block) % (tokens.size() - 1);
            size_t end = (std::min)(tokens.size() - 1, start + block);
            size_t T = end - start;
            if (T < 2) { fprintf(stderr, "token stream too short\n"); return 1; }

            double block_loss = 0.0;
            int valid_positions = 0;
            for (size_t t = start; t < end; t++) {
                int pos = (int)(t - start);
                record_forward(d, k, tokens[t], pos, W, model, nullptr, -1, &ba, &sl);
                submit_and_wait(d, k);
                if (pos == 0) {
                    // C++ selflearn skips the first block position entirely:
                    // no CE, no rule-c, and zero embedding gradient.
                    memset(bh.host + sl.smBase, 0, (size_t)V * 4);
                    continue;
                }
                // Softmax of scaled logits; fold (softmax - onehot) into the
                // history slot so embgrad only has to do a dot product.
                float* lg = ba.host + LOGITS;
                float mx = -1e30f;
                for (int i = 0; i < V; i++) { float s = lg[i] * scale; if (s > mx) mx = s; }
                double sum = 0.0;
                for (int i = 0; i < V; i++) {
                    softmax_buf[i] = (float)std::exp((double)lg[i] * scale - mx);
                    sum += softmax_buf[i];
                }
                if (sum > 0) for (int i = 0; i < V; i++) softmax_buf[i] /= (float)sum;
                int target = tokens[t + 1];
                block_loss += -std::log((double)(softmax_buf[target] + 1e-12));
                valid_positions++;
                float* sm = bh.host + sl.smBase + (size_t)pos * V;
                for (int i = 0; i < V; i++) sm[i] = softmax_buf[i] - (i == target ? 1.0f : 0.0f);
            }

            // Rule-c gradients + embedding gradient, one command buffer.
            record_grads(d, k, sl, tm, V, H, (int)T);
            memcpy(grads_host.data(), bg.host, (size_t)sl.gradFloats * 4);

            // Feed deltas into accumulators (rule 'c'), reading grads from the
            // host staging copy (same math as sl_feed_predictive).
            for (size_t wi = 0; wi < wlist.size(); wi++) {
                auto git = sl.gradOff.find(wnames[wi]);
                if (git == sl.gradOff.end()) continue;
                TernaryWeightXNOR& w = *wlist[wi];
                const float* g = grads_host.data() + git->second;
                const size_t n = (size_t)w.rows * w.cols;
                for (size_t j = 0; j < n; j++) {
                    float gg = g[j];
                    float d = (gg > 0.0f) ? -1.0f : ((gg < 0.0f) ? 1.0f : 0.0f);
                    w.accumulator[j] = w.accumulator[j] * decay + d;
                }
            }

            // Embedding local SGD: per-row clip + decoupled WD.
            {
                const float* ge0 = grads_host.data() + sl.gradE_off;
                for (int v = 0; v < V; v++) {
                    const float* ge = ge0 + (size_t)v * H;
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
                memcpy(bf.host + W.embOff, emb, (size_t)V * H * 4);
                memcpy(bw.host + W.lmOff, emb, (size_t)V * H * 4);
            }

            // Bit flips every N steps (same bookkeeping as selflearn.cpp).
            long long total_flips = 0;
            long long real_changes = 0;
            if (flip_every > 0 && (step + 1) % flip_every == 0) {
                long long acc_over20 = 0, acc_over15 = 0;
                double acc_max = 0;
                for (size_t wi = 0; wi < wlist.size(); wi++) {
                    TernaryWeightXNOR& w = *wlist[wi];
                    for (size_t i = 0; i < w.accumulator.size(); i++) {
                        double aa = fabs((double)w.accumulator[i]);
                        if (aa > acc_max) acc_max = aa;
                        if (aa > 20.0) acc_over20++;
                        else if (aa > 15.0) acc_over15++;
                    }
                }
                fprintf(stderr, "  [acc stats] max=%.4f >20=%lld in(15,20]=%lld\n",
                        acc_max, acc_over20, acc_over15);
                const float eff_thr = thr_anneal > 0.0f
                        ? thr + thr_anneal * ((float)((step + 1) / flip_every) - 1.0f)
                        : thr;
                const bool eff_toggle = toggle && (toggle_window <= 0 || step + 1 <= toggle_window);
                for (size_t wi = 0; wi < wlist.size(); wi++) {
                    TernaryWeightXNOR& w = *wlist[wi];
                    auto& hist = hists[wi];
                    const size_t n = w.floats.size();
                    std::vector<float> before(n);
                    memcpy(before.data(), w.floats.data(), n * sizeof(float));
                    total_flips += apply_bit_flips(w, eff_thr, eff_toggle, 3.0f, 0.0f);
                    const float* f = w.floats.data();
                    for (size_t i = 0; i < n; i++)
                        if (f[i] != before[i]) { hist[i]++; real_changes++; }
                }
                long long ever = 0, m2 = 0, m4 = 0, m8 = 0;
                for (auto& h : hists)
                    for (auto c : h) {
                        if (c) ever++;
                        if (c >= 2) m2++;
                        if (c >= 4) m4++;
                        if (c >= 8) m8++;
                    }
                double tot = (double)total_w;
                fprintf(stderr, "  flips=%lld (real changes=%lld, %.1f%% no-op, eff_thr=%.1f) | "
                                "churn: ever=%.2f%% >=2=%.2f%% >=4=%.2f%% >=8=%.2f%%\n",
                        total_flips, real_changes,
                        total_flips > 0 ? 100.0 * (1.0 - (double)real_changes / total_flips) : 0.0,
                        eff_thr,
                        ever / tot * 100.0, m2 / tot * 100.0, m4 / tot * 100.0, m8 / tot * 100.0);
                if (toggle_window > 0 && step + 1 == toggle_window) {
                    size_t nacc = 0;
                    for (size_t wi = 0; wi < wlist.size(); wi++)
                        nacc += wlist[wi]->accumulator.size();
                    for (size_t wi = 0; wi < wlist.size(); wi++) {
                        auto& accs = wlist[wi]->accumulator;
                        std::fill(accs.begin(), accs.end(), 0.0f);
                    }
                    fprintf(stderr, "  >> toggle window ended at block %d: %zu accumulators zeroed\n",
                            toggle_window, nacc);
                }
                // Refresh GPU weights after flips (floats changed in place).
                for (auto& name : wnames) {
                    const auto& w = model.ternary_weights.at(name);
                    memcpy(bw.host + tm.off.at(name), w.floats.data(), w.floats.size() * 4);
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
    }

    return 0;
}
