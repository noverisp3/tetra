"""Benchmark 500m stochastic model on CPU."""
import sys, os, time, torch
sys.path.insert(0, r'C:\Users\Noveris\Desktop\Oxide')

# Force C++ ext build
from torch.utils.cpp_extension import load
load('ternary_ops', ['ternary_llm/csrc/ternary_ops_avx2.cpp'],
     extra_cflags=['/arch:AVX2'], verbose=False)

from ternary_llm.quantization import _has_cpp
from ternary_llm.transformer import StochasticTransformerModel

print(f"C++ loaded: {_has_cpp}")

device = torch.device('cpu')
torch.set_num_threads(4)

print("Creating 500m model...")
t0 = time.perf_counter()
model = StochasticTransformerModel(
    vocab_size=8192, hidden_dim=2560, num_layers=6,
    num_heads=40, ffn_dim=6826, max_seq_len=1024, scale=1.0)
model = model.to(device).train()
print(f"  Created in {time.perf_counter()-t0:.1f}s")

total = sum(p.numel() for p in model.parameters())
ternary = sum(p.numel() for n,p in model.named_buffers() if 'packed' in n) * 2
print(f"  Total params: {total:,} ({total/1e6:.0f}M)")
print(f"  Ternary params: {ternary:,} ({ternary/1e6:.0f}M, {ternary/8/1024:.0f}KB packed)")

batch, seq = 2, 512  # half context to manage memory
print(f"  Batch: {batch}, Seq: {seq}")
x = torch.randint(0, 8192, (batch, seq))
y = torch.randint(0, 8192, (batch, seq))

# Warmup (1 step)
print("\nWarmup...")
_, loss, _ = model(x, y)
loss.backward()
for p in model.parameters():
    if p.grad is not None: p.grad.zero_()
# Zero accumulators
for name, buf in model.named_buffers():
    if 'accumulator' in name: buf.zero_()
print("  Warmup done")

# Benchmark
print("\nBenchmark (3 steps):")
fwd_times, bwd_times = [], []
for step in range(3):
    t0 = time.perf_counter()
    _, loss, _ = model(x, y)
    t1 = time.perf_counter()
    loss.backward()
    t2 = time.perf_counter()
    fwd_times.append(t1-t0)
    bwd_times.append(t2-t1)
    ram = __import__('psutil').Process().memory_info().rss / 1024**3
    print(f"  Step {step+1}: fwd={t1-t0:.2f}s  bwd={t2-t1:.2f}s  total={t2-t0:.2f}s  RAM={ram:.1f}GB")

af = sum(fwd_times)/len(fwd_times)
ab = sum(bwd_times)/len(bwd_times)
print(f"\nAvg: fwd={af:.2f}s  bwd={ab:.2f}s  total={af+ab:.2f}s")
print(f"Estimated 10K steps: {(af+ab)*10000/3600:.1f} hours")
