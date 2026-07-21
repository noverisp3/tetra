"""Benchmark AVX2 vs AVX-512 for ternary ops."""
import sys, os, time

# Activate VS 2022 dev environment
vs_vcvars = r"C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"
if os.path.exists(vs_vcvars):
    import subprocess
    result = subprocess.run(
        f'cmd /c "call \"{vs_vcvars}\" >nul 2>nul && set"',
        capture_output=True, text=True, shell=True
    )
    for line in result.stdout.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            os.environ[k] = v

import torch
from torch.utils.cpp_extension import load

base = r"C:\Users\Noveris\Desktop\Oxide\ternary_llm\csrc"

print("Loading AVX2...")
avx2 = load('ternary_ops', [os.path.join(base, 'ternary_ops_avx2.cpp')],
            extra_cflags=['/arch:AVX2'], verbose=False)

print("Loading AVX-512...")
avx512 = load('ternary_ops_avx512', [os.path.join(base, 'ternary_ops_avx512.cpp')],
              extra_cflags=['/arch:AVX512'], verbose=False)

N = 1000000
w = torch.randint(0, 3, (N,), dtype=torch.uint8).float() - 1.0
shape = [N]

# Warmup
for mod in [avx2, avx512]:
    _ = mod.unpack_ternary(mod.pack_ternary(w), shape)

# Benchmark pack/unpack
for name, mod in [('AVX2', avx2), ('AVX-512', avx512)]:
    t0 = time.perf_counter()
    for _ in range(100):
        packed = mod.pack_ternary(w)
    t1 = time.perf_counter()
    unpacked = mod.unpack_ternary(packed, shape)
    t2 = time.perf_counter()
    err = (w - unpacked).abs().max().item()
    print(f'\n{name}:')
    print(f'  pack   = {(t1-t0)/100*1e6:.1f} us')
    print(f'  unpack = {(t2-t1)/100*1e6:.1f} us')
    print(f'  maxerr = {err}')

# Benchmark matmul
M, K, Nmat = 128, 2048, 512
x = torch.randn(M, K)
w_packed = avx2.pack_ternary(torch.randn(Nmat, K))
scale = 0.5

for name, mod in [('AVX2', avx2), ('AVX-512', avx512)]:
    t0 = time.perf_counter()
    for _ in range(50):
        _ = mod.ternary_matmul(x, w_packed, Nmat, K, scale)
    t1 = time.perf_counter()
    t = (t1 - t0) / 50
    gflops = M * K * Nmat * 2 / t / 1e9
    print(f'\n{name} matmul ({M}x{K}x{Nmat}):')
    print(f'  time   = {t*1000:.2f} ms')
    print(f'  perf   = {gflops:.1f} GFLOPS')
