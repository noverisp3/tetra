"""Build C++ ternary_ops extensions for AVX2 and AVX-512.

Requirements: Visual Studio 2022 with MSVC, PyTorch
Usage: python build_cpp.py
"""
import subprocess, sys, os

# Activate VS 2022 dev environment
vs_vcvars = r"C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"
if os.path.exists(vs_vcvars):
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

base_dir = os.path.join(os.path.dirname(__file__), "ternary_llm", "csrc")

# Build AVX2 version (baseline, always available)
print("\nBuilding ternary_ops (AVX2)...")
ops_avx2 = load(
    name="ternary_ops",
    sources=[os.path.join(base_dir, "ternary_ops_avx2.cpp")],
    extra_cflags=["/arch:AVX2"],
    verbose=False,
)
# Validate
n = 1000
w = torch.randint(0, 3, (n,), dtype=torch.uint8).float() - 1.0
packed = ops_avx2.pack_ternary(w)
w2 = ops_avx2.unpack_ternary(packed, [n])
assert (w - w2).abs().max().item() < 1e-5, "AVX2 pack/unpack failed"
print("  AVX2: OK")

# Build AVX-512 version (loaded at runtime if CPU supports it)
avx512_src = os.path.join(base_dir, "ternary_ops_avx512.cpp")
if os.path.exists(avx512_src):
    print("\nBuilding ternary_ops_avx512...")
    try:
        ops_avx512 = load(
            name="ternary_ops_avx512",
            sources=[avx512_src],
            extra_cflags=["/arch:AVX512"],
            verbose=False,
        )
        # Validate
        packed2 = ops_avx512.pack_ternary(w)
        w3 = ops_avx512.unpack_ternary(packed2, [n])
        assert (w - w3).abs().max().item() < 1e-5, "AVX-512 pack/unpack failed"
        print("  AVX-512: OK")
    except Exception as e:
        print(f"  AVX-512: skipped ({e})")
else:
    print("\n  AVX-512 source not found, skipping")

print("\nDone.")
