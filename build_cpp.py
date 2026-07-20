"""Build the C++ ternary_ops extension.

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

print("Building ternary_ops...")
ternary_ops = load(
    name="ternary_ops",
    sources=[r"ternary_llm\csrc\ternary_ops.cpp"],
    extra_cflags=["/arch:AVX2"],
    verbose=True,
)

# Quick validation
n = 1000
w = torch.randint(0, 3, (n,), dtype=torch.uint8).float() - 1.0
packed = ternary_ops.pack_ternary(w)
w2 = ternary_ops.unpack_ternary(packed, [n])
assert (w - w2).abs().max().item() < 1e-5, "pack/unpack failed"
print("✓ Build OK, validation passed")
