"""Build C++ ternary_ops extensions for AVX2 and AVX-512.

Requirements: Visual Studio 2022 with MSVC, PyTorch
Usage: python build_cpp.py
"""
import subprocess, sys, os

# Activate VS dev environment (search multiple locations)
_vcvars_paths = [
    os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
                 "Microsoft Visual Studio", "2022", "Community", "VC",
                 "Auxiliary", "Build", "vcvars64.bat"),
    os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"),
                 "Microsoft Visual Studio", "2022", "Community", "VC",
                 "Auxiliary", "Build", "vcvars64.bat"),
    os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"),
                 "Microsoft Visual Studio", "2022", "Professional", "VC",
                 "Auxiliary", "Build", "vcvars64.bat"),
    os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"),
                 "Microsoft Visual Studio", "2022", "Professional", "VC",
                 "Auxiliary", "Build", "vcvars64.bat"),
]
_vcvars = next((p for p in _vcvars_paths if os.path.exists(p)), None)
if _vcvars is None:
    # Try vswhere as fallback
    try:
        vswhere = subprocess.run(["vswhere", "-latest", "-property", "installationPath"],
                                 capture_output=True, text=True)
        if vswhere.returncode == 0:
            vs_path = vswhere.stdout.strip()
            candidate = os.path.join(vs_path, "VC", "Auxiliary", "Build", "vcvars64.bat")
            if os.path.exists(candidate):
                _vcvars = candidate
    except FileNotFoundError:
        pass

if _vcvars is None:
    print("ERROR: Visual Studio not found. Cannot build C++ extensions.")
    sys.exit(1)

result = subprocess.run(
    f'cmd /c "call \"{_vcvars}\" >nul 2>nul && set"',
    capture_output=True, text=True, shell=True
)
for line in result.stdout.splitlines():
    if '=' in line:
        k, v = line.split('=', 1)
        os.environ[k] = v

import torch
from torch.utils.cpp_extension import load

base_dir = os.path.join(os.path.dirname(__file__), "ternary_llm", "csrc")

# Platform-appropriate SIMD flags
if sys.platform == "win32":
    avx2_flag = "/arch:AVX2"
    avx512_flag = "/arch:AVX512"
else:
    avx2_flag = "-mavx2 -mfma"
    avx512_flag = "-mavx512f -mavx512bw"

# Build AVX2 version (baseline, always available)
print("\nBuilding ternary_ops (AVX2)...")
ops_avx2 = load(
    name="ternary_ops",
    sources=[os.path.join(base_dir, "ternary_ops_avx2.cpp")],
    extra_cflags=[avx2_flag],
    verbose=False,
)
# Validate
n = 1000
w = torch.randint(0, 3, (n,), dtype=torch.uint8).float() - 1.0
packed = ops_avx2.pack_ternary(w)
w2 = ops_avx2.unpack_ternary(packed, [n])
assert (w - w2).abs().max().item() < 1e-5, "AVX2 pack/unpack failed"
print("AVX2: OK")

# Build AVX-512 version (loaded at runtime if CPU supports it)
avx512_src = os.path.join(base_dir, "ternary_ops_avx512.cpp")
if os.path.exists(avx512_src):
    print("\nBuilding ternary_ops_avx512...")
    try:
        ops_avx512 = load(
            name="ternary_ops_avx512",
            sources=[avx512_src],
            extra_cflags=[avx512_flag],
            verbose=False,
        )
        # Validate
        packed2 = ops_avx512.pack_ternary(w)
        w3 = ops_avx512.unpack_ternary(packed2, [n])
        assert (w - w3).abs().max().item() < 1e-5, "AVX-512 pack/unpack failed"
        print("AVX-512: OK")
    except Exception as e:
        print(f"AVX-512: skipped ({e})")
else:
    print("\n  AVX-512 source not found, skipping")

print("\nDone.")
