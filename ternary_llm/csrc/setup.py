"""PyTorch C++ extension setup for ternary_ops.

Build: python csrc/setup.py build_ext --inplace
Or during import (auto-build): from csrc import ternary_ops
"""
import sys
from setuptools import setup, Extension
from torch.utils import cpp_extension

avx2_flag = "/arch:AVX2" if sys.platform == "win32" else "-mavx2 -mfma"

setup(
    name="ternary_ops",
    ext_modules=[
        cpp_extension.CppExtension(
            "ternary_ops",
            ["csrc/ternary_ops_avx2.cpp"],
            extra_compile_args=[avx2_flag],
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
