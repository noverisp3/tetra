"""Tetra — Pure Ternary LLM with {-1, 0, 1} weights.

Architecture: BitNet b1.58-style ternary transformer decoder
- Ternary weights via absmean quantization + STE
- XNOR+popcount inference (C++ SIMD ready)
- Weight tying, RMSNorm, SwiGLU FFN

For inference: export ternary weights, run with XNOR+popcount kernels.
"""

from .quantization import TernaryQuantizer
from .layers import TernaryLinear, RMSNorm
from .attention import TernaryMultiHeadAttention
from .ffn import TernaryFFN
from .transformer import TernaryTransformerBlock, TernaryTransformerModel
from .int8 import Int8Quantizer, int8_quantize

__all__ = [
    "TernaryQuantizer",
    "Int8Quantizer",
    "TernaryLinear",
    "RMSNorm",
    "TernaryMultiHeadAttention",
    "TernaryFFN",
    "TernaryTransformerBlock",
    "TernaryTransformerModel",
]
