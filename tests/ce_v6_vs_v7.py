"""Phase 3: measure block CE under v6 vs v7 interpretation of a binary.

Loads a v6/v7 TETR binary (verify_export's loader), decodes the ternary
weights either with the side-channel blob (v7: code 11 -> +-2) or without
(v6: code 11 -> 0), applies per-row/per-group alphas like the C++ runtime,
then runs 128-token blocks over a fixed TinyStories window (torch reference
of the C++ forward).

Usage: python tests/ce_v6_vs_v7.py <model.bin> [--v6] [--blocks N] [--start-token N]
"""
import argparse
import struct
import sys

from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))
from verify_export import load_binary_model  # noqa: E402

torch.set_num_threads(4)

LUT_V6 = np.array([-1.0, 0.0, 1.0, 0.0])  # code 11 -> 0 (no side channel)


def parse_ternary_meta(path):
    """Return {name: (rows, cols, gs, alphas, packed_bytes)} per ternary entry."""
    d = np.fromfile(path, dtype=np.uint8)
    version = int(np.frombuffer(d[4:8], dtype="<u4")[0])
    out = {}
    pos = 64
    while True:
        nlen = int(np.frombuffer(d[pos:pos + 4], dtype="<u4")[0])
        if nlen > 1024 or nlen == 0:
            break
        name = bytes(d[pos + 4:pos + 4 + nlen]).decode()
        if not name.endswith("latent_weights"):
            break
        rows, cols = struct.unpack_from("<HH", d.tobytes(), pos + 4 + nlen)
        p = pos + 4 + nlen + 4
        gs, na = struct.unpack_from("<HH", d.tobytes(), p)
        p += 4
        alphas = np.frombuffer(d[p:p + na * 4].tobytes(), dtype="<f4") if na else np.zeros(0)
        p += na * 4
        psz = (rows * cols + 3) // 4
        packed = d[p:p + psz]
        p += psz
        if version >= 6:
            p += rows * cols * 4  # accumulator
        if version >= 7:
            n_out = int(np.frombuffer(d[p:p + 4], dtype="<u4")[0])
            p += 4 + (n_out + 7) // 8  # blob
        out[name] = (rows, cols, gs, alphas, packed)
        pos = p
    return out, version


def decode_packed(packed, n, codes_lut):
    flat = np.zeros(n, dtype=np.float32)
    nb = 0
    for byte in packed:
        for i in range(4):
            if nb >= n:
                break
            flat[nb] = codes_lut[(byte >> (6 - i * 2)) & 3]
            nb += 1
    return flat


def apply_alphas(w, gs, alphas):
    rows, cols = w.shape
    if gs > 0:
        ng = (cols + gs - 1) // gs
        out = np.zeros_like(w)
        for r in range(rows):
            for g in range(ng):
                s = g * gs
                e = min(cols, s + gs)
                out[r, s:e] = w[r, s:e] * alphas[r * ng + g]
        return out
    if alphas.size:
        return w * alphas.reshape(-1, 1)
    return w


class TorchModel:
    def __init__(self, tensors, V, H, L, NH, FFN, seq):
        self.V, self.H, self.L, self.NH, self.FFN, self.seq = V, H, L, NH, FFN, seq
        self.HD = H // NH
        self.emb = torch.tensor(tensors["token_embedding.weight"], dtype=torch.float32)
        self.head = (torch.tensor(tensors["lm_head.weight"], dtype=torch.float32)
                     if "lm_head.weight" in tensors else self.emb)
        self.pos = torch.tensor(tensors["pos_embedding.weight"], dtype=torch.float32)
        self.ternary = {k: torch.tensor(v, dtype=torch.float32)
                        for k, v in tensors.items() if k.endswith("latent_weights")}
        self.norms = {k: torch.tensor(v, dtype=torch.float32)
                      for k, v in tensors.items() if k.endswith("norm.weight")}
        self.eps = 1e-6

    def forward(self, tok, cache):
        H, L, NH, HD, FFN = self.H, self.L, self.NH, self.HD, self.FFN
        x = self.emb[tok] + self.pos[cache["pos"]]
        for l in range(L):
            p = f"layers.{l}."
            normed = self.rmsnorm(x, self.norms[p + "attn_norm.weight"])
            q = self.ternary[p + "attn.q_proj.latent_weights"] @ normed
            k = self.ternary[p + "attn.k_proj.latent_weights"] @ normed
            v = self.ternary[p + "attn.v_proj.latent_weights"] @ normed
            cache["k"][l][cache["pos"]] = k
            cache["v"][l][cache["pos"]] = v
            attn_out = torch.zeros(H)
            for head in range(NH):
                qh = q[head * HD:(head + 1) * HD]
                kh = cache["k"][l][:cache["pos"] + 1, head * HD:(head + 1) * HD]
                vh = cache["v"][l][:cache["pos"] + 1, head * HD:(head + 1) * HD]
                s = (kh @ qh) / (HD ** 0.5)
                s = s.clamp(-80.0, 80.0)
                s = torch.softmax(s, dim=0)
                attn_out[head * HD:(head + 1) * HD] = s @ vh
            x = x + self.ternary[p + "attn.o_proj.latent_weights"] @ attn_out
            normed = self.rmsnorm(x, self.norms[p + "ffn_norm.weight"])
            fused = self.ternary[p + "ffn.gate_up_proj.latent_weights"] @ normed
            gate, up = fused[:FFN], fused[FFN:]
            hidden = torch.nn.functional.silu(gate) * up
            x = x + self.ternary[p + "ffn.down_proj.latent_weights"] @ hidden
        x = self.rmsnorm(x, self.norms["norm.weight"])
        logits = self.head @ x
        cache["pos"] += 1
        return logits

    def rmsnorm(self, x, w):
        rms = torch.sqrt(x.pow(2).mean() + self.eps)
        return (x / rms) * w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", default="D:/Tetra/tinydata/tinystories.bin")
    ap.add_argument("--v6", action="store_true",
                    help="decode code 11 as 0 (old reader, no blob)")
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--start-token", type=int, default=1000000,
                    help="fixed offset into tinystories.bin (deterministic window)")
    ap.add_argument("--logits-scale", type=float, default=0.0625)
    args = ap.parse_args()

    tensors, header_info = load_binary_model(args.model)
    V = header_info["vocab_size"]
    H = header_info["hidden_dim"]
    L = header_info["num_layers"]
    NH = header_info["num_heads"]
    FFN = header_info["ffn_dim"]
    seq = header_info["max_seq_len"]

    meta, version = parse_ternary_meta(args.model)
    for name, (rows, cols, gs, alphas, packed) in meta.items():
        if args.v6:
            w = decode_packed(packed, rows * cols, LUT_V6).reshape(rows, cols)
        else:
            w = tensors[name]
        tensors[name] = apply_alphas(w, gs, alphas)
    print(f"version={version} mode={'v6 (code11->0)' if args.v6 else 'v7 (blob)'}")

    model = TorchModel(tensors, V, H, L, NH, FFN, seq)
    toks = np.fromfile(args.tokens, dtype="<u2")

    start = args.start_token
    total_ce, total_valid = 0.0, 0
    for b in range(args.blocks):
        cache = {
            "k": [torch.zeros(seq, H) for _ in range(L)],
            "v": [torch.zeros(seq, H) for _ in range(L)],
            "pos": 0,
        }
        s = start + b * 128
        ce, valid = 0.0, 0
        with torch.no_grad():
            for t in range(s, min(len(toks) - 1, s + 128)):
                logits = model.forward(int(toks[t]), cache)
                logits = logits * args.logits_scale
                ce += torch.nn.functional.cross_entropy(
                    logits.unsqueeze(0), torch.tensor([int(toks[t + 1])])).item()
                valid += 1
        print(f"  block {b}: CE {ce / valid:.4f} (pos {s}..{s + valid})")
        total_ce += ce
        total_valid += valid
    print(f"mean block CE: {total_ce / total_valid:.4f} over {total_valid} positions")


if __name__ == "__main__":
    main()
