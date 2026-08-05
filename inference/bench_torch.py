"""Pure-PyTorch baseline for the Tetra self-learning runtime.

Loads a v6 .bin export and mirrors the C++ forward pass (tetra.h) with plain
torch ops. Reports per-block timing (128-token decode block) and block CE on
the eval slice, so it can be compared 1:1 against:
  - C++ AVX2  : selflearn.exe --eval ...
  - Vulkan    : vulkan_forward.exe --eval ...

Usage:
  python inference/bench_torch.py checkpoints_discrete_c3/exp_tog_s0_zacc.bin
"""

import argparse
import struct
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)


def parse_v6(path):
    d = np.fromfile(path, dtype=np.uint8)
    magic, ver, V, H, L, NH, FFN, seq = struct.unpack_from("<4sIIIIIII", d, 0)
    assert magic == b"TETR" and ver >= 6
    tensors = {}
    pos = 64
    ternary = []
    while True:
        nlen = struct.unpack_from("<I", d, pos)[0]
        if nlen > 1024 or nlen == 0:
            break
        name = bytes(d[pos + 4:pos + 4 + nlen]).decode()
        if not name.endswith("latent_weights"):
            break
        rows, cols = struct.unpack_from("<HH", d, pos + 4 + nlen)
        p = pos + 4 + nlen + 4
        gs, na = struct.unpack_from("<HH", d, p)
        p += 4
        alphas = d[p:p + na * 4].view("<f4") if na else np.zeros(0)
        p += na * 4
        psz = (rows * cols + 3) // 4
        packed = d[p:p + psz]
        p += psz
        acc = d[p:p + rows * cols * 4].view("<f4").copy() if ver >= 6 else np.zeros(0)
        p += rows * cols * 4
        lut = np.array([-1.0, 0.0, 1.0, 0.0])
        flat = np.zeros(rows * cols, dtype=np.float32)
        nbytes = psz
        nbits = 0
        for b in range(nbytes):
            byte = packed[b]
            rem = min(4, cols - nbits) if (cols * rows) - nbits >= 0 else 0
            for i in range(min(4, (rows * cols) - nbits)):
                flat[nbits] = lut[(byte >> (6 - i * 2)) & 3]
                nbits += 1
        tensors[name] = flat.reshape(rows, cols), (gs, alphas)
        ternary.append(name)
        pos = p
    while pos + 4 <= len(d) - 4:
        nlen = struct.unpack_from("<I", d, pos)[0]
        if nlen > 1024 or nlen == 0:
            break
        name = bytes(d[pos + 4:pos + 4 + nlen]).decode()
        ndim, dtype = struct.unpack_from("<BB", d, pos + 4 + nlen)
        dims = tuple(struct.unpack_from("<IIII", d, pos + 4 + nlen + 2)[:ndim])
        p = pos + 4 + nlen + 2 + 16
        ne = int(np.prod(dims))
        if dtype == 1:  # INT8 with float scale
            scale = struct.unpack_from("<f", d, p)[0]
            p += 4
            arr = d[p:p + ne].view(np.int8).astype(np.float32) * scale
            p += ne
        else:
            arr = d[p:p + ne * 4].view("<f4").copy()
            p += ne * 4
        tensors[name] = arr.reshape(dims)
        pos = p
    return tensors, (V, H, L, NH, FFN, seq)


class TorchModel:
    def __init__(self, tensors, V, H, L, NH, FFN, seq):
        self.V, self.H, self.L, self.NH, self.FFN, self.seq = V, H, L, NH, FFN, seq
        self.HD = H // NH
        self.emb = torch.tensor(tensors["token_embedding.weight"], dtype=torch.float32)
        self.pos = torch.tensor(tensors["pos_embedding.weight"], dtype=torch.float32)
        self.ternary = {}
        for k, v in tensors.items():
            if k.endswith("latent_weights"):
                w, (gs, alphas) = v
                self.ternary[k] = torch.tensor(w, dtype=torch.float32)
        self.norms = {k: torch.tensor(v, dtype=torch.float32)
                      for k, v in tensors.items() if k.endswith("norm.weight")}
        self.eps = 1e-6

    @torch.no_grad()
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
        logits = self.emb @ x
        cache["pos"] += 1
        return logits

    def rmsnorm(self, x, w):
        rms = torch.sqrt(x.pow(2).mean() + self.eps)
        return (x / rms) * w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--tokens", default="examples/discrete/slice100k.bin")
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--eval-positions", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    tensors, (V, H, L, NH, FFN, seq) = parse_v6(args.model)
    model = TorchModel(tensors, V, H, L, NH, FFN, seq)
    toks = np.fromfile(args.tokens, dtype="<u2")

    cache = {
        "k": [torch.zeros(seq, H) for _ in range(L)],
        "v": [torch.zeros(seq, H) for _ in range(L)],
        "pos": 0,
    }

    def run_block(start):
        ce, valid = 0.0, 0
        for t in range(start, min(len(toks) - 1, start + 128)):
            logits = model.forward(int(toks[t]), cache)
            logits = logits * 0.0625
            ce += torch.nn.functional.cross_entropy(
                logits.unsqueeze(0), torch.tensor([int(toks[t + 1])])).item()
            valid += 1
        return ce / valid

    for _ in range(args.warmup):
        run_block(0)

    times = []
    ces = []
    for b in range(args.blocks):
        start = (b * 128) % (len(toks) - 1)
        t0 = time.perf_counter()
        ces.append(run_block(start))
        times.append((time.perf_counter() - t0) * 1000)
    print(f"torch block CE: {np.mean(ces):.4f} (blocks {args.blocks}, warmup {args.warmup})")
    print(f"torch ms/block: {np.mean(times):.2f} (min {min(times):.2f}, max {max(times):.2f})")
    print(f"torch ms/token: {np.mean(times) / 128:.3f}")


if __name__ == "__main__":
    main()
