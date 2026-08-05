"""Dump intermediates + first-12 logits per position (for Vulkan parity debugging).

Usage: python inference/dbg_logits.py <model.bin> <tokens.bin> [n_positions]
"""
import sys
sys.path.insert(0, "D:/Tetra/inference")
from bench_torch import parse_v6, TorchModel
import numpy as np
import torch

model_path = sys.argv[1]
toks_path = sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 3

tensors, (V, H, L, NH, FFN, seq) = parse_v6(model_path)
model = TorchModel(tensors, V, H, L, NH, FFN, seq)
toks = np.fromfile(toks_path, dtype="<u2")
cache = {"k": [torch.zeros(seq, H) for _ in range(L)],
         "v": [torch.zeros(seq, H) for _ in range(L)], "pos": 0}


def show(label, v):
    v = v.detach().numpy().flatten() if torch.is_tensor(v) else v
    print(f"  {label:<18}:", " ".join(f"{x:.4f}" for x in v[:8]))


with torch.no_grad():
    for t in range(n):
        H_ = model.H
        L_ = model.L
        tok, pos = int(toks[t]), cache["pos"]
        x = model.emb[tok] + model.pos[pos]
        show("embed", x)
        for l in range(L_):
            p = f"layers.{l}."
            normed = model.rmsnorm(x, model.norms[p + "attn_norm.weight"])
            q = model.ternary[p + "attn.q_proj.latent_weights"] @ normed
            k = model.ternary[p + "attn.k_proj.latent_weights"] @ normed
            v = model.ternary[p + "attn.v_proj.latent_weights"] @ normed
            if l == 0:
                show("normed", normed)
                show("q", q)
                show("k", k)
                show("v", v)
            cache["k"][l][pos] = k
            cache["v"][l][pos] = v
            attn_out = torch.zeros(H_)
            for head in range(model.NH):
                qh = q[head * model.HD:(head + 1) * model.HD]
                kh = cache["k"][l][:pos + 1, head * model.HD:(head + 1) * model.HD]
                vh = cache["v"][l][:pos + 1, head * model.HD:(head + 1) * model.HD]
                s = (kh @ qh) / (model.HD ** 0.5)
                s = s.clamp(-80.0, 80.0)
                s = torch.softmax(s, dim=0)
                attn_out[head * model.HD:(head + 1) * model.HD] = s @ vh
            if l == 0:
                show("layer0_attn", attn_out)
            x = x + model.ternary[p + "attn.o_proj.latent_weights"] @ attn_out
            if l == 1:
                show("L1_q", q)
                show("L1_v", v)
                show("L1_attn", attn_out)
                show("L1_proj", model.ternary[p + "attn.o_proj.latent_weights"] @ attn_out)
            if l == 0:
                show("layer0_attn_resid", x)
            if l == 1:
                show("L1_x", x)
            normed = model.rmsnorm(x, model.norms[p + "ffn_norm.weight"])
            if l == 0:
                show("ffn_normed", normed)
            fused = model.ternary[p + "ffn.gate_up_proj.latent_weights"] @ normed
            gate, up = fused[:FFN], fused[FFN:]
            hidden = torch.nn.functional.silu(gate) * up
            if l == 0:
                show("gate_up", fused)
                show("hidden", hidden)
            if l == 1:
                show("L1_ffn_normed", normed)
                show("L1_gate_up", fused)
                show("L1_hidden", hidden)
                show("L1_down", model.ternary[p + "ffn.down_proj.latent_weights"] @ hidden)
            x = x + model.ternary[p + "ffn.down_proj.latent_weights"] @ hidden
            if l == 0:
                show("layer0_ffn_resid", x)
        lg = model.emb @ model.rmsnorm(x, model.norms["norm.weight"])
        cache["pos"] += 1
        show("L5_final_x", x)
        show("L5_normed", model.rmsnorm(x, model.norms["norm.weight"]))
        print(f"pos {t} tok {tok} logits:", " ".join(f"{v:.4f}" for v in lg.numpy()[:12]))
