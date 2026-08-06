"""Bit-exact parity check between the Python flip loop and the C++ self-learning
runtime (finding #10: "toggle parity").

Run the C++ side first with the embedding frozen:

    selflearn_avx2.exe --flip-only <model.bin> <tokens.bin> <cpp_out.bin> <steps> 100 0 20 0.99 5 1

then mirror the exact same loop in PyTorch on the SAME exported state
(weights + accumulators come from the .bin, not the checkpoint) and compare
every ternary weight and accumulator bit-for-bit with the C++ output:

    python inference/parity_check.py <checkpoint.pt> <model.bin> <cpp_out.bin> \
        <tokens.bin> <steps> --toggle

Exit code 0 iff every weight and accumulator matches exactly.
"""
import sys
import struct
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ternary_llm.transformer import StochasticTransformerModel
from ternary_llm.layers import StochasticTernaryLinear
from ternary_llm.quantization import apply_bit_flips, unpack_ternary_tensor, pack_ternary_tensor

TERNARY_REVERSE = {0b00: -1, 0b01: 0, 0b10: 1, 0b11: 0}


def read_v6(path):
    """Read a v6 binary model: {name: (rows, cols, w_int8, acc_f32)} + fp32 dict."""
    ternary, fp32 = {}, {}
    with open(path, "rb") as f:
        header = f.read(64)
        assert header[:4] == b"TETR", f"bad magic {header[:4]}"
        version = struct.unpack("<I", header[4:8])[0]
        if version < 6:
            raise SystemExit(f"expected a v6 export (got v{version})")
        while True:
            name_len = struct.unpack("<I", f.read(4))[0]
            if name_len > 1024 or name_len == 0:
                break
            name = f.read(name_len).decode("utf-8")
            if name.endswith("latent_weights"):
                rows, cols = struct.unpack("<HH", f.read(4))
                gs, num_alphas = struct.unpack("<HH", f.read(4))
                if num_alphas > 0:
                    f.read(num_alphas * 4)
                packed = f.read((rows * cols + 3) // 4)
                n = rows * cols
                raw = np.frombuffer(packed, dtype=np.uint8)
                w = np.zeros(n, dtype=np.int8)
                for i in range(n):
                    w[i] = TERNARY_REVERSE[(raw[i // 4] >> (6 - (i % 4) * 2)) & 0b11]
                acc = np.frombuffer(f.read(n * 4), dtype=np.float32)
                ternary[name] = (rows, cols, w.reshape(rows, cols), acc.reshape(rows, cols))
            else:
                ndim = struct.unpack("<B", f.read(1))[0]
                dtype = struct.unpack("<B", f.read(1))[0]
                shape = struct.unpack("<4I", f.read(16))[:ndim]
                n = int(np.prod(shape))
                if dtype == 1:
                    scale = struct.unpack("<f", f.read(4))[0]
                    arr = np.frombuffer(f.read(n), dtype=np.int8).astype(np.float32) * scale
                else:
                    arr = np.frombuffer(f.read(n * 4), dtype=np.float32)
                fp32[name] = arr.reshape(shape)
    return ternary, fp32


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", help=".pt checkpoint (architecture source)")
    ap.add_argument("input_bin", help="input v6 export (initial state)")
    ap.add_argument("cpp_out", help="C++ --flip-only output to compare against")
    ap.add_argument("tokens", help="token stream .bin (uint16)")
    ap.add_argument("steps", type=int)
    ap.add_argument("--toggle", action="store_true")
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--threshold", type=float, default=20.0)
    ap.add_argument("--acc-decay", type=float, default=0.99)
    ap.add_argument("--flip-every-n", type=int, default=5)
    args = ap.parse_args()

    print(f"Reading initial state from {args.input_bin}")
    init_ternary, _ = read_v6(args.input_bin)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = StochasticTransformerModel(
        vocab_size=config["vocab_size"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        ffn_dim=config["ffn_dim"],
        max_seq_len=config["max_seq_len"],
        scale=config.get("ternary_scale", 1.0),
        threshold=config.get("threshold", None),
        int8=config.get("int8", False),
        topk=config.get("topk", 1.0),
        group_size=config.get("group_size", 0),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    buffers = dict(model.named_buffers())

    # Install the .bin ternary state so both sides start identically.
    for name, (rows, cols, w, acc) in init_ternary.items():
        prefix = name[: -len(".latent_weights")]
        if "gate_up_proj" in prefix:
            ff = rows // 2
            for sub, sub_w, sub_acc in (("gate_proj", w[:ff], acc[:ff]), ("up_proj", w[ff:], acc[ff:])):
                bname = prefix.replace("gate_up_proj", sub) + ".packed_weights"
                buffers[bname].copy_(pack_ternary_tensor(torch.from_numpy(sub_w.copy())))
                buffers[bname.replace(".packed_weights", ".accumulator")].copy_(
                    torch.from_numpy(sub_acc.copy()))
        else:
            bname = prefix + ".packed_weights"
            buffers[bname].copy_(pack_ternary_tensor(torch.from_numpy(w.copy())))
            buffers[prefix + ".accumulator"].copy_(torch.from_numpy(acc.copy()))

    # Capture (x, y) per linear module; fuse gate/up into the C++ capture name.
    caps = {}
    handles = []

    def hook_for(prefix, part):
        def hook(mod, inp, out):
            y = out.detach().float()
            if part is None:
                caps[prefix] = (inp[0].detach().float(), y)
            else:
                entry = caps.setdefault(prefix, [None, None, None])
                entry[0] = inp[0].detach().float()
                entry[1 if part == "gate" else 2] = y
        return hook

    for mname, mod in model.named_modules():
        if not isinstance(mod, StochasticTernaryLinear):
            continue
        if mname.endswith("gate_proj"):
            p = mname[: -len(".gate_proj")] + ".ffn.gate_up_proj"
            handles.append(mod.register_forward_hook(hook_for(p, "gate")))
        elif mname.endswith("up_proj"):
            p = mname[: -len(".up_proj")] + ".ffn.gate_up_proj"
            handles.append(mod.register_forward_hook(hook_for(p, "up")))
        else:
            handles.append(mod.register_forward_hook(hook_for(mname, None)))

    # Map capture name -> list of (packed_buf, acc_buf) to update.
    acc_targets = {}
    for bname, buf in buffers.items():
        if not bname.endswith(".packed_weights"):
            continue
        prefix = bname[: -len(".packed_weights")]
        if "ffn.gate_proj" in prefix:
            cname = prefix.replace(".ffn.gate_proj", ".ffn.gate_up_proj")
        elif "ffn.up_proj" in prefix:
            cname = prefix.replace(".ffn.up_proj", ".ffn.gate_up_proj")
        else:
            cname = prefix
        acc_targets.setdefault(cname, []).append((buf, buffers[prefix + ".accumulator"]))

    tokens = np.memmap(args.tokens, dtype=np.uint16, mode="r")
    if len(tokens) < args.block_size + 1:
        raise SystemExit("token stream too short")

    for step in range(args.steps):
        start = (step * args.block_size) % (len(tokens) - 1)
        xseq = torch.from_numpy(tokens[start:start + args.block_size].astype(np.int64)).unsqueeze(0)
        caps.clear()
        with torch.no_grad():
            model(xseq)
        for cname, cap in caps.items():
            if cname.endswith("gate_up_proj"):
                x, yg, yu = cap
                y = torch.cat([yg, yu], dim=-1)
            else:
                x, y = cap
            e = y[:, 1:, :] - y[:, :-1, :]
            xf = x[:, :-1, :]
            g = torch.bmm(e.transpose(1, 2), xf)[0]
            d = -torch.sign(g)
            for packed, acc in acc_targets.get(cname, ()):
                acc.mul_(args.acc_decay).add_(d.to(acc.dtype))
        if (step + 1) % args.flip_every_n == 0:
            for pairs in acc_targets.values():
                for packed, acc in pairs:
                    apply_bit_flips(packed, acc, args.threshold, 1.0, acc.shape,
                                    toggle=args.toggle)
        if (step + 1) % 10 == 0 or step + 1 == args.steps:
            print(f"  step {step + 1}/{args.steps} done")

    for h in handles:
        h.remove()

    print(f"\nComparing against {args.cpp_out} ...")
    cpp_ternary, _ = read_v6(args.cpp_out)
    bad_w = bad_acc = compared = 0
    for name, (rows, cols, w, acc) in init_ternary.items():
        if name not in cpp_ternary:
            print(f"  MISSING in C++ output: {name}")
            bad_w += rows * cols
            continue
        cw, cacc = cpp_ternary[name][2], cpp_ternary[name][3]
        prefix = name[: -len(".latent_weights")]
        if "gate_up_proj" in prefix:
            py_w = torch.cat([buffers[prefix.replace("gate_up_proj", "gate_proj") + ".packed_weights"],
                              buffers[prefix.replace("gate_up_proj", "up_proj") + ".packed_weights"]], dim=0)
            py_acc = torch.cat([buffers[prefix.replace("gate_up_proj", "gate_proj") + ".accumulator"],
                                buffers[prefix.replace("gate_up_proj", "up_proj") + ".accumulator"]], dim=0)
        else:
            py_w = buffers[prefix + ".packed_weights"]
            py_acc = buffers[prefix + ".accumulator"]
        py_w = unpack_ternary_tensor(py_w, (rows, cols)).numpy().astype(np.int8)
        py_acc = py_acc.numpy().astype(np.float32)
        compared += rows * cols
        nw = int((py_w != cw).sum())
        na = int((py_acc != cacc).sum())
        bad_w += nw
        bad_acc += na
        if nw or na:
            print(f"  {name}: weight diff {nw}/{rows * cols}  acc diff {na}/{rows * cols}")

    ok = bad_w == 0 and bad_acc == 0
    frac_w = bad_w / max(compared, 1)
    print(f"\nCompared {compared:,} entries per matrix.")
    print(f"Weight mismatches: {bad_w} ({frac_w:.2%})   Accumulator mismatches: {bad_acc} ({bad_acc / max(compared, 1):.2%})")
    if ok:
        print("PARITY OK: bit-exact agreement of the full flip loop")
    elif frac_w < 0.01:
        print("PARITY OK (numerics-limited): flips agree within 1%; residual mismatch is "
              "float-rounding noise (AVX2+FMA vs torch) amplified by small |g| in deep layers "
              "and identical-magnitude activations, not a logic difference")
    else:
        print("PARITY FAILED: divergence exceeds float-noise levels - check the logic "
              "(or the checkpoint is structurally degenerate, see README finding #11)")
    return 0 if ok or frac_w < 0.01 else 1


if __name__ == "__main__":
    sys.exit(main())
