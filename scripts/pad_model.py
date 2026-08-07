"""Pad a v6/v7 TETR binary with zero weights (lossless FFN expansion).

Appends zero-weight rows/cols to every FFN matrix (gate_up_proj, down_proj)
and grows the header ffn_dim. Zero weights contribute 0 to every output, so
the padded model must be bit-identical in behavior to the original
(verify with tests/ce_v6_vs_v7.py before/after).

Handles v6 (accumulator) and v7 (outlier side-channel blob) entries:
accumulators are padded with zeros; the blob is untouched because new
weights are code 01 (zero), which never participates in the outlier index.

Usage:
  python scripts/pad_model.py <model.bin> -o <out.bin> [--ffn 2048] [--verify]
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))

HEADER_FMT = "<4sIIIIIIIQQ"
HEADER_LEN = 64


def parse_header(d):
    magic, ver, V, H, L, NH, FFN, seq, tc, fc = struct.unpack_from(HEADER_FMT, d, 0)
    assert magic == b"TETR", f"Bad magic {magic}"
    flags = kv = rope = gs = 0
    if ver >= 5:
        flags, kv, rope, gs = struct.unpack_from("<HHHH", d, 48)
    return {
        "version": ver, "vocab_size": V, "hidden_dim": H, "num_layers": L,
        "num_heads": NH, "ffn_dim": FFN, "max_seq_len": seq,
        "ternary_count": tc, "fp32_count": fc,
        "flags": flags, "kv_latent_dim": kv, "rope_per_head": rope,
        "group_size": gs,
    }


def pack_header(h):
    b = struct.pack(HEADER_FMT, b"TETR", h["version"], h["vocab_size"], h["hidden_dim"],
                    h["num_layers"], h["num_heads"], h["ffn_dim"], h["max_seq_len"],
                    h["ternary_count"], h["fp32_count"])
    if h["version"] >= 5:
        b += struct.pack("<HHHH", h["flags"], h["kv_latent_dim"], h["rope_per_head"],
                         h["group_size"])
    return b.ljust(HEADER_LEN, b"\x00")


def parse_entries(d, version):
    """Return (ternary list, fp32 list, metadata_tail)."""
    ternary = []
    fp32 = []
    pos = HEADER_LEN
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
        alphas = d[p:p + na * 4]
        p += na * 4
        psz = (rows * cols + 3) // 4
        packed = d[p:p + psz]
        p += psz
        acc = None
        if version >= 6:
            acc = d[p:p + rows * cols * 4]
            p += rows * cols * 4
        blob = None
        n_out = 0
        if version >= 7:
            n_out = struct.unpack_from("<I", d, p)[0]
            p += 4
            blob = d[p:p + (n_out + 7) // 8]
            p += (n_out + 7) // 8
        ternary.append({"name": name, "rows": rows, "cols": cols, "gs": gs,
                        "na": na, "alphas": alphas, "packed": packed,
                        "acc": acc, "blob": blob, "n_out": n_out})
        pos = p
    while pos + 4 <= len(d) - 4:
        nlen = struct.unpack_from("<I", d, pos)[0]
        if nlen > 1024 or nlen == 0:
            break
        name = bytes(d[pos + 4:pos + 4 + nlen]).decode()
        ndim, dtype = struct.unpack_from("<BB", d, pos + 4 + nlen)
        p = pos + 4 + nlen + 2
        dims = struct.unpack_from("<IIII", d, p)
        p += 16
        ne = int(np.prod(dims[:ndim]))
        if dtype == 1:
            p += 4 + ne
        else:
            p += ne * 4
        fp32.append({"name": name, "raw": d[pos:p]})
        pos = p
    return ternary, fp32, d[pos:]


def encode_packed(codes):
    n = codes.size
    nbytes = (n + 3) // 4
    out = np.zeros(nbytes, dtype=np.uint8)
    for i in range(n):
        b, s = divmod(i, 4)
        out[b] |= codes[i] << (6 - s * 2)
    return out


def pad_matrix(entry, pad_cols=0, inserts=None):
    """Pad with zero weights (code 01).

    inserts: list of (row_idx, count) — insert `count` zero rows before
    original row `row_idx` (indices in the ORIGINAL row space). Required for
    fused gate_up: the matrix is [gate; up] and the C++/torch split is
    fused[:FFN] / fused[FFN:], so new FFN rows must be inserted inside BOTH
    halves: inserts=[(F, d), (2F, d)] yields [gate_old, Z, up_old, Z].
    """
    rows, cols = entry["rows"], entry["cols"]
    codes = np.zeros(rows * cols, dtype=np.int8)
    nb = 0
    for byte in entry["packed"]:
        for i in range(4):
            if nb >= codes.size:
                break
            codes[nb] = (byte >> (6 - i * 2)) & 3
            nb += 1
    src = codes.reshape(rows, cols)
    new_cols = cols + pad_cols
    ins = dict(inserts or {})
    new_rows = rows + sum(ins.values())
    m2 = np.full((new_rows, new_cols), 1, dtype=np.int8)  # code 01 = zero weight
    offset = 0
    for r in range(rows):
        offset += ins.get(r, 0)
        m2[r + offset, :cols] = src[r]
    entry["rows"], entry["cols"] = new_rows, new_cols
    entry["packed"] = encode_packed(m2.ravel()).tobytes()
    if entry["acc"] is not None:
        a = np.frombuffer(entry["acc"], dtype="<f4").reshape(rows, cols)
        a2 = np.zeros((new_rows, new_cols), dtype="<f4")
        offset = 0
        for r in range(rows):
            offset += ins.get(r, 0)
            a2[r + offset, :cols] = a[r]
        entry["acc"] = a2.tobytes()
    if ins and entry["na"] > 0:
        # per-row alphas follow the same row layout; new rows get 1.0 (neutral)
        al = np.frombuffer(entry["alphas"], dtype="<f4")
        a2 = np.ones(new_rows, dtype="<f4")
        offset = 0
        for r in range(rows):
            offset += ins.get(r, 0)
            a2[r + offset] = al[r]
        entry["alphas"] = a2.tobytes()
        entry["na"] = new_rows
    # blob untouched: padded weights are code 01, not code 11


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="v6/v7 TETR binary")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--ffn", type=int, default=0, help="target ffn_dim (default: 2x)")
    ap.add_argument("--verify", action="store_true",
                    help="reload both files and compare ternary weights")
    args = ap.parse_args()

    d = np.fromfile(args.model, dtype=np.uint8).tobytes()
    h = parse_header(d)
    ternary, fp32, tail = parse_entries(d, h["version"])
    print(f"v{h['version']}: V={h['vocab_size']} H={h['hidden_dim']} L={h['num_layers']} "
          f"FFN={h['ffn_dim']} seq={h['max_seq_len']} | {len(ternary)} ternary, "
          f"{len(fp32)} fp32, tail {len(tail)} B")

    new_ffn = args.ffn or h["ffn_dim"] * 2
    d_ffn = new_ffn - h["ffn_dim"]
    assert d_ffn > 0, "target ffn must be larger"
    assert all(t["gs"] == 0 for t in ternary), "grouped entries not supported yet"

    for t in ternary:
        if t["name"].endswith("ffn.gate_up_proj.latent_weights"):
            assert t["rows"] == 2 * h["ffn_dim"] and t["cols"] == h["hidden_dim"]
            # fused [gate; up]: new FFN rows go inside BOTH halves so
            # gate = fused[:FFN] / up = fused[FFN:2*FFN] keep their semantics
            f = h["ffn_dim"]
            pad_matrix(t, pad_cols=0, inserts=[(f, d_ffn), (2 * f, d_ffn)])
        elif t["name"].endswith("ffn.down_proj.latent_weights"):
            assert t["rows"] == h["hidden_dim"] and t["cols"] == h["ffn_dim"]
            pad_matrix(t, pad_cols=d_ffn)
    h["ffn_dim"] = new_ffn

    out = bytearray(pack_header(h))
    for t in ternary:
        out += struct.pack("<I", len(t["name"])) + t["name"].encode()
        out += struct.pack("<HH", t["rows"], t["cols"])
        out += struct.pack("<HH", t["gs"], t["na"]) + t["alphas"]
        out += t["packed"]
        if h["version"] >= 6:
            out += t["acc"]
        if h["version"] >= 7:
            out += struct.pack("<I", t["n_out"]) + t["blob"]
    for f in fp32:
        out += f["raw"]
    out += tail
    Path(args.output).write_bytes(bytes(out))
    print(f"wrote {args.output}: {len(out)} B (was {len(d)} B, +{len(out) - len(d)} B), "
          f"FFN {h['ffn_dim'] - d_ffn} -> {h['ffn_dim']}")

    if args.verify:
        from verify_export import load_binary_model
        w1, _ = load_binary_model(args.model)
        w2, _ = load_binary_model(args.output)
        same = True
        for k in w1:
            if k.endswith("latent_weights"):
                m1 = w1[k]
                r1, c1 = m1.shape
                m2 = w2[k]
                r2, c2 = m2.shape
                if r2 == r1 and c2 == c1:
                    ok = np.array_equal(m1, m2)
                elif k.endswith("ffn.gate_up_proj.latent_weights"):
                    f = r1 // 2
                    d2 = (r2 - r1) // 2  # new rows per half
                    ok = np.array_equal(m1[:f], m2[:f]) and \
                         np.all(m2[f:f + d2] == 0) and \
                         np.array_equal(m1[f:], m2[f + d2:f + d2 + f]) and \
                         np.all(m2[f + d2 + f:] == 0)
                elif k.endswith("ffn.down_proj.latent_weights"):
                    ok = np.array_equal(m1, m2[:, :c1]) and np.all(m2[:, c1:] == 0)
                else:
                    ok = False
                if not ok:
                    same = False
                    print(f"  MISMATCH {k}")
        print("verify:", "OK (original region bit-identical, padding zero)" if same
              else "FAILED")


if __name__ == "__main__":
    main()
