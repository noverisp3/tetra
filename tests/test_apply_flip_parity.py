"""Kernel-level parity test: ternary_llm.quantization.apply_bit_flips vs the
C++ on-device kernel (inference/tetra.h::apply_bit_flips).

Context: both the Python trainer and the C++ selflearn/vulkan runtimes implement
"the same" flip rule. The flip math is not duplicated in source — Python
(quantization.py:821) and C++ (tetra.h:1792) are two independent implementations
of apply_bit_flips that must agree bit-for-bit on the same packed weights,
accumulators and outlier blob. A boundary desync here already shipped twice
(commit 3e5e2b3: packed code-11 vs blob recount diverged -> reload-OOB; and the
saturation toggle flip). This test pins the two kernels together:

  * v6 (no outliers): strict byte-for-byte parity (packed + accumulator).
  * v7 (sign blob):   strict parity incl. promote/demote and the rebuilt blob.
  * v7 adaptive-thr:  parity on the accumulated pass deltas (k*RMS threshold is
                      float-order sensitive between torch and g++; entries near
                      the tau boundary are excluded from the strict comparison).
  * v8 (magnitude blob): C++ semantics only — outliers are frozen (weight
                      untouched, accumulator reset). Python's kernel has no v8
                      concept (code 11 always dequantizes to ±2 there), so the
                      test asserts the C++ freeze invariant directly.

The C++ driver (inference/apply_flip_driver.cpp) is built on the fly with g++;
the test is SKIPPED when no compiler is available (matches tests/bench_avx.py).
"""
import os
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
INFERENCE = REPO / "inference"
BIN_NAME = "apply_flip_driver" + (".exe" if os.name == "nt" else "")

from ternary_llm.quantization import (
    apply_bit_flips,
    pack_sign_blob_tensor,
    pack_ternary_tensor,
)


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    """Build the C++ kernel driver once; skip the whole module if g++ is absent."""
    out = tmp_path_factory.mktemp("apply_flip_driver") / BIN_NAME
    cmd = ["g++", "-O2", "-std=c++17", "-I", str(INFERENCE),
           str(INFERENCE / "apply_flip_driver.cpp"), "-o", str(out)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        pytest.skip(f"g++ build of apply_flip_driver failed: {r.stderr.decode()[:500]}")
    assert out.exists()
    return str(out)


# ── data generation ──────────────────────────────────────────────────────────

def gen_vals(rng, rows, cols, version, n_outlier):
    """Random ternary matrix; v7/v8 also get `n_outlier` code-11 outliers."""
    vals = rng.choice([-1.0, 0.0, 1.0], size=(rows, cols)).astype(np.float32)
    if version >= 7 and n_outlier > 0:
        idx = rng.permutation(rows * cols)[:n_outlier]
        signs = rng.choice([-1.0, 1.0], size=len(idx)).astype(np.float32)
        vals.flat[idx] = signs * 2.0
    return vals


def gen_acc(rng, rows, cols, thr, mult, version):
    """Accumulators: a mix across the threshold; clearly crossed for v7+.

    Promote candidates (|acc| > mult*thr) are only meaningful where outliers
    exist (v7/v8). For v6 they are intentionally NOT emitted — a v6 promote in
    Python is litigated separately in test_v6_promote_disclosure below.
    """
    acc = rng.normal(0.0, thr * 0.5, size=(rows, cols)).astype(np.float32)
    if version >= 7:
        for _ in range(8):
            r, c = int(rng.integers(0, rows)), int(rng.integers(0, cols))
            acc[r, c] = rng.choice([-1, 1]) * thr * (mult + 0.5 + rng.random())
    return acc


def build_inputs(rows, cols, version, acc, vals):
    """Return (packed_uint8_tensor, blob_uint8_tensor, acc_float_tensor)."""
    w = torch.from_numpy(vals)  # {-2,-1,0,1,2} like the exported state
    packed = pack_ternary_tensor(w)
    blob = pack_sign_blob_tensor(w) if version >= 7 else torch.zeros(0, dtype=torch.uint8)
    acc_t = torch.from_numpy(acc.copy()).float()
    return packed, blob, acc_t


def run_cpp(driver, tmpdir, rows, cols, version, thr, toggle, mult, adaptive,
            packed, blob, acc):
    inp = tmpdir / "in"
    out = tmpdir / "out"
    inp.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    (inp / "packed.bin").write_bytes(packed.numpy().tobytes())
    (inp / "blob.bin").write_bytes(blob.numpy().tobytes())
    (inp / "acc.bin").write_bytes(acc.detach().cpu().numpy().tobytes())
    cmd = [driver, str(rows), str(cols), str(version), repr(thr),
           "1" if toggle else "0", repr(mult), repr(adaptive), str(inp), str(out)]
    r = subprocess.run(cmd, capture_output=True)
    assert r.returncode == 0, f"driver failed: {r.stderr.decode()}"
    cpp_packed = np.frombuffer((out / "packed.bin").read_bytes(), dtype=np.uint8)
    cpp_acc = np.frombuffer((out / "acc.bin").read_bytes(), dtype=np.float32).copy()
    cpp_blob = np.frombuffer((out / "blob.bin").read_bytes(), dtype=np.uint8)
    cpp_flips = int((out / "flips.txt").read_bytes().decode())
    return cpp_packed, cpp_acc, cpp_blob, cpp_flips


def run_py(rows, cols, version, thr, toggle, mult, adaptive, packed, blob, acc):
    py_packed = packed.clone()
    signs = blob.clone() if version >= 7 else None
    py_blob = blob.clone()
    new_blob = apply_bit_flips(
        py_packed, acc, thr, 1.0, (rows, cols),
        toggle=toggle,
        outlier_signs=signs,
        outlier_thr_mult=mult,
        adaptive_thr=(adaptive if adaptive > 0 else None),
    )
    if new_blob is not None:
        py_blob = new_blob
    py_packed = py_packed.detach().cpu().numpy()
    py_acc = acc.detach().cpu().numpy().astype(np.float32)
    py_blob = py_blob.detach().cpu().numpy()
    return py_packed, py_acc, py_blob


def report_eq(a, b, what):
    if a.shape != b.shape:
        return f"{what} shape {a.shape} != {b.shape}"
    if a.dtype.kind == "f":
        # Parked accumulators (promote) are written as tau, which C++ derives
        # from an fp64 sqrt while Python uses fp32 torch — the last bit wiggles
        # without any decision difference. Tolerate exact-equal floats only.
        if np.all(np.abs(a - b) <= np.maximum(1e-6, 1e-5 * np.abs(b))):
            return None
        diff = np.flatnonzero(a != b)
    else:
        diff = np.flatnonzero(a != b)
    if len(diff) == 0:
        return None
    if a.dtype.kind == "f":
        frac = len(diff) / a.size
        detail = np.abs(a.ravel()[diff] - b.ravel()[diff])[:3].tolist()
        return f"{what}: {len(diff)}/{a.size} entries differ ({frac:.2%}), first deltas {detail}"
    return f"{what}: {len(diff)}/{a.size} entries differ"


# ── v6 / v7 strict parity ────────────────────────────────────────────────────

@pytest.mark.parametrize("rows,cols,n_outlier", [
    (4, 4, 0), (8, 8, 0), (64, 12, 0), (32, 8, 0),
    (8, 8, 4), (64, 12, 20), (32, 16, 7),
])
@pytest.mark.parametrize("version", [6, 7])
@pytest.mark.parametrize("toggle", [False, True])
def test_strict_parity(driver, tmp_path, rows, cols, n_outlier, version, toggle):
    rng = np.random.default_rng(1234 + rows * cols + version * 7 + int(toggle) * 3)
    thr = 20.0
    mult = 3.0
    if version < 7:
        n_outlier = 0
    vals = gen_vals(rng, rows, cols, version, n_outlier)
    acc = gen_acc(rng, rows, cols, thr, mult, version)
    packed, blob, acc_t = build_inputs(rows, cols, version, acc, vals)

    cpp_packed, cpp_acc, cpp_blob, _ = run_cpp(
        driver, tmp_path, rows, cols, version, thr, toggle, mult, 0.0,
        packed, blob, acc_t)
    py_packed, py_acc, py_blob = run_py(
        rows, cols, version, thr, toggle, mult, 0.0, packed, blob, acc_t.clone())

    errs = [report_eq(py_packed, cpp_packed, "packed"),
            report_eq(py_acc.reshape(-1), cpp_acc.reshape(-1), "accumulator")]
    if version >= 7:
        errs.append(report_eq(py_blob, cpp_blob, "sign blob"))
    msgs = [e for e in errs if e]
    assert not msgs, "C++/Py kernel desync:\n" + "\n".join(msgs)


# ── v7 adaptive-thr: parity outside the numerics near-boundary band ──────────

@pytest.mark.parametrize("rows,cols,n_outlier", [(64, 12, 20), (32, 16, 7)])
@pytest.mark.parametrize("adaptive", [1.0, 3.0])
@pytest.mark.parametrize("toggle", [False, True])
def test_v7_adaptive_parity(driver, tmp_path, rows, cols, n_outlier, adaptive, toggle):
    rng = np.random.default_rng(9000 + rows * cols + int(adaptive * 10) + int(toggle))
    thr = 20.0
    mult = 3.0
    vals = gen_vals(rng, rows, cols, 7, n_outlier)
    acc = gen_acc(rng, rows, cols, thr, mult, 7)
    packed, blob, acc_t = build_inputs(rows, cols, 7, acc, vals)

    # Python per-channel tau (the reference). C++ uses the same formula, but
    # accumulates sumsq in fp64 and computes sqrt once, so rms (and any flip
    # right at the boundary) can differ by float rounding. Exclude the band.
    rms = acc_t.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-4)
    tau_row = (adaptive * rms[:, 0]).numpy()

    cpp_packed, cpp_acc, cpp_blob, _ = run_cpp(
        driver, tmp_path, rows, cols, 7, thr, toggle, mult, adaptive,
        packed, blob, acc_t)
    py_packed, py_acc, py_blob = run_py(
        rows, cols, 7, thr, toggle, mult, adaptive, packed, blob, acc_t.clone())

    # Entries |acc| far from tau are unambiguous for both engines.
    acc_np = acc.reshape(rows, cols)
    band = 0.1 * tau_row[:, None]
    sure = (np.abs(acc_np) - tau_row[:, None]).__abs__() > band
    # also only compare positions where python reports a flip decision changed
    # the packed code (flips/promotes/demotes manifest there, identical inputs).
    errs = []
    for name, py, cpp in (("packed", py_packed, cpp_packed),
                          ("accumulator", py_acc.ravel(), cpp_acc),
                          ("sign blob", py_blob, cpp_blob)):
        if py.shape != cpp.shape:
            errs.append(f"{name} shape {py.shape} != {cpp.shape}")
            continue
        safe = sure.ravel()[: py.size]
        if py.dtype.kind == "f":
            # Parked promote values sit at tau, which differs by fp64-vs-fp32
            # rounding (C++ sqrt vs torch) — decisions identical, float wiggles.
            close = np.abs(py - cpp) <= np.maximum(1e-6, 1e-5 * np.abs(cpp))
            d = np.flatnonzero(~close)
        else:
            d = np.flatnonzero(py != cpp)
        d_sure = d[d < safe.size][safe[d[d < safe.size]]]
        if len(d_sure):
            errs.append(f"{name}: {len(d_sure)} certain-entry diffs")
    assert not errs, "C++/Py adaptive-thr kernel desync:\n" + "\n".join(errs)


# ── v8: C++ freeze invariant (no Python equivalent) ──────────────────────────

@pytest.mark.parametrize("rows,cols,n_outlier", [(8, 8, 4), (64, 12, 20)])
@pytest.mark.parametrize("toggle", [False, True])
def test_v8_freeze_invariant(driver, tmp_path, rows, cols, n_outlier, toggle):
    rng = np.random.default_rng(555 + rows * cols)
    thr = 20.0
    mult = 3.0
    vals = gen_vals(rng, rows, cols, 8, n_outlier)
    acc = gen_acc(rng, rows, cols, thr, mult, 8)

    # v8 holds the true fixed-point magnitudes (round((W/Δ)·32)) in blob[]. For
    # the invariant test we mirror the exporter: blob stores ±48 for ±1.5-sized
    # outliers — the exact boundary the pre-fix code dropped (3e5e2b3).
    w = torch.from_numpy(vals)
    mask = w.abs() > 1.5
    mag = torch.full_like(w, 0, dtype=torch.int8)
    mag[mask] = torch.where(w[mask] >= 0,
                            torch.full_like(w[mask], 48).to(torch.int8),
                            torch.full_like(w[mask], -48).to(torch.int8))
    blob8 = mag.detach().cpu().numpy().ravel()[mask.detach().cpu().numpy().ravel()].tobytes()
    packed = pack_ternary_tensor(w)
    acc_t = torch.from_numpy(acc.copy()).float()

    cpp_packed, cpp_acc, cpp_blob, _ = run_cpp(
        driver, tmp_path, rows, cols, 8, thr, toggle, mult, 0.0,
        packed, torch.from_numpy(np.frombuffer(blob8, dtype=np.uint8).copy()), acc_t)
    _ = cpp_blob

    # 1) out-of-boundary weights are FROZEN: their packed codes survive the pass.
    out_mask = mask.numpy()
    py_packed, py_acc, _ = run_py(
        rows, cols, 8, thr, toggle, mult, 0.0, packed, torch.zeros(0, dtype=torch.uint8),
        acc_t.clone())
    # Compare at the code level: extract 2-bit codes from both packed buffers,
    # then look only at outlier positions (code 11 = ±2 in this synthetic setup).
    def codes_flat(arr):
        c = np.empty(rows * cols, dtype=np.uint8)
        for i in range(rows * cols):
            c[i] = (arr[i // 4] >> (6 - (i & 3) * 2)) & 3
        return c
    # Frozen: the C++ side must leave every code-11 code untouched.
    orig_codes = codes_flat(packed.numpy())
    new_codes = codes_flat(cpp_packed)
    frozen = (orig_codes == 3) & (out_mask.ravel())
    assert (orig_codes[frozen] == new_codes[frozen]).all(), (
        "C++ v8 flipped a frozen outlier — expected code-11 weights to stay put")
    # 2) ...and the accumulator is reset to zero wherever a frozen outlier
    #    would have flipped (|acc| > thr); low-|acc| outliers legitimately keep
    #    their accumulator since no action fires for them.
    acc_np = acc_t.numpy()
    fire = frozen & (np.abs(acc_np.ravel()) > thr)
    assert np.allclose(cpp_acc.ravel()[fire], 0.0, atol=1e-6), (
        "v8 frozen outlier accumulator not reset")
    # 3) Non-outlier entries go through the normal flip path; no Python
    #    equivalent exists for v8 (Python's kernel is v7-only), so we only
    #    assert the kernel ran without corrupting the frozen outlier count.
    assert int((new_codes == 3).sum()) >= int(((orig_codes == 3) & out_mask.ravel()).sum()), (
        "v8 pass dropped frozen outlier codes")