"""ERC transfer-test driver: TinyStories (Domain A) -> FineWeb (Domain B).

Runs two Phase-2 arms from the same Phase-1 checkpoint:
  1. baseline: hard-STE fine-tune on the new domain (no ERC)
  2. erc:      same fine-tune + ERC residual (fast adaptation + consolidation)

Each arm logs held-out CE on the OLD slice (retention) and the NEW domain
slice (adaptation) into save_dir/eval_history.json; the driver prints a
comparison table.

Usage:
  python scripts/erc_transfer_test.py                       # full: 300+300 steps
  python scripts/erc_transfer_test.py --smoke               # 10+10 steps, fast
  python scripts/erc_transfer_test.py --phase1-checkpoint checkpoints_exp9_s10_seed1/checkpoint_000300.pt --phase2-steps 300
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_train(args, save_dir, resume=None, extra=None):
    # --steps is the TOTAL step budget; when resuming a Phase-1 checkpoint
    # (step == phase1_steps), training continues to phase1_steps+phase2_steps.
    total_steps = args.phase1_steps + args.phase2_steps
    cmd = [sys.executable, str(ROOT / "train.py"),
           "--preset", "tiny",
           "--steps", str(total_steps),
           "--data-cache", args.data_b,
           "--lr", str(args.lr),
           "--warmup-steps", str(args.warmup),
           "--min-lr", "1e-5",
           "--eval-interval", str(args.eval_every),
           "--eval-slice", args.slice_path,
           "--domain-eval", args.domain_path,
           "--eval-positions", str(args.eval_positions),
           "--save-best",
           "--save-dir", save_dir,
           "--seed", str(args.seed),
           ]
    if extra:
        cmd += extra
    if resume is not None:
        cmd += ["--resume", resume]
    print("=== CMD ===")
    print(" ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def load_history(save_dir):
    p = Path(save_dir) / "eval_history.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def fmt_history(hist):
    slices = {int(s): ce for s, ce in hist.get("slice_ces", [])}
    doms = {int(s): ce for s, ce in hist.get("domain_ces", [])}
    return slices, doms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-steps", type=int, default=300)
    ap.add_argument("--phase2-steps", type=int, default=300)
    ap.add_argument("--phase1-checkpoint", default=None,
                    help="reuse an existing Phase-1 checkpoint (skips Phase-1 training)")
    ap.add_argument("--data-a", default="tinydata")
    ap.add_argument("--data-b", default="data/fineweb_10bt")
    ap.add_argument("--slice-path", default="examples/discrete/sliceEval100k.bin")
    ap.add_argument("--domain-path", default="examples/discrete/finewebEval100k.bin")
    ap.add_argument("--lr", type=float, default=1e-4, help="Phase-2 fine-tune LR (core)")
    ap.add_argument("--erc-lr", type=float, default=0.01, help="Phase-2 residual LR")
    ap.add_argument("--erc-decay", type=float, default=0.99)
    ap.add_argument("--erc-commit-interval", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-positions", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="checkpoints_erc")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="10+10 steps")
    ap.add_argument("--erc-fp16", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.phase1_steps = 10
        args.phase2_steps = 10
        args.eval_every = 5

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # -- Phase 1: train on Domain A (TinyStories) ----------------------------
    phase1_ckpt = args.phase1_checkpoint
    if not phase1_ckpt:
        p1dir = str(out / "phase1")
        run_train(args, p1dir, extra=[
            "--steps", str(args.phase1_steps),
            "--data-cache", args.data_a,
            "--lr", "3e-4",
            "--warmup-steps", "50",
            "--save-best",
        ])
        p1_files = sorted((out / "phase1").glob("checkpoint_*.pt"))
        if not p1_files:
            print("ERROR: no Phase-1 checkpoint produced")
            sys.exit(1)
        phase1_ckpt = str(p1_files[-1])
        print(f"Phase-1 latest checkpoint: {phase1_ckpt}")

    print(f"\nPhase-1 checkpoint: {phase1_ckpt}")

    # -- Phase 2 arm A: baseline hard-STE fine-tune ---------------------------
    base_dir = str(out / "baseline")
    run_train(args, base_dir, resume=phase1_ckpt)

    # -- Phase 2 arm B: ERC ----------------------------------------------------
    erc_dir = str(out / "erc")
    run_train(args, erc_dir, resume=phase1_ckpt, extra=[
        "--erc",
        "--erc-lr", str(args.erc_lr),
        "--erc-decay", str(args.erc_decay),
        "--erc-commit-interval", str(args.erc_commit_interval),
    ] + (["--erc-fp16"] if args.erc_fp16 else []))

    if args.dry_run:
        return

    # -- Table -----------------------------------------------------------------
    print("\n" + "=" * 64)
    print("ERC TRANSFER TEST - TinyStories (A) -> FineWeb (B)")
    print("=" * 64)
    for label, d in [("baseline (hard-STE)", base_dir), ("erc", erc_dir)]:
        slices, doms = fmt_history(load_history(d))
        if not slices and not doms:
            print(f"\n[{label}] no eval_history.json - parse FINAL lines from stdout above")
            continue
        print(f"\n[{label}] {d}")
        print(f"  {'step':>6}  {'sliceCE(old)':>14}  {'domainCE(new)':>14}")
        steps = sorted(set(slices) | set(doms))
        for s in steps:
            a = f"{slices[s]:.4f}" if s in slices else "-"
            b = f"{doms[s]:.4f}" if s in doms else "-"
            print(f"  {s:>6}  {a:>14}  {b:>14}")
        if slices:
            k0, k1 = min(slices), max(slices)
            print(f"  RETENTION  delta_sliceCE: {slices[k0]:.4f} -> {slices[k1]:.4f}"
                  f"  ({slices[k1] - slices[k0]:+.4f})")
        if doms:
            k0, k1 = min(doms), max(doms)
            print(f"  ADAPTATION delta_domainCE: {doms[k0]:.4f} -> {doms[k1]:.4f}"
                  f"  ({doms[k1] - doms[k0]:+.4f})")


if __name__ == "__main__":
    main()
