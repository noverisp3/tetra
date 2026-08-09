"""ERC transfer-test driver: TinyStories (Domain A) -> FineWeb (Domain B).

Runs several Phase-2 arms from the same Phase-1 checkpoint and compares
old-domain retention (sliceCE) vs new-domain adaptation (domainCE):

  baseline       hard-STE fine-tune on the new domain (no ERC)
  baseline-lowlr same fine-tune at 1/10 core LR  (is it just LR?)
  erc            ERC residual (fast adaptation + consolidation)
  erc-nocommit   ERC residual, commits disabled (core never changes)
  erc-freeze     ERC with the latent core frozen (R-only learning)
  erc-common     ERC with commits every 2 steps (faster consolidation)

Each arm logs held-out CE on the OLD slice (retention) and the NEW domain
slice (adaptation) into save_dir/eval_history.json; the driver prints a
comparison table.

Usage:
  python scripts/erc_transfer_test.py                       # full: 300+300 steps
  python scripts/erc_transfer_test.py --smoke               # 10+10 steps, fast
  python scripts/erc_transfer_test.py --arms baseline,erc   # subset of arms
  python scripts/erc_transfer_test.py --phase1-checkpoint checkpoints_exp9_s10_seed1/checkpoint_000300.pt --phase2-steps 300
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# (arm name, extra train.py args). Order = table order.
ARMS = {
    "baseline": ("baseline (hard-STE)", []),
    "baseline-lowlr": ("baseline (LR/10)", ["--lr", "1e-5", "--min-lr", "1e-6"]),
    "erc": ("erc (R lr 0.01, commit 10)", ["--erc", "--erc-lr", "0.01", "--erc-decay", "0.99", "--erc-commit-interval", "10"]),
    "erc-nocommit": ("erc (R, no commit)", ["--erc", "--erc-lr", "0.01", "--erc-decay", "0.99", "--erc-commit-interval", "0"]),
    "erc-freeze": ("erc (R, core frozen)", ["--erc", "--erc-lr", "0.01", "--erc-decay", "0.99", "--erc-commit-interval", "10", "--erc-freeze-core"]),
    "erc-common": ("erc (R lr 0.01, commit 2)", ["--erc", "--erc-lr", "0.01", "--erc-decay", "0.99", "--erc-commit-interval", "2"]),
}


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
    sc = {int(s): ce for s, ce in hist.get("slice_core_ces", [])}
    dc = {int(s): ce for s, ce in hist.get("domain_core_ces", [])}
    return slices, doms, sc, dc


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
    ap.add_argument("--arms", default="baseline,baseline-lowlr,erc,erc-nocommit,erc-freeze,erc-common",
                    help="comma-separated arm names (see ARMS dict)")
    args = ap.parse_args()

    arm_list = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arm_list:
        if a not in ARMS:
            print(f"ERROR: unknown arm '{a}' (choose from {sorted(ARMS)})")
            sys.exit(1)

    if args.smoke:
        if not args.phase1_checkpoint:
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

    # -- Phase 2 arms ----------------------------------------------------------
    dirs = {}
    for arm in arm_list:
        label, extra = ARMS[arm]
        arm_dir = str(out / arm)
        dirs[arm] = arm_dir
        print(f"\n### ARM: {label} -> {arm_dir}")
        run_train(args, arm_dir, resume=phase1_ckpt, extra=extra)

    if args.dry_run:
        return

    # -- Table -----------------------------------------------------------------
    print("\n" + "=" * 64)
    print("ERC TRANSFER TEST - TinyStories (A) -> FineWeb (B)")
    print("=" * 64)
    for arm in arm_list:
        label, _ = ARMS[arm]
        d = dirs[arm]
        slices, doms, sc, dc = fmt_history(load_history(d))
        if not slices and not doms:
            print(f"\n[{label}] no eval_history.json - parse FINAL lines from stdout above")
            continue
        print(f"\n[{label}] {d}")
        print(f"  {'step':>6}  {'sliceCE(old)':>14}  {'coreOnly':>10}  {'domainCE(new)':>14}  {'coreOnly':>10}")
        steps = sorted(set(slices) | set(doms))
        for s in steps:
            a = f"{slices[s]:.4f}" if s in slices else "-"
            b = f"{doms[s]:.4f}" if s in doms else "-"
            ac = f"{sc[s]:.4f}" if s in sc else ""
            bc = f"{dc[s]:.4f}" if s in dc else ""
            print(f"  {s:>6}  {a:>14}  {ac:>10}  {b:>14}  {bc:>10}")
        if slices:
            k0, k1 = min(slices), max(slices)
            print(f"  RETENTION  delta_sliceCE: {slices[k0]:.4f} -> {slices[k1]:.4f}"
                  f"  ({slices[k1] - slices[k0]:+.4f})")
        if sc:
            k0, k1 = min(sc), max(sc)
            print(f"  RETENTION  delta_coreCE:  {sc[k0]:.4f} -> {sc[k1]:.4f}"
                  f"  ({sc[k1] - sc[k0]:+.4f})  <- core only (R=0)")
        if doms:
            k0, k1 = min(doms), max(doms)
            print(f"  ADAPTATION delta_domainCE: {doms[k0]:.4f} -> {doms[k1]:.4f}"
                  f"  ({doms[k1] - doms[k0]:+.4f})")
        if dc:
            k0, k1 = min(dc), max(dc)
            print(f"  ADAPTATION delta_domainCoreCE: {dc[k0]:.4f} -> {dc[k1]:.4f}"
                  f"  ({dc[k1] - dc[k0]:+.4f})  <- core only (R=0)")

    # -- Cross-arm summary -----------------------------------------------------
    print("\n" + "=" * 64)
    print("SUMMARY (delta over eval span)")
    print("=" * 64)
    print(f"  {'arm':<22} {'delta slice':>12} {'delta core':>12} {'delta domain':>13} {'delta dCore':>13}")
    for arm in arm_list:
        label, _ = ARMS[arm]
        slices, doms, sc, dc = fmt_history(load_history(dirs[arm]))
        ds = (lambda k0, k1: slices[k1] - slices[k0])(min(slices), max(slices)) if slices else float("nan")
        dsc = (lambda k0, k1: sc[k1] - sc[k0])(min(sc), max(sc)) if sc else float("nan")
        dd = (lambda k0, k1: doms[k1] - doms[k0])(min(doms), max(doms)) if doms else float("nan")
        ddc = (lambda k0, k1: dc[k1] - dc[k0])(min(dc), max(dc)) if dc else float("nan")
        print(f"  {arm:<22} {ds:>+12.4f} {dsc:>+12.4f} {dd:>+13.4f} {ddc:>+13.4f}")
    print("\n  delta slice = old-domain CE (R active)   delta core = same, R=0 (genuine core forgetting)")
    print("  delta domain = new-domain CE (R active)  delta dCore = same, R=0 (core-only adaptation)")


if __name__ == "__main__":
    main()
