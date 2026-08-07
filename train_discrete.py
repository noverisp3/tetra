"""Train Tetra with a pure discrete (gradient-free) learning rule.

Unlike ``train.py`` (global backprop), this script drives the existing
stochastic bit-flip accumulators with LOCAL update rules and never builds an
autograd graph for the ternary weights — training memory is O(1) in depth.

Usage:
    python train_discrete.py --rule c --steps 1000 --data-cache tinydata
    python train_discrete.py --rule b --steps 500 --random-data 20000
    python train_discrete.py --rule p --steps 500 --data-cache tinydata --aux 32
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from ternary_llm.data import create_dataloaders
from ternary_llm.discrete import (
    DiscreteConfig, DiscreteTrainer, random_token_array,
)

PRESETS = {
    "tiny":   dict(hidden_dim=256, num_layers=6,  num_heads=8,  ffn_dim=1024),
    "medium": dict(hidden_dim=512, num_layers=12, num_heads=8,  ffn_dim=2048),
    "large":  dict(hidden_dim=768, num_layers=12, num_heads=12, ffn_dim=2048),
    "500m":   dict(hidden_dim=2560, num_layers=6,  num_heads=40, ffn_dim=6826),
}


def main():
    parser = argparse.ArgumentParser(description="Train Tetra with discrete local rules")
    parser.add_argument("--preset", type=str, default=None, choices=list(PRESETS))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--rule", type=str, default="c",
                        choices=["c", "b", "p", "h", "e"],
                        help="c=predictive coding (default), b=forward-forward, "
                             "p=target PC (aux heads), h=hebbian (exp), e=entropy (exp)")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--data-cache", type=str, default=None,
                        help="Directory with metadata.json + tinystories.bin")
    parser.add_argument("--random-data", type=int, default=None,
                        help="Train on N synthetic random tokens (smoke test)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--threshold", type=float, default=20.0)
    parser.add_argument("--threshold-decay-to", type=float, default=None)
    parser.add_argument("--flip-every-n", type=int, default=5)
    parser.add_argument("--acc-decay", type=float, default=0.99,
                        help="Leaky accumulator decay per step (0.99 recommended)")
    parser.add_argument("--adaptive-thr", type=float, default=None,
                        help="Exp 3: adaptive flip threshold k (tau = k*RMS(acc) "
                             "per channel). None = fixed scalar threshold")
    parser.add_argument("--rule-energy", action="store_true",
                        help="Exp 4: keep gradient magnitude in local deltas "
                             "(-grad instead of -sign(grad)) so the accumulator "
                             "holds real energy (needed for adaptive threshold "
                             "to select a heavy tail)")
    parser.add_argument("--init", type=str, default="default",
                        choices=["default", "balanced"],
                        help="Ternary init: default (75%/-1) or balanced (33/33/33)")
    parser.add_argument("--toggle", action="store_true",
                        help="Anti-stiction: kick saturated weights to opposite extreme")
    parser.add_argument("--zero-center", type=float, default=0.0, metavar="FRAC",
                        help="Release +/-1 weights with |acc| < FRAC*threshold back to 0")
    parser.add_argument("--no-train-embedding", action="store_true",
                        help="Freeze the FP32 embedding (not recommended)")
    parser.add_argument("--lr-embedding", type=float, default=1e-4)
    parser.add_argument("--wd-embedding", type=float, default=0.1,
                        help="Decoupled weight decay for the embedding (keeps ||E|| bounded)")
    parser.add_argument("--aux", type=int, default=32, metavar="LATENT",
                        help="Bottleneck aux-head latent dim for --rule p")
    parser.add_argument("--aux-lr", type=float, default=3e-4)
    parser.add_argument("--ff-corrupt", type=float, default=0.3,
                        help="Corruption fraction for --rule b negative pass")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--save-dir", type=str, default="checkpoints_discrete")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Phase-1 checkpoint to continue from (local-rule "
                             "fine-tuning / continual learning). Accumulators "
                             "are reset on load.")
    parser.add_argument("--logit-scale", type=float, default=None,
                        help="Logit scale for CE / embedding gradient (default: "
                             "1/sqrt(hidden_dim); use 1.0 to continue a backprop "
                             "checkpoint in its natural logit regime)")
    args = parser.parse_args()

    # Seed BEFORE loaders/model so data order + embedding/ternary init are
    # reproducible across runs (trainer re-seeds internally, but the model is
    # built at construction, after the samplers are created otherwise).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = DiscreteConfig(
        rule=args.rule,
        block_size=args.block_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        max_steps=args.steps,
        device=args.device,
        threshold=args.threshold,
        threshold_decay_to=args.threshold_decay_to,
        flip_every_n_steps=args.flip_every_n,
        acc_decay=args.acc_decay,
        adaptive_thr=args.adaptive_thr,
        rule_energy=args.rule_energy,
        toggle=args.toggle,
        zero_center=args.zero_center,
        init_mode=args.init,
        train_embedding=not args.no_train_embedding,
        lr_embedding=args.lr_embedding,
        wd_embedding=args.wd_embedding,
        aux_latent_dim=args.aux,
        aux_lr=args.aux_lr,
        ff_corrupt=args.ff_corrupt,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        seed=args.seed,
        save_dir=args.save_dir,
        logit_scale=args.logit_scale,
    )

    if args.preset:
        for k, v in PRESETS[args.preset].items():
            setattr(config, k, v)
        print(f"Using preset: {args.preset}")

    # Data
    if args.random_data is not None:
        print(f"Using {args.random_data:,} synthetic random tokens")
        tokens = random_token_array(args.random_data, config.vocab_size, seed=args.seed)
        train_loader, val_loader = create_dataloaders(
            tokens, block_size=config.block_size, batch_size=config.batch_size,
            val_split=config.val_split, num_workers=0,
        )
    else:
        data_cache = Path(args.data_cache) if args.data_cache else Path("tinydata")
        manifest_path = data_cache / "manifest.json"
        meta_path = data_cache / "metadata.json"
        if manifest_path.exists():
            print(f"\nLoading multi-source data from {data_cache}...")
            from ternary_llm.data import create_multi_source_dataloaders
            with open(manifest_path) as f:
                manifest = json.load(f)
            config.vocab_size = manifest["vocab_size"]
            print(f"Sources: {list(manifest['sources'].keys())} | "
                  f"Total tokens: {manifest['total_tokens']:,}")
            train_loader, val_loader = create_multi_source_dataloaders(
                data_dir=data_cache, block_size=config.block_size,
                batch_size=config.batch_size, val_split=config.val_split,
                num_workers=0,
            )
        elif meta_path.exists() and (data_cache / "tinystories.bin").exists():
            with open(meta_path) as f:
                meta = json.load(f)
            config.vocab_size = meta["vocab_size"]
            tokens = np.memmap(str(data_cache / "tinystories.bin"), dtype=np.uint16, mode="r")
            print(f"Tokens: {len(tokens):,} | Vocab: {config.vocab_size}")
            train_loader, val_loader = create_dataloaders(
                tokens, block_size=config.block_size, batch_size=config.batch_size,
                val_split=config.val_split, num_workers=0,
            )
        else:
            print(f"ERROR: no manifest.json or metadata.json+tinystories.bin in {data_cache}")
            sys.exit(1)

    print(f"Rule: {config.rule} | Threshold: {config.threshold} "
          f"-> {config.threshold_decay_to} | flip every {config.flip_every_n_steps} "
          f"| acc decay {config.acc_decay} | adaptive thr {config.adaptive_thr} "
          f"| rule energy {config.rule_energy} | init {config.init_mode} "
          f"| toggle {config.toggle} | zero-center {config.zero_center}")
    print(f"Model: hidden={config.hidden_dim} layers={config.num_layers} "
          f"heads={config.num_heads} ffn={config.ffn_dim}")

    trainer = DiscreteTrainer(config, train_loader, val_loader)
    n_ternary = sum(m.accumulator.numel() for m in trainer._linear_map.values())
    print(f"Trainable ternary weights: {n_ternary:,} (2-bit packed)")
    if config.rule == "p":
        n_aux = sum(p.numel() for p in trainer.aux_heads.parameters())
        print(f"Aux bottleneck heads: {n_aux:,} params")

    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupted, saving...")

    trainer.save_checkpoint(trainer.step)
    print("Ternary distribution after training:")
    trainer.report_distribution()
    if trainer.val_losses:
        print(f"Final val CE: {trainer.val_losses[-1]:.4f}")
    if trainer.train_losses:
        print(f"Final train CE: {trainer.train_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
