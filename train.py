"""Main training script for Tetra (Ternary LLM).

Usage:
    python train.py                              # Train with default (tiny) config
    python train.py --preset 500m --steps 10000  # Train 500M config
    python train.py --resume                     # Auto-resume from latest checkpoint
    python train.py --resume checkpoints/checkpoint_000500.pt  # Resume from specific
    python train.py --data-cache tinydata        # Use pre-tokenized .bin cache (tinydata/)
"""
import json
import sys
import time
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ternary_llm.transformer import TernaryTransformerModel, StochasticTransformerModel
from ternary_llm.data import (
    download_and_tokenize, create_dataloaders,
    create_multi_source_dataloaders, get_tokenizer_compat,
)
from ternary_llm.trainer import TernaryTrainer, TrainingConfig

PRESETS = {
    "tiny":   dict(hidden_dim=256, num_layers=6,  num_heads=8,  ffn_dim=1024),
    "medium": dict(hidden_dim=512, num_layers=12, num_heads=8,  ffn_dim=2048),
    "large":  dict(hidden_dim=768, num_layers=12, num_heads=12, ffn_dim=2048),
    "500m":   dict(hidden_dim=2560, num_layers=6,  num_heads=40, ffn_dim=6826),
}


def export_graph(trainer, save_dir):
    """Export training loss/learning-rate plot using logged history."""
    if not trainer.train_losses:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed, skipping graph")
        return

    losses = np.array(trainer.train_losses)
    if len(trainer.train_log_steps) == len(losses):
        steps = trainer.train_log_steps
    else:
        steps = list(range(1, len(losses) + 1))

    # EMA smoothing (alpha=0.85)
    def smooth(y, alpha=0.85):
        s = np.copy(y)
        for i in range(1, len(s)):
            s[i] = alpha * s[i - 1] + (1 - alpha) * s[i]
        return s

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Training Loss
    ax = axes[0]
    ax.plot(steps, losses, color="#b0c4de", linewidth=1, alpha=0.5, label="Raw")
    ax.plot(steps, smooth(losses), color="#4a90d9", linewidth=2, label="Smoothed")
    if trainer.val_losses:
        val_idx = np.linspace(0, len(losses) - 1, len(trainer.val_losses), dtype=int)
        ax.plot(np.array(steps)[val_idx], trainer.val_losses,
                color="#e74c3c", linewidth=2, marker="o", markersize=4, label="Val")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Right: Learning Rate
    ax = axes[1]
    lr = trainer.learning_rates
    if lr:
        lr_steps = steps[:len(lr)] if len(lr) <= len(steps) else list(range(1, len(lr) + 1))
        ax.plot(lr_steps, lr, color="#e67e22", linewidth=2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    graph_path = Path(save_dir) / "loss_plot.png"
    plt.savefig(graph_path, dpi=150)
    plt.close()
    print(f"Training graph saved to {graph_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Tetra")
    parser.add_argument("--preset", type=str, default=None, choices=["tiny", "medium", "large", "500m"],
                        help="Model size preset (overrides hidden/layers/heads/ffn)")
    parser.add_argument("--steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--block-size", type=int, default=None, help="Block size (context length)")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--ffn-dim", type=int, default=None, help="FFN intermediate dim")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="Resume from checkpoint (no arg = auto-find latest)")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory (legacy TinyStories)")
    parser.add_argument("--data-cache", type=str, default=None, help="Multi-source data dir (data/)")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer", help="Tokenizer directory")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--max-stories", type=int, default=None, help="Max stories to load")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "directml"],
                        help="Force device (default: auto-detect)")
    parser.add_argument("--hybrid", action="store_true",
                        help="Hybrid mode: model on GPU, optimizer on CPU (avoids DML fallbacks)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers for async prefetch (default: 4)")
    parser.add_argument("--mode", type=str, default="ste", choices=["ste", "stochastic", "hybrid"],
                        help="Training mode: STE, Stochastic Bit-Flip, or Hybrid SSM-Attention (default: ste)")
    parser.add_argument("--ssm-every", type=int, default=5,
                        help="[Hybrid] Place attention every N blocks (default: 5 -> 80%% SSM, 20%% attention)")
    parser.add_argument("--expand-factor", type=int, default=2,
                        help="[Hybrid] SSM expansion factor (default: 2)")
    parser.add_argument("--ternary-scale", type=float, default=1.0,
                        help="[STE] Dynamic threshold scale: delta = scale x mean(|W|) (default: 1.0, "
                             "tuned in Exp 9: lower CE + fewer +-2 outliers)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for model init / data order (default: unseeded)")
    parser.add_argument("--per-channel", action="store_true",
                        help="[STE] Per-channel quantization threshold (instead of per-tensor)")
    parser.add_argument("--group-size", type=int, default=0,
                        help="Block size for per-group scaling alphas (STE + Stochastic) (default: 0 = off)")
    parser.add_argument("--soft-quant-gamma", action="store_true",
                        help="[STE] Exp 8: soft-to-hard sigmoid-surrogate quantization warmup "
                             "(gamma 2->50 over warmup, then hard round+STE)")
    parser.add_argument("--soft-quant-steps", type=int, default=0, metavar="N",
                        help="[STE] Exp 8: warmup length for the soft surrogate (default: 25%% of --steps)")
    parser.add_argument("--soft-quant-gamma-init", type=float, default=2.0, metavar="G",
                        help="[STE] Exp 8: starting surrogate temperature (default: 2.0)")
    parser.add_argument("--soft-quant-gamma-max", type=float, default=50.0, metavar="G",
                        help="[STE] Exp 8: final temperature at end of warmup, then hard STE (default: 50.0)")
    parser.add_argument("--init", type=str, default="kaiming", choices=["kaiming", "balanced"],
                        help="[STE] Latent init: kaiming (default) or balanced 33/33/33 ternary (anti rank-collapse)")
    parser.add_argument("--ortho-reg", type=float, default=0.0, metavar="LAMBDA",
                        help="[STE] Orthogonalization penalty weight on latent rows (0 = off)")
    parser.add_argument("--rank-monitor-interval", type=int, default=500,
                        help="Report unique ternary rows per matrix every N steps (0 = off)")
    parser.add_argument("--rank-halt", action="store_true",
                        help="Halt training when a matrix collapses (unique_rows <= rows/4)")
    parser.add_argument("--save-best", action="store_true",
                        help="Keep checkpoint_best.pt (lowest validation loss)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="[Stochastic] Bit-flip threshold (default: 20.0 / scale, auto-computed)")
    parser.add_argument("--threshold-decay-to", type=float, default=None,
                        help="[Stochastic] Decay threshold to this value by end of training (default: same as --threshold, no decay)")
    parser.add_argument("--int8", action="store_true",
                        help="[Stochastic] Use INT8 forward matmul (quantize activations to int8)")
    parser.add_argument("--topk", type=float, default=None,
                        help="Keep top-k fraction of activations after norm (e.g. 0.2 = 20%%, default: 1.0 = off)")
    parser.add_argument("--outlier-thr-mult", type=float, default=3.0,
                        help="[Stochastic] Promote weight to ±2 outlier when |acc| exceeds this × threshold (default: 3.0)")
    parser.add_argument("--flip-every-n-steps", type=int, default=5,
                        help="[Stochastic] Check threshold & flip bits every N optimizer steps (default: 5)")
    parser.add_argument("--flip-ungated", action="store_true",
                        help="[Stochastic] Flip every weight whose accumulator is non-zero (no surprise gate)")
    parser.add_argument("--acc-energy", action="store_true",
                        help="[Stochastic] Exp 3: energy accumulator (leaky EMA of -grad) instead of ±1 sign votes")
    parser.add_argument("--acc-decay", type=float, default=0.99,
                        help="[Stochastic] Exp 3: leaky accumulator decay per step (energy mode; default 0.99)")
    parser.add_argument("--adaptive-thr", type=float, default=None,
                        help="[Stochastic] Exp 3: adaptive flip threshold k (tau = k*RMS(acc) per channel). "
                             "None = fixed scalar threshold")
    parser.add_argument("--graph", action="store_true",
                        help="Export training loss plot to checkpoints/loss_plot.png")
    parser.add_argument("--debug", action="store_true",
                        help="Print MEM/TIME diagnostics")
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "float16", "bfloat16"],
                        help="Training dtype: float32 (default), float16, or bfloat16 (CUDA)")
    parser.add_argument("--mla", action="store_true",
                        help="Use Multi-head Latent Attention (MLA) with compressed KV cache")
    parser.add_argument("--kv-latent-dim", type=int, default=None,
                        help="MLA KV latent dimension (default: 2 * head_dim)")
    parser.add_argument("--rope-per-head", type=int, default=None,
                        help="MLA RoPE dimension per head (default: max(4, head_dim//4))")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Linear warmup steps (overrides preset)")
    parser.add_argument("--min-lr", type=float, default=None,
                        help="Floor for the cosine LR schedule (default: 1e-4)")
    parser.add_argument("--eval-interval", type=int, default=None,
                        help="Validation cadence (default: 500)")
    parser.add_argument("--eval-positions", type=int, default=None,
                        help="Tokens to score per held-out eval slice (default: 20000)")
    parser.add_argument("--eval-slice", type=str, default=None,
                        help="Held-out old-domain eval file (path to .bin) for slice CE")
    parser.add_argument("--domain-eval", type=str, default=None,
                        help="Held-out new-domain eval file (path to .bin) for domain CE")
    parser.add_argument("--erc", action="store_true",
                        help="Enable ERC: residual R (high LR) + periodic commits into the latent core")
    parser.add_argument("--erc-lr", type=float, default=None,
                        help="ERC fast residual learning rate (default: 0.01)")
    parser.add_argument("--erc-decay", type=float, default=None,
                        help="Leaky EMA factor applied to R each step (<1 fades old echoes)")
    parser.add_argument("--erc-commit-interval", type=int, default=None,
                        help="Commit R -> latent core every N optimizer steps (default: 10)")
    parser.add_argument("--erc-fp16", action="store_true",
                        help="Store/learn the residual in FP16 (short-term memory)")
    parser.add_argument("--erc-freeze-core", action="store_true",
                        help="Freeze latent cores entirely (residual-only learning)")
    args = parser.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Seed: {args.seed}")

    # Input validation
    if args.steps is not None and args.steps < 1:
        print("ERROR: --steps must be >= 1")
        sys.exit(1)
    if args.lr is not None and args.lr <= 0:
        print("ERROR: --lr must be positive")
        sys.exit(1)
    if args.batch_size is not None and args.batch_size < 1:
        print("ERROR: --batch-size must be >= 1")
        sys.exit(1)
    if args.grad_accum is not None and args.grad_accum < 1:
        print("ERROR: --grad-accum must be >= 1")
        sys.exit(1)
    if args.block_size is not None and args.block_size < 16:
        print("ERROR: --block-size must be >= 16")
        sys.exit(1)
    if args.hidden_dim is not None and args.hidden_dim < 64:
        print("ERROR: --hidden-dim must be >= 64")
        sys.exit(1)
    if args.num_layers is not None and args.num_layers < 1:
        print("ERROR: --num-layers must be >= 1")
        sys.exit(1)
    if args.num_heads is not None and args.num_heads < 1:
        print("ERROR: --num-heads must be >= 1")
        sys.exit(1)
    if args.ffn_dim is not None and args.ffn_dim < 64:
        print("ERROR: --ffn-dim must be >= 64")
        sys.exit(1)
    if args.ternary_scale is not None and args.ternary_scale <= 0:
        print("ERROR: --ternary-scale must be positive")
        sys.exit(1)
    if args.topk is not None and not (0 < args.topk <= 1):
        print("ERROR: --topk must be in (0, 1]")
        sys.exit(1)
    if args.num_workers is not None and args.num_workers < 0:
        print("ERROR: --num-workers must be >= 0")
        sys.exit(1)

    config = TrainingConfig()

    # Apply preset
    if args.preset:
        preset = PRESETS[args.preset]
        for k, v in preset.items():
            setattr(config, k, v)
        print(f"Using preset: {args.preset} ({sum(v*v*4 for v in [preset['hidden_dim']]*3):,}+ params)")

    # Override from args (highest priority)
    if args.steps:
        config.max_steps = args.steps
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.grad_accum:
        config.gradient_accumulation_steps = args.grad_accum
    if args.block_size:
        config.block_size = args.block_size
    if args.hidden_dim:
        config.hidden_dim = args.hidden_dim
    if args.num_layers:
        config.num_layers = args.num_layers
    if args.num_heads:
        config.num_heads = args.num_heads
    if args.ffn_dim:
        config.ffn_dim = args.ffn_dim
    config.data_dir = args.data_dir
    config.save_dir = args.save_dir
    if args.device:
        config.device = args.device
    if args.hybrid:
        config.hybrid_optimizer = True
    config.mode = args.mode
    config.ternary_scale = args.ternary_scale
    config.per_channel = args.per_channel
    config.group_size = args.group_size
    if args.soft_quant_gamma:
        config.soft_quant = True
        if args.soft_quant_steps > 0:
            config.soft_quant_steps = args.soft_quant_steps
        config.soft_quant_gamma_init = args.soft_quant_gamma_init
        config.soft_quant_gamma_max = args.soft_quant_gamma_max
        print(f"Exp 8 soft-to-hard quant: gamma {config.soft_quant_gamma_init} -> "
              f"{config.soft_quant_gamma_max} over {config.soft_quant_steps or '25% of steps'} steps")
    config.init_mode = args.init
    config.ortho_reg = args.ortho_reg
    config.rank_monitor_interval = args.rank_monitor_interval
    config.rank_halt = args.rank_halt
    config.save_best = args.save_best
    config.flip_every_n_steps = args.flip_every_n_steps
    config.threshold = args.threshold if args.threshold is not None else 20.0
    if args.threshold_decay_to is not None:
        config.threshold_decay_to = args.threshold_decay_to
    if args.debug:
        config.debug = True
    if args.dtype:
        config.dtype = args.dtype
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
    if args.min_lr is not None:
        config.min_lr = args.min_lr
    if args.eval_interval is not None:
        config.eval_interval = args.eval_interval
    if args.eval_positions is not None:
        config.eval_positions = args.eval_positions
    if args.eval_slice:
        config.eval_slice_path = args.eval_slice
    if args.domain_eval:
        config.domain_eval_path = args.domain_eval
    if args.erc:
        config.erc = True
        if args.erc_lr is not None:
            config.erc_lr = args.erc_lr
        if args.erc_decay is not None:
            config.erc_decay = args.erc_decay
        if args.erc_commit_interval is not None:
            config.erc_commit_interval = args.erc_commit_interval
        if args.erc_fp16:
            config.erc_residual_dtype = "fp16"
        if args.erc_freeze_core:
            config.erc_freeze_core = True
        print(f"ERC: residual at lr {config.erc_lr} (decay {config.erc_decay}, "
              f"commit every {config.erc_commit_interval} steps, "
              f"dtype {config.erc_residual_dtype}"
              f"{', core FROZEN' if config.erc_freeze_core else ''})")

    # Step 1: Prepare data
    if args.data_cache:
        data_cache = Path(args.data_cache)
        if not data_cache.exists():
            print(f"ERROR: {data_cache} not found. Run prepare_data.py first.")
            sys.exit(1)
        manifest_path = data_cache / "manifest.json"
        metadata_path = data_cache / "metadata.json"
        if manifest_path.exists():
            print("\nLoading multi-source data...")
            with open(manifest_path) as f:
                manifest = json.load(f)
            config.vocab_size = manifest["vocab_size"]
            print(f"Sources: {list(manifest['sources'].keys())}")
            print(f"Total tokens: {manifest['total_tokens']:,}")
            multi_source = True
        elif metadata_path.exists():
            print(f"\nLoading cached TinyStories from {data_cache}")
            with open(metadata_path) as f:
                meta = json.load(f)
            config.vocab_size = meta["vocab_size"]
            bin_file = data_cache / "tinystories.bin"
            tokens = np.memmap(str(bin_file), dtype=np.uint16, mode="r")
            print(f"Tokens: {len(tokens):,}")
            multi_source = False
        else:
            print(f"ERROR: no manifest.json or metadata.json found in {data_cache}")
            sys.exit(1)
    else:
        print("\nPreparing data (TinyStories)...")
        tokens, metadata = download_and_tokenize(
            cache_dir=config.data_dir,
            tokenizer_dir=args.tokenizer_dir,
            max_stories=args.max_stories,
        )
        config.vocab_size = metadata["vocab_size"]
        multi_source = False

    # Step 2: Create dataloaders
    print("\nCreating dataloaders...")
    if multi_source:
        train_loader, val_loader = create_multi_source_dataloaders(
            data_dir=args.data_cache,
            block_size=config.block_size,
            batch_size=config.batch_size,
            val_split=config.val_split,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        train_loader, val_loader = create_dataloaders(
            tokens=tokens,
            block_size=config.block_size,
            batch_size=config.batch_size,
            val_split=config.val_split,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # Step 3: Create model
    print("\nCreating model...")
    is_stochastic = args.mode == "stochastic"
    is_hybrid = args.mode == "hybrid"

    if is_hybrid:
        from ternary_llm.hybrid import HybridTransformerModel
        model = HybridTransformerModel(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            max_seq_len=config.max_seq_len,
            scale=config.ternary_scale,
            threshold=args.threshold,
            int8=args.int8,
            topk=args.topk if args.topk is not None else 1.0,
            expand_factor=args.expand_factor,
            ssm_every=args.ssm_every,
        )
    elif is_stochastic:
        if args.mla:
            from ternary_llm.transformer import StochasticMLAModel
            model = StochasticMLAModel(
                vocab_size=config.vocab_size,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                max_seq_len=config.max_seq_len,
                scale=config.ternary_scale,
                threshold=args.threshold,
                int8=args.int8,
                topk=args.topk if args.topk is not None else 1.0,
                per_channel=config.per_channel,
                group_size=config.group_size,
                kv_latent_dim=args.kv_latent_dim,
                rope_per_head=args.rope_per_head,
                outlier_thr_mult=args.outlier_thr_mult,
            )
        else:
            model = StochasticTransformerModel(
                vocab_size=config.vocab_size,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                max_seq_len=config.max_seq_len,
                scale=config.ternary_scale,
                threshold=args.threshold,
                int8=args.int8,
                topk=args.topk if args.topk is not None else 1.0,
                per_channel=config.per_channel,
                group_size=config.group_size,
                outlier_thr_mult=args.outlier_thr_mult,
            )
    else:
        model = TernaryTransformerModel(
            vocab_size=config.vocab_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ffn_dim=config.ffn_dim,
            max_seq_len=config.max_seq_len,
            ternary_scale=config.ternary_scale,
            per_channel=config.per_channel,
            topk=args.topk if args.topk is not None else 1.0,
            group_size=config.group_size,
            init_mode=config.init_mode,
        )

    total_params = sum(p.numel() for p in model.parameters())
    if is_stochastic and args.flip_ungated:
        from ternary_llm.layers import StochasticTernaryLinear
        n_gated = 0
        for m in model.modules():
            if isinstance(m, StochasticTernaryLinear):
                m.ungated = True
                n_gated += 1
        print(f"Flip gating: UNGATED on {n_gated:,} linear layers "
              f"(flip on every accumulator sign, no threshold)")
    if is_stochastic and (args.acc_energy or args.adaptive_thr is not None):
        model.set_flip_config(acc_decay=args.acc_decay, energy=args.acc_energy,
                              adaptive_thr=args.adaptive_thr)
        print(f"Exp 3 flip mechanics: energy acc={args.acc_energy} "
              f"(decay {args.acc_decay}) | adaptive thr k={args.adaptive_thr}")
    if is_hybrid:
        ternary_params = sum(
            p.numel() for n, p in model.named_buffers()
            if "packed_weights" in n
        ) * 2
        attn_layers = sum(1 for l in model.layers if l.is_attention)
        ssm_layers = sum(1 for l in model.layers if not l.is_attention)
        print(f"Mode: Hybrid ({ssm_layers}x SSM + {attn_layers}x Attention)")
        print(f"Total params: {total_params:,}")
        print(f"Ternary params: {ternary_params:,} ({ternary_params / 8 / 1024:.0f} KB packed)")
    elif is_stochastic:
        ternary_params = sum(
            p.numel() for n, p in model.named_buffers()
            if "packed_weights" in n
        ) * 2  # 2 bits per weight
        print(f"Mode: Stochastic Bit-Flip (no latent weights)")
        print(f"Total params: {total_params:,}")
        print(f"Ternary params: {ternary_params:,} ({ternary_params / 8 / 1024:.0f} KB packed)")
        print(f"FP32 params: {total_params:,} ({(total_params) * 4 / 1024:.0f} KB)")
    else:
        ternary_params = sum(
            p.numel() for name, p in model.named_parameters()
            if "latent_weights" in name
        )
        print(f"Mode: STE (latent weights)")
        print(f"Init: {config.init_mode} | ortho-reg: {config.ortho_reg} | "
              f"group-size: {config.group_size} | rank-monitor: {config.rank_monitor_interval}"
              f"{' (halt)' if config.rank_halt else ''} | save-best: {config.save_best}")
        print(f"Total params: {total_params:,}")
        print(f"Ternary params: {ternary_params:,} ({ternary_params * 2 / 8 / 1024:.0f} KB packed)")
        print(f"FP32 params: {total_params - ternary_params:,} ({(total_params - ternary_params) * 4 / 1024:.0f} KB)")

    # Step 4: Train
    print("\nStarting training...")
    trainer = TernaryTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    if args.resume:
        if args.resume == "auto":
            checkpoints = sorted(Path(config.save_dir).glob("checkpoint_*.pt"))
            if checkpoints:
                resume_path = str(checkpoints[-1])
            else:
                print("No checkpoints found, starting fresh")
                resume_path = None
        else:
            resume_path = args.resume
        if resume_path:
            start_step = trainer.load_checkpoint(resume_path)
        else:
            start_step = 0
    else:
        start_step = 0

    try:
        trainer.train(resume_step=start_step)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted, saving checkpoint...")
        step = trainer.scheduler.step_count
        trainer.save_checkpoint(step)
        if args.graph:
            export_graph(trainer, config.save_dir)
        print("Checkpoint saved. Exiting.")
        sys.exit(130)

    if is_stochastic:
        from ternary_llm.layers import StochasticTernaryLinear
        lin = [m for m in model.modules() if isinstance(m, StochasticTernaryLinear)]
        total_flips = sum(m.flip_count for m in lin)
        total_w = sum(m.accumulator.numel() for m in lin)
        n_apply = max(1, max(1, trainer.scheduler.step_count) // max(1, config.flip_every_n_steps))
        frac = total_flips / max(1, total_w * n_apply)
        print(f"\nFlip stats: {total_flips:,} total flips over ~{n_apply} flip passes "
              f"({total_w * n_apply:,} opportunities)")
        print(f"Avg % weights flipped per pass: {frac * 100:.2f}%")

    if args.graph:
        export_graph(trainer, config.save_dir)

    # Generate sample
    print("\nSample generation:")
    enc = get_tokenizer_compat(args.tokenizer_dir)
    prompt = "Hello"
    prompt_ids = enc.encode(prompt)
    prompt_tensor = torch.tensor([prompt_ids], device=trainer.device)
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            prompt_tensor,
            max_new_tokens=200,
            temperature=0.8,
            top_k=50,
        )
    gen_time = time.perf_counter() - t0
    n_prompt = prompt_tensor.size(1)
    n_gen = output.size(1) - n_prompt
    generated = enc.decode(output[0].tolist())
    print(f"Prompt: {n_prompt} tokens -> Generated: {n_gen} tokens in {gen_time:.2f}s ({n_gen/gen_time:.1f} tok/s)")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"\n{generated}\n")


if __name__ == "__main__":
    main()
