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
                        help="[Hybrid] Place attention every N blocks (default: 5 -> 80% SSM, 20% attention)")
    parser.add_argument("--expand-factor", type=int, default=2,
                        help="[Hybrid] SSM expansion factor (default: 2)")
    parser.add_argument("--ternary-scale", type=float, default=0.7,
                        help="[STE] Dynamic threshold scale: delta = scale x mean(|W|) (default: 0.7)")
    parser.add_argument("--per-channel", action="store_true",
                        help="[STE] Per-channel quantization threshold (instead of per-tensor)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="[Stochastic] Bit-flip threshold (default: 20.0 / scale, auto-computed)")
    parser.add_argument("--threshold-decay-to", type=float, default=None,
                        help="[Stochastic] Decay threshold to this value by end of training (default: same as --threshold, no decay)")
    parser.add_argument("--int8", action="store_true",
                        help="[Stochastic] Use INT8 forward matmul (quantize activations to int8)")
    parser.add_argument("--topk", type=float, default=None,
                        help="Keep top-k fraction of activations after norm (e.g. 0.2 = 20%%, default: 1.0 = off)")
    parser.add_argument("--flip-every-n-steps", type=int, default=5,
                        help="[Stochastic] Check threshold & flip bits every N optimizer steps (default: 5)")
    parser.add_argument("--graph", action="store_true",
                        help="Export training loss plot to checkpoints/loss_plot.png")
    parser.add_argument("--debug", action="store_true",
                        help="Print MEM/TIME diagnostics")
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "float16", "bfloat16"],
                        help="Training dtype: float32 (default), float16, or bfloat16 (CUDA)")
    args = parser.parse_args()

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
    config.flip_every_n_steps = args.flip_every_n_steps
    config.threshold = args.threshold if args.threshold is not None else 20.0
    if args.threshold_decay_to is not None:
        config.threshold_decay_to = args.threshold_decay_to
    if args.debug:
        config.debug = True
    if args.dtype:
        config.dtype = args.dtype

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
        )

    total_params = sum(p.numel() for p in model.parameters())
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
    print(f"\n{generated}\n")


if __name__ == "__main__":
    main()
