"""Main training script for Tetra (Ternary LLM).

Usage:
    python train.py                              # Train with default (tiny) config
    python train.py --preset 500m --steps 10000  # Train 500M config
    python train.py --resume                     # Auto-resume from latest checkpoint
    python train.py --resume checkpoints/checkpoint_000500.pt  # Resume from specific
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ternary_llm.transformer import TernaryTransformerModel, StochasticTransformerModel
from ternary_llm.data import (
    download_and_tokenize, create_dataloaders,
    create_multi_source_dataloaders, get_tokenizer_compat,
)
from ternary_llm.trainer import TernaryTrainer, TrainingConfig

PRESETS = {
    "tiny":   dict(hidden_dim=128, num_layers=4,  num_heads=4,  ffn_dim=512),
    "medium": dict(hidden_dim=512, num_layers=12, num_heads=8,  ffn_dim=2048),
    "large":  dict(hidden_dim=768, num_layers=12, num_heads=12, ffn_dim=2048),
    "500m":   dict(hidden_dim=2560, num_layers=6,  num_heads=40, ffn_dim=6826),
}


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
                        help="[Hybrid] Place attention every N blocks (default: 5 → 80% SSM, 20% attention)")
    parser.add_argument("--expand-factor", type=int, default=2,
                        help="[Hybrid] SSM expansion factor (default: 2)")
    parser.add_argument("--ternary-scale", type=float, default=0.7,
                        help="[STE] Dynamic threshold scale: delta = scale x mean(|W|) (default: 0.7)")
    parser.add_argument("--per-channel", action="store_true",
                        help="[STE] Per-channel quantization threshold (instead of per-tensor)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="[Stochastic] Bit-flip threshold (default: 20.0 / scale, auto-computed)")
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

    config = TrainingConfig()

    # Apply preset
    if args.preset:
        preset = PRESETS[args.preset]
        for k, v in preset.items():
            setattr(config, k, v)
        print(f"  Using preset: {args.preset} ({sum(v*v*4 for v in [preset['hidden_dim']]*3):,}+ params)")

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
    if args.debug:
        config.debug = True
    if args.dtype:
        config.dtype = args.dtype

    # Step 1: Prepare data
    if args.data_cache:
        print("\n[1/4] Loading multi-source data...")
        data_cache = Path(args.data_cache)
        if not data_cache.exists():
            print(f"  ERROR: {data_cache} not found. Run prepare_data.py first.")
            sys.exit(1)
        import json
        with open(data_cache / "manifest.json") as f:
            manifest = json.load(f)
        config.vocab_size = manifest["vocab_size"]
        print(f"  Sources: {list(manifest['sources'].keys())}")
        print(f"  Total tokens: {manifest['total_tokens']:,}")
    else:
        print("\n[1/4] Preparing data (TinyStories)...")
        tokens, metadata = download_and_tokenize(
            cache_dir=config.data_dir,
            tokenizer_dir=args.tokenizer_dir,
            max_stories=args.max_stories,
        )
        config.vocab_size = metadata["vocab_size"]

    # Step 2: Create dataloaders
    print("\n[2/4] Creating dataloaders...")
    if args.data_cache:
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
    print("\n[3/4] Creating model...")
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
        print(f"  Mode: Hybrid ({ssm_layers}× SSM + {attn_layers}× Attention)")
        print(f"  Total params: {total_params:,}")
        print(f"  Ternary params: {ternary_params:,} ({ternary_params / 8 / 1024:.0f} KB packed)")
    elif is_stochastic:
        ternary_params = sum(
            p.numel() for n, p in model.named_buffers()
            if "packed_weights" in n
        ) * 2  # 2 bits per weight
        print(f"  Mode: Stochastic Bit-Flip (no latent weights)")
        print(f"  Total params: {total_params:,}")
        print(f"  Ternary params: {ternary_params:,} ({ternary_params / 8 / 1024:.0f} KB packed)")
        print(f"  FP32 params: {total_params:,} ({(total_params) * 4 / 1024:.0f} KB)")
    else:
        ternary_params = sum(
            p.numel() for name, p in model.named_parameters()
            if "latent_weights" in name
        )
        print(f"  Mode: STE (latent weights)")
        print(f"  Total params: {total_params:,}")
        print(f"  Ternary params: {ternary_params:,} ({ternary_params * 2 / 8 / 1024:.0f} KB packed)")
        print(f"  FP32 params: {total_params - ternary_params:,} ({(total_params - ternary_params) * 4 / 1024:.0f} KB)")

    # Step 4: Train
    print("\n[4/4] Starting training...")
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
                print("  No checkpoints found, starting fresh")
                resume_path = None
        else:
            resume_path = args.resume
        if resume_path:
            start_step = trainer.load_checkpoint(resume_path)
        else:
            start_step = 0
    else:
        start_step = 0

    trainer.train(resume_step=start_step)

    # Export training loss graph
    if args.graph and trainer.train_losses:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            steps = list(range(1, len(trainer.train_losses) + 1))
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(steps, trainer.train_losses, color="#4a90d9", linewidth=1.5)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.grid(alpha=0.3)
            if trainer.learning_rates:
                plt.subplot(1, 2, 2)
                plt.plot(steps, trainer.learning_rates, color="#e67e22", linewidth=1.5)
                plt.xlabel("Step")
                plt.ylabel("LR")
                plt.title("Learning Rate")
                plt.grid(alpha=0.3)
            plt.tight_layout()
            graph_path = Path(config.save_dir) / "loss_plot.png"
            plt.savefig(graph_path, dpi=150)
            plt.close()
            print(f"  [GRAPH] Saved to {graph_path}")
        except ImportError:
            print("  [GRAPH] matplotlib not installed, skipping")

    # Generate sample
    print("\n[Sample Generation]")
    import torch
    enc = get_tokenizer_compat(args.tokenizer_dir)
    prompt = "Hello"
    prompt_ids = enc.encode(prompt)
    prompt_tensor = torch.tensor([prompt_ids], device=trainer.device)
    model.eval()
    with torch.no_grad():
        output = model.generate(
            prompt_tensor,
            max_new_tokens=200,
            temperature=0.8,
            top_k=50,
        )
    generated = enc.decode(output[0].tolist())
    print(f"\n{generated}\n")


if __name__ == "__main__":
    main()
