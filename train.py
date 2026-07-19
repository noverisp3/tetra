"""Main training script for Tetra (Ternary LLM).

Usage:
    python train.py                    # Train with default config
    python train.py --steps 50000      # Train for 50000 steps
    python train.py --resume checkpoint_latest.pt
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ternary_llm.transformer import TernaryTransformerModel
from ternary_llm.data import download_and_tokenize, create_dataloaders, get_tokenizer_compat
from ternary_llm.trainer import TernaryTrainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Train Tetra")
    parser.add_argument("--steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--block-size", type=int, default=None, help="Block size (context length)")
    parser.add_argument("--hidden-dim", type=int, default=None, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--ffn-dim", type=int, default=None, help="FFN intermediate dim")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer", help="Tokenizer directory")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--max-stories", type=int, default=None, help="Max stories to load")
    args = parser.parse_args()

    config = TrainingConfig()

    # Override from args
    if args.steps:
        config.max_steps = args.steps
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
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

    # Step 1: Download and tokenize data
    print("\n[1/4] Preparing data...")
    tokens, metadata = download_and_tokenize(
        cache_dir=config.data_dir,
        tokenizer_dir=args.tokenizer_dir,
        max_stories=args.max_stories,
    )
    config.vocab_size = metadata["vocab_size"]

    # Step 2: Create dataloaders
    print("\n[2/4] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        tokens=tokens,
        block_size=config.block_size,
        batch_size=config.batch_size,
        val_split=config.val_split,
    )

    # Step 3: Create model
    print("\n[3/4] Creating model...")
    model = TernaryTransformerModel(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        max_seq_len=config.max_seq_len,
    )

    total_params = sum(p.numel() for p in model.parameters())
    ternary_params = sum(
        p.numel() for name, p in model.named_parameters()
        if "latent_weights" in name
    )
    emb_params = sum(p.numel() for p in model.parameters() if "embedding" in p.shape.__repr__() or "embed" in "")
    print(f"  Total params: {total_params:,}")
    print(f"  Ternary params: {ternary_params:,} ({ternary_params * 2 / 8 / 1024:.1f} KB packed)")
    print(f"  FP32 params: {total_params - ternary_params:,}")

    # Step 4: Train
    print("\n[4/4] Starting training...")
    trainer = TernaryTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    if args.resume:
        start_step = trainer.load_checkpoint(args.resume)
    else:
        start_step = 0

    trainer.train(resume_step=start_step)

    # Generate sample
    print("\n[Sample Generation]")
    import torch
    enc = get_tokenizer_compat(args.tokenizer_dir)
    prompt = "Once upon a time"
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
