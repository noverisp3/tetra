"""Prototype test: Train a small Ternary LLM on synthetic data.

This script validates the entire pipeline:
1. Model initialization
2. Forward pass with ternary weights
3. Backward pass with STE
4. Gradient updates to latent weights
5. Loss decreases over training

Expected: Loss should decrease, confirming STE + ternary quantization works.
"""
import torch
import torch.nn.functional as F
from ternary_llm import TernaryTransformerModel


def create_synthetic_data(
    vocab_size: int = 256,
    seq_len: int = 32,
    num_samples: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create simple sequential pattern data.

    Pattern: token i → token (i+1) % vocab_size
    """
    data = torch.zeros(num_samples, seq_len + 1, dtype=torch.long)
    for i in range(num_samples):
        start = torch.randint(0, vocab_size, (1,)).item()
        data[i] = torch.arange(start, start + seq_len + 1) % vocab_size

    input_ids = data[:, :-1]
    targets = data[:, 1:]
    return input_ids, targets


def train_step(
    model: TernaryTransformerModel,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Single training step."""
    model.train()
    optimizer.zero_grad()

    logits, loss = model(input_ids, targets)
    loss.backward()

    # Gradient clipping (important for ternary training)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    return loss.item()


def main():
    # Hyperparameters (tiny model for testing)
    VOCAB_SIZE = 256
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    NUM_HEADS = 4
    FFN_DIM = 128
    SEQ_LEN = 32
    BATCH_SIZE = 16
    NUM_STEPS = 50
    LR = 3e-4

    print("=" * 60)
    print("Ternary LLM Prototype Test")
    print("=" * 60)

    # Create model
    model = TernaryTransformerModel(
        vocab_size=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ffn_dim=FFN_DIM,
        max_seq_len=SEQ_LEN,
    )

    total_params = sum(p.numel() for p in model.parameters())
    ternary_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if "latent_weights" in name
    )
    print(f"\nModel Info:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Ternary parameters: {ternary_params:,}")
    print(f"  Ternary bits: ~{ternary_params * 1.58 / 1e6:.2f} MB (at 1.58 bits)")

    # Create data
    input_ids, targets = create_synthetic_data(
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        num_samples=200,
    )

    # Split train/val
    train_ids, train_targets = input_ids[:160], targets[:160]
    val_ids, val_targets = input_ids[160:], targets[160:]

    # Optimizer (AdamW, following BitNet)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)

    # Training loop
    print(f"\nTraining for {NUM_STEPS} steps...")
    print("-" * 60)

    losses = []
    for step in range(NUM_STEPS):
        # Sample batch
        idx = torch.randint(0, len(train_ids), (BATCH_SIZE,))
        batch_ids = train_ids[idx]
        batch_targets = train_targets[idx]

        # Train
        loss = train_step(model, batch_ids, batch_targets, optimizer)
        losses.append(loss)

        # Print progress
        if (step + 1) % 10 == 0:
            avg_loss = sum(losses[-10:]) / 10
            print(f"  Step {step+1:3d}/{NUM_STEPS} | Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits, val_loss = model(val_ids, val_targets)

    print("-" * 60)
    print(f"Final train loss: {losses[-1]:.4f}")
    print(f"Validation loss: {val_loss.item():.4f}")

    # Verify ternary weights
    print("\nVerifying ternary weights:")
    for name, param in model.named_parameters():
        if "latent_weights" in name:
            w_ternary = param.data.clone()
            # Quantize
            gamma = w_ternary.abs().mean()
            w_ternary = (w_ternary / gamma).clamp(-1, 1).round()
            unique = w_ternary.unique().tolist()
            zeros_pct = (w_ternary == 0).float().mean() * 100
            print(f"  {name}: values={unique}, zeros={zeros_pct:.1f}%")

    # Test generation
    print("\nTesting generation:")
    model.eval()
    prompt = torch.randint(0, VOCAB_SIZE, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"  Prompt:     {prompt[0].tolist()}")
    print(f"  Generated:  {generated[0].tolist()}")

    # Check if loss decreased
    early_loss = sum(losses[:5]) / 5
    late_loss = sum(losses[-5:]) / 5
    if late_loss < early_loss:
        print("\n[PASS] Loss decreased - STE + ternary quantization working!")
    else:
        print("\n[WARN] Loss did not decrease - may need hyperparameter tuning")

    print("\n" + "=" * 60)
    print("Prototype test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
