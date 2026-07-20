"""Compare stochastic Bit-Flip vs STE convergence."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from ternary_llm.transformer import StochasticTransformerModel, TernaryTransformerModel
from ternary_llm.data import ChunkedDataset
from torch.utils.data import DataLoader

vocab, bs, seq, steps = 256, 8, 32, 300
rng = np.random.RandomState(42)
tokens = rng.randint(1, vocab, size=200000).astype(np.int64)
ds = ChunkedDataset(tokens, seq)

def run(label, model, loader):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses, best = [], float("inf")
    for i, (x, y) in enumerate(loader):
        if i >= steps: break
        _, loss, _ = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        best = min(best, loss.item())
    delta = losses[-1] - losses[0]
    print(f"  {label}: start={losses[0]:.2f} best={best:.2f} end={losses[-1]:.2f} delta={delta:+.2f}")

print("=" * 55)
loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=0)
run("STE", TernaryTransformerModel(vocab, 128, 2, 4, 512, max_seq_len=seq, ternary_scale=1.0), loader)

for th in [2, 5, 10, 20, 50]:
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=0)
    run(f"Stochastic(th={th})", StochasticTransformerModel(vocab, 128, 2, 4, 512, max_seq_len=seq, threshold=float(th)), loader)
