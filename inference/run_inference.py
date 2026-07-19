"""Tetra Inference Runner — bridges custom BPE tokenizer with C++ engine.

Usage:
    python inference/run_inference.py <model.bin> "Once upon a time"
    python inference/run_inference.py <model.bin> "Once upon a time" --max-tokens 200 --temp 0.8 --top-k 50 --top-p 0.9
"""
import sys
import subprocess
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ternary_llm.data import get_tokenizer_compat


BOS_TOKEN = 1
EOS_TOKEN = 2


def run_inference(model_path, prompt, max_tokens=100, temperature=0.8,
                  top_k=50, top_p=0.9, tokenizer_dir="tokenizer"):
    enc = get_tokenizer_compat(tokenizer_dir)

    # Tokenize (no BOS/EOS — C++ handles them)
    tokens = enc.encode(prompt)
    token_str = ",".join(str(t) for t in tokens)
    print(f"Prompt: {prompt}")
    print(f"Tokens: {tokens}")

    # Run C++ binary
    exe_path = Path(__file__).parent / "tetra.exe"
    if not exe_path.exists():
        exe_path = Path(__file__).parent / "tetra"

    if not exe_path.exists():
        print(f"\nC++ binary not found at {exe_path}")
        print("Build it first: cd inference && build.bat")
        print("\nFalling back to Python inference...\n")
        return python_inference(model_path, tokens, max_tokens, temperature, top_k)

    cmd = [
        str(exe_path), model_path, token_str,
        str(max_tokens), str(temperature), str(top_k), str(top_p),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)

    # Stream: print generated token IDs line
    for line in result.stdout.split("\n"):
        if line.startswith("Output token IDs:"):
            ids_str = line.split(":", 1)[1].strip()
            all_tokens = [int(x) for x in ids_str.split(",")]

            # Separate prompt from generated
            prompt_len = len(tokens) + 1  # +1 for BOS prepended by C++
            gen_tokens = all_tokens[prompt_len:]

            # Decode generated tokens (filter out EOS)
            gen_tokens_clean = [t for t in gen_tokens if t != EOS_TOKEN]
            generated = enc.decode(gen_tokens_clean)

            print(f"\n{'='*60}")
            print(f"Generated text:\n")
            print(generated)
            print(f"{'='*60}")
            return generated


def python_inference(model_path, prompt_tokens, max_tokens, temperature, top_k=50):
    """Fallback: pure Python inference using checkpoint."""
    import torch
    from ternary_llm.transformer import TernaryTransformerModel

    enc = get_tokenizer_compat()

    candidates = [
        "checkpoints/checkpoint_latest.pt",
    ]

    model = None
    for c in candidates:
        if Path(c).exists():
            ckpt = torch.load(c, map_location="cpu", weights_only=False)
            config = ckpt["config"]
            model = TernaryTransformerModel(
                vocab_size=config["vocab_size"],
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
                ffn_dim=config["ffn_dim"],
                max_seq_len=config["max_seq_len"],
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            print(f"Loaded from {c}")
            break

    if model is None:
        print("ERROR: No checkpoint found")
        return

    prompt_t = torch.tensor([prompt_tokens])
    with torch.no_grad():
        out = model.generate(prompt_t, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)

    generated = enc.decode(out[0].tolist())
    print(f"\n{'='*60}")
    print(f"Generated text:\n")
    print(generated)
    print(f"{'='*60}")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Tetra Inference Runner")
    parser.add_argument("model", help="Path to tetra_model.bin")
    parser.add_argument("prompt", help="Input text prompt")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer")
    args = parser.parse_args()

    run_inference(args.model, args.prompt, args.max_tokens, args.temp,
                  args.top_k, args.top_p, args.tokenizer_dir)


if __name__ == "__main__":
    main()
