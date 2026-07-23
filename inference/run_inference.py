"""Tetra Inference Runner."""
import sys
import subprocess
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ternary_llm.data import get_tokenizer_compat


EOS_TOKEN = 2


def find_exe() -> Path | None:
    # Try optimized builds first, then scalar fallback
    for name in ["tetra_avx2.exe", "tetra_avx10.exe", "tetra_avx512.exe", "tetra.exe", "tetra"]:
        exe = Path(__file__).parent / name
        if exe.exists():
            return exe
    return None


def run_inference(model_path, prompt, max_tokens=100, temperature=0.8,
                  top_k=50, top_p=0.9, tokenizer_dir="tokenizer"):
    enc = get_tokenizer_compat(tokenizer_dir)
    tokens = enc.encode(prompt)
    token_str = ",".join(str(t) for t in tokens)

    exe_path = find_exe()
    if exe_path is None:
        print("C++ binary not found. Build first: cd inference && build.bat")
        print("Falling back to Python inference...\n")
        return python_inference(model_path, tokens, max_tokens, temperature, top_k)

    print(f"Prompt: {prompt}\n")

    cmd = [
        str(exe_path), model_path, token_str,
        str(max_tokens), str(temperature), str(top_k), str(top_p),
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    all_ids = []
    prev_text = ""
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Output token IDs:"):
            break
        parts = line.split()
        for part in parts:
            try:
                token_id = int(part)
            except ValueError:
                continue
            if token_id == EOS_TOKEN:
                break
            all_ids.append(token_id)
            text = enc.decode(all_ids)
            new_chunk = text[len(prev_text):]
            prev_text = text
            print(new_chunk, end="", flush=True)
        else:
            continue
        break

    proc.stdout.close()
    proc.wait()

    stderr = proc.stderr.read()
    proc.stderr.close()

    if stderr:
        print(stderr, file=sys.stderr, end="")
    return prev_text


def python_inference(model_path, prompt_tokens, max_tokens, temperature, top_k=50):
    """Fallback: pure Python inference using checkpoint."""
    import torch
    from ternary_llm.transformer import TernaryTransformerModel

    enc = get_tokenizer_compat()
    candidates = sorted(Path("checkpoints").glob("checkpoint_*.pt"))

    model = None
    if candidates:
        c = candidates[-1]
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

    if model is None:
        print("ERROR: No checkpoint found")
        return

    prompt_t = torch.tensor([prompt_tokens])
    with torch.no_grad():
        out = model.generate(prompt_t, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)

    generated = enc.decode(out[0].tolist())
    print(generated)
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
