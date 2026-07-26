"""Tetra Inference Runner.

Provides both C++ binary and Python fallback inference.
"""
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from ternary_llm.data import get_tokenizer_compat


EOS_TOKEN = 2


def find_exe() -> Optional[Path]:
    """Find the best available C++ binary (prefers optimized builds)."""
    # Try optimized builds first, then scalar fallback
    for name in ["tetra_avx2.exe", "tetra_avx10.exe", "tetra_avx512.exe", "tetra.exe", "tetra"]:
        exe = Path(__file__).parent / name
        if exe.exists():
            return exe
    return None


def run_inference(
    model_path: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    tokenizer_dir: str = "tokenizer",
    repeat_penalty: float = 1.0,
) -> str:
    """Run inference using C++ binary or Python fallback.

    Args:
        model_path: path to tetra_model.bin
        prompt: input text prompt
        max_tokens: maximum tokens to generate
        temperature: sampling temperature
        top_k: top-k sampling
        top_p: nucleus sampling threshold
        tokenizer_dir: tokenizer directory
        repeat_penalty: repetition penalty (>1.0 penalizes repeats)

    Returns:
        Generated text
    """
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
        str(repeat_penalty),
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

    print()
    if stderr:
        print(stderr.rstrip())
    return prev_text


def python_inference(
    model_path: str,
    prompt_tokens: list[int],
    max_tokens: int,
    temperature: float,
    top_k: int = 50,
) -> str:
    """Fallback: pure Python inference using checkpoint.

    Args:
        model_path: path to model binary (unused, uses latest checkpoint)
        prompt_tokens: list of input token ids
        max_tokens: maximum tokens to generate
        temperature: sampling temperature
        top_k: top-k sampling

    Returns:
        Generated text
    """
    import torch
    from ternary_llm.transformer import (
        TernaryTransformerModel, StochasticTransformerModel, StochasticMLAModel,
    )

    enc = get_tokenizer_compat()
    candidates = sorted(Path("checkpoints").glob("checkpoint_*.pt"))

    model = None
    if candidates:
        c = candidates[-1]
        ckpt = torch.load(c, map_location="cpu", weights_only=False)
        config = ckpt["config"]
        sd = ckpt["model_state_dict"]
        mode = config.get("mode", "ste")
        mla = config.get("mla", False) or any("kv_down_proj" in k for k in sd)

        if mode == "stochastic" and mla:
            kv_latent_dim = config.get("kv_latent_dim", None)
            rope_per_head = config.get("rope_per_head", None)
            hidden_dim = config.get("hidden_dim", 256)
            num_heads = config.get("num_heads", 4)
            for k, v in sd.items():
                if k.endswith("kv_down_proj.packed_weights") and kv_latent_dim is None:
                    kv_latent_dim = v.numel() * 4 // hidden_dim
                if k.endswith("q_rope_proj.packed_weights") and rope_per_head is None:
                    rope_dim = v.numel() * 4 // hidden_dim
                    rope_per_head = rope_dim // num_heads
            model = StochasticMLAModel(
                vocab_size=config["vocab_size"], hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"], num_heads=config["num_heads"],
                ffn_dim=config["ffn_dim"], max_seq_len=config["max_seq_len"],
                scale=config.get("ternary_scale", 1.0),
                threshold=config.get("threshold", None),
                int8=config.get("int8", False),
                topk=config.get("topk", 1.0),
                group_size=config.get("group_size", 0),
                kv_latent_dim=kv_latent_dim,
                rope_per_head=rope_per_head,
            )
        elif mode == "stochastic":
            model = StochasticTransformerModel(
                vocab_size=config["vocab_size"], hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"], num_heads=config["num_heads"],
                ffn_dim=config["ffn_dim"], max_seq_len=config["max_seq_len"],
                scale=config.get("ternary_scale", 1.0),
                threshold=config.get("threshold", None),
                int8=config.get("int8", False),
                topk=config.get("topk", 1.0),
                group_size=config.get("group_size", 0),
            )
        else:
            model = TernaryTransformerModel(
                vocab_size=config["vocab_size"], hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"], num_heads=config["num_heads"],
                ffn_dim=config["ffn_dim"], max_seq_len=config["max_seq_len"],
            )
        model.load_state_dict(sd)
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
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer")
    args = parser.parse_args()

    run_inference(args.model, args.prompt, args.max_tokens, args.temp,
                  args.top_k, args.top_p, args.tokenizer_dir, args.repeat_penalty)


if __name__ == "__main__":
    main()
