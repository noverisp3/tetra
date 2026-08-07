"""Generate conversation training data for Tetra from an LLM teacher (GGUF via llama-cpp-python).

Drives multi-turn dialogues: the script plays the user with templated openers/
follow-ups, the teacher plays the assistant. Output is written as

    Human: {opener}
    Assistant: {reply}
    ...

in uint16 token chunks + manifest.json (Tetra multi-source format), so the
output can be mixed with existing data via --data-cache.

Usage:
    python scripts/generate_teacher_data.py --model Qwen3.5-9B-Q4_K_M.gguf
    python scripts/generate_teacher_data.py --model Qwen3.5-9B-Q4_K_M.gguf --target-tokens 2e6 --gpu-layers -1
    python scripts/generate_teacher_data.py --model Qwen3.5-9B-Q4_K_M.gguf --tag teacher2 --seed 7 --output-dir data_teacher

Run multiple instances with different --tag/--seed against the same output-dir
to parallelize generation; manifests are merged.
"""
import sys
import re
import json
import time
import argparse
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ternary_llm.data import get_tokenizer_compat

CHUNK_TOKENS = 5_000_000

OPENERS = [
    "Hello! What is your name?",
    "Do you like animals? Which one do you like the most?",
    "What is your favorite food?",
    "Can you tell me about your day?",
    "What color do you like and why?",
    "Do you like to sing? Can you sing me a song?",
    "What is your favorite game to play?",
    "Tell me a funny joke!",
    "What do you do in the morning?",
    "Do you have any friends? What are they like?",
    "What is your favorite place to visit?",
    "Do you like the rain or the sun?",
    "What would you do if you found a magic hat?",
    "Can you teach me something new today?",
    "What is your favorite thing about school?",
    "Do you like to draw? What do you like to draw?",
    "What is your favorite season and why?",
    "If you could fly, where would you go?",
    "Do you like robots? What do they do?",
    "What makes you happy?",
    "What is your favorite bedtime story?",
    "Do you like helping other people? How?",
    "What is the best thing that happened to you today?",
    "If you had a pet dragon, what would you do?",
]

FOLLOW_UPS = [
    "That is interesting! Tell me more.",
    "Why do you like that?",
    "What happened next?",
    "Can you explain that to me?",
    "That sounds fun. What else can you do?",
    "Do you have a pet?",
    "What about your favorite place?",
    "How do you do that?",
    "That is funny! Can you tell me another one?",
    "What do you think about that?",
]

SYSTEM_PROMPT = (
    "You are a friendly, simple chatbot for young children. "
    "Talk in short, simple English sentences. Be kind, clear, and cheerful. "
    "Keep every answer to 1 to 3 short sentences. Never ask the child a question."
)


def strip_think_blocks(text):
    """Remove Qwen3.5 thinking blocks (<think>...</think>)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def is_clean_english(text):
    """Reject text containing non-ASCII content (Qwen tends to mix in CJK)."""
    if not text:
        return False
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) >= 0.99


def run_turn(llm, messages, args):
    """Send messages, return assistant reply text or None on failure."""
    try:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            repeat_penalty=args.repeat_penalty,
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  WARNING: turn failed ({e})")
        return None


def clean_reply(text):
    """Clean assistant reply: strip thinking blocks, collapse newlines."""
    text = strip_think_blocks(text)
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def build_conversation(llm, opener, args):
    """Play one multi-turn dialogue with the teacher. Returns (text, n_turns) or (None, 0)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    lines = []
    turns = 0

    turn_text = opener
    for t in range(args.max_turns):
        messages.append({"role": "user", "content": turn_text})
        reply = run_turn(llm, messages, args)
        if not reply or not reply.strip():
            return None, 0
        reply = clean_reply(reply)
        if not reply:
            return None, 0
        lines.append(f"Human: {turn_text}")
        lines.append(f"Assistant: {reply}")
        turns += 1
        messages.append({"role": "assistant", "content": reply.strip()})
        if t < args.max_turns - 1:
            turn_text = FOLLOW_UPS[t % len(FOLLOW_UPS)]

    return "\n".join(lines), turns


def tokenize_text(tokenizer, text):
    ids = tokenizer.encode(text.strip())
    if not ids:
        return []
    ids.append(tokenizer.eot_token)
    return ids


def load_existing_tokens(output_dir, tag):
    chunks = sorted(output_dir.glob(f"{tag}_*.bin"))
    total = sum(c.stat().st_size // 2 for c in chunks)
    return chunks, total


def write_chunk(buffer, output_dir, tag, chunk_idx):
    import numpy as np
    path = output_dir / f"{tag}_{chunk_idx:04d}.bin"
    np.array(buffer, dtype=np.uint16).tofile(str(path))
    return path


def update_manifest(output_dir, tag, n_tokens, n_convs, vocab_size, ratio):
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {"sources": {}, "total_tokens": 0, "vocab_size": vocab_size}

    manifest["vocab_size"] = vocab_size
    if tag in manifest["sources"]:
        prev = manifest["sources"][tag]
        n_tokens += prev["n_tokens"]
        n_convs += prev["n_documents"]
    manifest["sources"][tag] = {
        "n_tokens": n_tokens,
        "n_documents": n_convs,
        "ratio": ratio,
    }

    others = [k for k in manifest["sources"] if k != tag]
    remain = max(0.0, 1.0 - ratio)
    for k in others:
        manifest["sources"][k]["ratio"] = remain / len(others) if others else 0.0
    manifest["total_tokens"] = sum(
        info["n_tokens"] for info in manifest["sources"].values()
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Generate teacher conversation data for Tetra")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to teacher GGUF file (e.g. Qwen3.5-9B-Q4_K_M.gguf)")
    parser.add_argument("--output-dir", type=str, default="data_teacher",
                        help="Output directory (default: data_teacher)")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer")
    parser.add_argument("--tag", type=str, default="teacher",
                        help="Chunk file prefix + manifest source name (default: teacher)")
    parser.add_argument("--target-tokens", type=float, default=5_000_000,
                        help="Target tokens to generate (default: 5M)")
    parser.add_argument("--gpu-layers", type=int, default=0,
                        help="GPU layers for llama.cpp: 0 = CPU only, -1 = all (default: 0)")
    parser.add_argument("--threads", type=int, default=0,
                        help="CPU threads (default: auto)")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context size (default: 2048)")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max tokens per assistant reply (default: 128)")
    parser.add_argument("--max-turns", type=int, default=6,
                        help="Turns per conversation (default: 6)")
    parser.add_argument("--min-words", type=int, default=30,
                        help="Minimum word count per conversation (default: 30)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--repeat-penalty", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratio", type=float, default=0.30,
                        help="Manifest ratio when mixed with existing sources (default: 0.30)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        sys.exit(1)

    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama-cpp-python not installed. Run: pip install llama-cpp-python")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Tetra Teacher Conversation Data Generation")
    print(f"Model: {model_path.name} ({model_path.stat().st_size / 1e9:.2f} GB)")
    print(f"Target: {int(args.target_tokens):,} tokens")
    print(f"Output: {output_dir}/ (tag={args.tag})")

    tokenizer = get_tokenizer_compat(args.tokenizer_dir)
    print(f"Tokenizer vocab: {tokenizer.n_vocab}")

    chunks, existing = load_existing_tokens(output_dir, args.tag)
    print(f"Existing chunks: {len(chunks)} ({existing:,} tokens) — resuming")

    llm = Llama(
        model_path=str(model_path),
        n_ctx=args.n_ctx,
        n_gpu_layers=args.gpu_layers,
        n_threads=args.threads if args.threads > 0 else None,
        seed=args.seed,
        verbose=False,
    )

    target = int(args.target_tokens)
    remaining = target - existing
    if remaining <= 0:
        print("Target already reached. Nothing to do.")
        return

    from tqdm import tqdm
    pbar = tqdm(total=remaining, desc=f"  {args.tag}", unit="tok")

    buffer = []
    chunk_idx = len(chunks)
    total_tokens = existing
    n_convs = 0
    seen = set()
    conv_index = 0
    start_time = time.time()

    while total_tokens < target:
        opener = OPENERS[conv_index % len(OPENERS)]
        conv_index += 1

        text, turns = build_conversation(llm, opener, args)
        if text is None or turns == 0:
            continue

        text = clean_reply(text)
        if not is_clean_english(text):
            continue
        if len(text.split()) < args.min_words:
            continue
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)

        ids = tokenize_text(tokenizer, text)
        if not ids:
            continue

        buffer.extend(ids)
        n_convs += 1
        total_tokens += len(ids)
        pbar.update(len(ids))

        while len(buffer) >= CHUNK_TOKENS:
            write_chunk(buffer[:CHUNK_TOKENS], output_dir, args.tag, chunk_idx)
            buffer = buffer[CHUNK_TOKENS:]
            chunk_idx += 1

    if buffer:
        write_chunk(buffer, output_dir, args.tag, chunk_idx)
        chunk_idx += 1

    pbar.close()

    elapsed = time.time() - start_time
    generated = total_tokens - existing
    rate = generated / elapsed if elapsed > 0 else 0.0

    manifest_path = update_manifest(
        output_dir, args.tag, generated, n_convs,
        tokenizer.n_vocab, args.ratio,
    )
    print(f"Done: {n_convs:,} conversations, {generated:,} tokens, {chunk_idx} chunks")
    print(f"Throughput: {rate:.1f} tok/s ({elapsed / 60:.1f} min)")
    print(f"Manifest: {manifest_path}")
    print(f"Train with: python train.py --data-cache {output_dir}")


if __name__ == "__main__":
    main()
