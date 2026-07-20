"""Streaming data preparation for Tetra.

Streams FineWeb-Edu, Cosmopedia, SlimOrca from HuggingFace,
tokenizes with our custom BPE, writes disk chunks for training.

SlimOrca is cycled (infinite loop) to maintain 20% ratio
against the much larger datasets.

Usage:
    python prepare_data.py                          # Default: 1B tokens total
    python prepare_data.py --target-tokens 5e8      # 500M tokens
    python prepare_data.py --test                   # Quick test: 1M tokens
"""
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from itertools import cycle
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from ternary_llm.data import get_tokenizer_compat


# Config
CHUNK_TOKENS = 25_000_000  # 25M tokens per chunk file

SOURCES = {
    "fineweb": {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "split": "train",
        "field": "text",
        "ratio": 0.50,
    },
    "cosmopedia": {
        "dataset": "HuggingFaceTB/cosmopedia",
        "name": "stories",
        "split": "train",
        "field": "text",
        "ratio": 0.30,
    },
    "orca": {
        "dataset": "Open-Orca/SlimOrca",
        "name": None,
        "split": "train",
        "field": None,  # special formatting
        "ratio": 0.20,
    },
}


def format_orca(item):
    """Format SlimOrca multi-turn conversation into text."""
    parts = []
    conversation = item.get("conversations", item.get("conversation", []))
    for turn in conversation:
        role = turn.get("from", "").lower()
        value = turn.get("value", "")
        if not value:
            continue
        if role == "system":
            parts.append(f"System: {value}")
        elif role == "human":
            parts.append(f"Human: {value}")
        elif role in ("gpt", "assistant"):
            parts.append(f"Assistant: {value}")

    return "\n".join(parts)


def stream_source(source_key, source_cfg, tokenizer, target_tokens, output_dir):
    """Stream one source, tokenize, write chunks."""
    from datasets import load_dataset

    ratio = source_cfg["ratio"]
    source_target = int(target_tokens * ratio)

    print(f"\n{'='*60}")
    print(f"  {source_key.upper()}")
    print(f"  Target: {source_target:,} tokens ({ratio*100:.0f}%)")
    print(f"{'='*60}")

    # Load dataset in streaming mode
    kwargs = {"split": source_cfg["split"], "streaming": True}
    if source_cfg["name"]:
        kwargs["name"] = source_cfg["name"]

    try:
        ds = load_dataset(source_cfg["dataset"], **kwargs)
    except Exception as e:
        print(f"  ERROR loading {source_cfg['dataset']}: {e}")
        print(f"  Skipping this source.")
        return 0, 0

    # Wrap SlimOrca in infinite cycle
    if source_key == "orca":
        ds = _cycling_iter(ds)
        print("  (cycling enabled for oversampling)")

    eos_id = tokenizer.token_to_id("<EOS>")
    buffer = []
    chunk_idx = 0
    total_tokens = 0
    total_docs = 0
    done = False

    pbar = tqdm(desc=f"  {source_key}", unit="tok", total=source_target)

    for item in ds:
        if done:
            break

        # Get text
        if source_cfg["field"] is None:
            text = format_orca(item)
        else:
            text = item.get(source_cfg["field"], "")
            if not text:
                continue

        # Tokenize
        enc = tokenizer.encode(text)
        ids = enc.ids if hasattr(enc, 'ids') else list(enc)
        if not ids:
            continue
        ids.append(eos_id)
        buffer.extend(ids)
        total_docs += 1

        # Flush when buffer >= remaining target or >= CHUNK_TOKENS
        remaining = source_target - total_tokens
        while len(buffer) >= min(CHUNK_TOKENS, max(remaining, 1)) and total_tokens < source_target:
            take = min(CHUNK_TOKENS, len(buffer), max(remaining, len(buffer)))
            if take <= 0:
                break
            chunk_tokens = buffer[:take]
            buffer = buffer[take:]
            chunk_path = output_dir / f"{source_key}_{chunk_idx:04d}.bin"
            np.array(chunk_tokens, dtype=np.uint16).tofile(str(chunk_path))
            chunk_idx += 1
            total_tokens += take
            pbar.update(take)
            remaining = source_target - total_tokens

        if total_tokens >= source_target:
            done = True

    # Write remaining buffer — partial chunk if target reached with leftovers
    if buffer and total_tokens < source_target and len(buffer) > 1000:
        chunk_path = output_dir / f"{source_key}_{chunk_idx:04d}.bin"
        np.array(buffer, dtype=np.uint16).tofile(str(chunk_path))
        total_tokens += len(buffer)
        pbar.update(len(buffer))
        chunk_idx += 1
    elif buffer and total_tokens >= source_target and len(buffer) > 1000:
        # Append leftover to the last chunk or write as extra
        chunk_path = output_dir / f"{source_key}_{chunk_idx:04d}.bin"
        np.array(buffer, dtype=np.uint16).tofile(str(chunk_path))
        total_tokens += len(buffer)
        pbar.update(len(buffer))
        chunk_idx += 1

    pbar.close()
    print(f"  Done: {total_tokens:,} tokens in {chunk_idx} chunks")
    return total_tokens, total_docs


def _cycling_iter(ds):
    """Infinite cycling iterator for a dataset."""
    while True:
        for item in ds:
            yield item


def main():
    parser = argparse.ArgumentParser(description="Prepare multi-source training data")
    parser.add_argument("--target-tokens", type=float, default=1e9,
                        help="Total tokens to prepare (default: 1B)")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for chunks")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer")
    parser.add_argument("--test", action="store_true",
                        help="Quick test: prepare only 1M tokens")
    parser.add_argument("--sources", nargs="+", default=["fineweb", "cosmopedia", "orca"],
                        help="Which sources to prepare")
    args = parser.parse_args()

    if args.test:
        args.target_tokens = 1_000_000

    target = int(args.target_tokens)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Tetra Multi-Source Data Preparation")
    print("=" * 60)
    print(f"  Target: {target:,} tokens")
    print(f"  Chunk size: {CHUNK_TOKENS:,} tokens")
    print(f"  Output: {output_dir}/")
    print(f"  Sources: {args.sources}")

    # Load tokenizer
    tokenizer = get_tokenizer_compat(args.tokenizer_dir)
    eos_id = tokenizer.token_to_id("<EOS>")
    print(f"  Tokenizer vocab: {tokenizer.n_vocab}")
    print(f"  EOS token ID: {eos_id}")

    # Prepare each source
    manifest = {"sources": {}, "total_tokens": 0, "vocab_size": tokenizer.n_vocab}

    for source_key in args.sources:
        if source_key not in SOURCES:
            print(f"  Unknown source: {source_key}, skipping")
            continue

        source_cfg = SOURCES[source_key]
        n_tokens, n_docs = stream_source(
            source_key, source_cfg, tokenizer, target, output_dir,
        )
        manifest["sources"][source_key] = {
            "n_tokens": n_tokens,
            "n_documents": n_docs,
            "ratio": source_cfg["ratio"],
        }
        manifest["total_tokens"] += n_tokens

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  COMPLETE")
    print(f"{'='*60}")
    print(f"  Total tokens: {manifest['total_tokens']:,}")
    for src, info in manifest["sources"].items():
        print(f"    {src}: {info['n_tokens']:,} tokens, {info['n_documents']:,} docs")
    print(f"  Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
