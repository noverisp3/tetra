"""Train a custom BPE tokenizer on TinyStories or FineWeb Edu parquet.

Usage:
    python train_tokenizer.py                          # vocab=8192, trains on TinyStories
    python train_tokenizer.py --vocab-size 16384       # larger vocab
    python train_tokenizer.py --max-stories 50000      # quick test run
    python train_tokenizer.py --data-path data --from-parquet  # train on FineWeb Edu parquet

Output:
    tokenizer/tetra_tokenizer.json  — HuggingFace tokenizers format
    tokenizer/vocab_info.json       — vocab_size and metadata

Training uses a random 500MB sample of text by default (enough for BPE).
"""
import argparse
import random
import json
from pathlib import Path


def find_tinystories(cache_dir: str = "data") -> str:
    """Find TinyStories V2 GPT-4 train file in data directory."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Prefer V2 GPT-4 version (higher quality)
    candidates = [
        "TinyStoriesV2-GPT4-train.txt",
        "TinyStories-train.txt",
    ]
    for name in candidates:
        p = cache_path / name
        if p.exists() and p.stat().st_size > 1e6:
            print(f"Using {p.name} ({p.stat().st_size / 1e6:.0f} MB)")
            return str(p)

    print("ERROR: No TinyStories data found!")
    print(f"Please download TinyStoriesV2-GPT4-train.txt to {cache_path}/")
    print("https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStoriesV2-GPT4-train.txt")
    exit(1)


def _extract_parquet_sample(parquet_dir: str, output_dir: str, target_mb: int = 500) -> str:
    """Sample text from FineWeb Edu parquet files and write to a temp .txt for tokenizer training."""
    import pyarrow.parquet as pq
    target_bytes = target_mb * 1024 * 1024
    out_path = Path(output_dir) / "tokenizer_train_sample.txt"
    if out_path.exists():
        print(f"Using existing sample: {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")
        return str(out_path)

    parquet_files = sorted(Path(parquet_dir).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files in {parquet_dir}")

    print(f"Extracting ~{target_mb}MB text sample from {len(parquet_files)} parquet files...")
    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for pf_path in parquet_files:
            pf = pq.ParquetFile(str(pf_path))
            for batch in pf.iter_batches(batch_size=2000, columns=["text"]):
                for text in batch.column(0):
                    t = text.as_py()
                    if t:
                        line = t.replace("\n", " ") + "\n"
                        out.write(line)
                        written += len(line.encode("utf-8"))
                        if written >= target_bytes:
                            print(f"  Reached {target_mb}MB, stopping.")
                            return str(out_path)
    print(f"  Extracted {written / 1e6:.0f} MB to {out_path}")
    return str(out_path)


def train_tokenizer(
    vocab_size: int = 8192,
    data_path: str | None = None,
    output_dir: str = "tokenizer",
    max_stories: int | None = None,
    from_parquet: bool = False,
    sample_mb: int = 500,
):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get training text data
    train_file = None
    if data_path is not None and from_parquet:
        train_file = _extract_parquet_sample(data_path, output_dir, sample_mb)
    elif data_path is None:
        train_file = find_tinystories()
    else:
        train_file = data_path

    # Create tokenizer with special tokens
    special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.enable_padding(
        pad_id=0,
        pad_token="<PAD>",
        pad_to_multiple_of=None,
    )

    # Train BPE
    print(f"\nTraining BPE tokenizer (vocab_size={vocab_size})...")
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
        continuing_subword_prefix="",
    )

    if max_stories:
        with open(train_file, "r", encoding="utf-8") as f:
            text = f.read()
        stories = text.split("\n\n\n")
        stories = [s.strip() for s in stories if s.strip()][:max_stories]
        temp_path = Path(output_dir) / "train_subset.txt"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write("\n\n\n".join(stories))
        tokenizer.train(files=[str(temp_path)], trainer=trainer)
        temp_path.unlink()
    else:
        tokenizer.train(files=[train_file], trainer=trainer)

    # Add BOS/EOS after training
    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    tokenizer.post_processor = TemplateProcessing(
        single=f"<BOS>:0 $A:0 <EOS>:0",
        special_tokens=[("<BOS>", bos_id), ("<EOS>", eos_id)],
    )

    # Save
    out_path = Path(output_dir) / "tetra_tokenizer.json"
    tokenizer.save(str(out_path))
    print(f"Tokenizer saved to {out_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # Save metadata
    import json
    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
        },
        "backend": "huggingface_tokenizers",
        "type": "BPE",
    }
    meta_path = Path(output_dir) / "vocab_info.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    # Quick test
    test_texts = [
        "Once upon a time, there was a little girl.",
        "The bear was very happy and jumped around.",
        "And they lived happily ever after.",
    ]
    print("\nTest encodings:")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        print(f"'{text[:50]}...'")
        print(f"tokens: {encoded.ids}")
        print(f"decoded: '{decoded[:50]}...'")
        print()

    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer for Tetra")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Vocab size (default: 8192)")
    parser.add_argument("--data-path", type=str, default=None, help="Path to text file or parquet dir")
    parser.add_argument("--output-dir", type=str, default="tokenizer", help="Output directory")
    parser.add_argument("--max-stories", type=int, default=None, help="Max stories/rows for quick train")
    parser.add_argument("--from-parquet", action="store_true", help="data-path is a FineWeb parquet dir")
    parser.add_argument("--sample-mb", type=int, default=500, help="Text sample in MB (parquet mode)")
    args = parser.parse_args()

    train_tokenizer(
        vocab_size=args.vocab_size,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_stories=args.max_stories,
        from_parquet=args.from_parquet,
        sample_mb=args.sample_mb,
    )


if __name__ == "__main__":
    main()
