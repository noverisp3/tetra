"""Tokenize local FineWeb-Edu parquet files into uint16 chunk files for training.

Usage:
    python scripts/prepare_parquet.py
    python scripts/prepare_parquet.py --parquet-dir data --output-dir data/fineweb_10bt --chunk-tokens 25000000
"""
import sys, json, argparse, numpy as np, pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ternary_llm.data import get_tokenizer_compat

def main():
    parser = argparse.ArgumentParser(description="Tokenize local FineWeb-Edu parquet files")
    parser.add_argument("--parquet-dir", type=str, default="data",
                        help="Directory containing *.parquet files (default: data)")
    parser.add_argument("--output-dir", type=str, default="data/fineweb_10bt",
                        help="Output directory for tokenized chunks (default: data/fineweb_10bt)")
    parser.add_argument("--chunk-tokens", type=int, default=25_000_000,
                        help="Tokens per chunk file (default: 25M)")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer")
    parser.add_argument("--source-name", type=str, default="fineweb",
                        help="Source name in manifest (default: fineweb)")
    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No .parquet files found in {parquet_dir}")
        sys.exit(1)

    print(f"Tetra Parquet Tokenizer")
    print(f"Parquet files: {len(parquet_files)} ({sum(f.stat().st_size for f in parquet_files) / 1e9:.1f} GB)")
    print(f"Output: {output_dir}/")
    print(f"Chunk size: {args.chunk_tokens:,} tokens")

    tokenizer = get_tokenizer_compat(args.tokenizer_dir)
    eos_id = tokenizer.token_to_id("<EOS>")
    bos_id = tokenizer.token_to_id("<BOS>")
    if eos_id is None:
        eos_id = 0
    print(f"Tokenizer vocab: {tokenizer.n_vocab}, EOS: {eos_id}")

    batch_size = 2000
    total_tokens = 0
    total_docs = 0
    chunk_idx = 0
    buffer = []

    CHUNK_TOKENS = args.chunk_tokens

    for pf_path in parquet_files:
        pf = pq.ParquetFile(str(pf_path))
        n_rows = pf.metadata.num_rows
        print(f"\n{pf_path.name}: {n_rows:,} rows")

        total_batches = (n_rows + batch_size - 1) // batch_size
        for batch in tqdm(pf.iter_batches(batch_size=batch_size, columns=["text"]),
                         desc=f"  {pf_path.name}", total=total_batches):
            texts_batch = [t.as_py() for t in batch.column(0) if t.as_py()]
            if not texts_batch:
                continue

            for text in texts_batch:
                enc = tokenizer.encode(text)
                if not enc:
                    continue
                enc.append(eos_id)
                buffer.extend(enc)
                total_docs += 1

            while len(buffer) >= CHUNK_TOKENS:
                chunk = buffer[:CHUNK_TOKENS]
                buffer = buffer[CHUNK_TOKENS:]
                chunk_path = output_dir / f"{args.source_name}_{chunk_idx:04d}.bin"
                np.array(chunk, dtype=np.uint16).tofile(str(chunk_path))
                chunk_idx += 1
                total_tokens += CHUNK_TOKENS

    # Flush remaining buffer
    if len(buffer) > 1000:
        chunk_path = output_dir / f"{args.source_name}_{chunk_idx:04d}.bin"
        np.array(buffer, dtype=np.uint16).tofile(str(chunk_path))
        total_tokens += len(buffer)
        chunk_idx += 1

    # Write manifest
    manifest = {
        "sources": {
            args.source_name: {
                "n_tokens": total_tokens,
                "n_documents": total_docs,
                "ratio": 1.0,
            }
        },
        "total_tokens": total_tokens,
        "vocab_size": tokenizer.n_vocab,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nComplete!")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total documents: {total_docs:,}")
    print(f"Chunks: {chunk_idx}")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
