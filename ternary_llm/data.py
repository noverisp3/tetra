"""Data pipeline for Tetra -- tokenizer, dataset, dataloaders.

Uses a custom BPE tokenizer trained on TinyStories (not GPT-2/tiktoken).
Tokenizer is stored in tokenizer/tetra_tokenizer.json.
"""
__all__ = [
    "get_tokenizer", "TokenizerWrapper", "get_tokenizer_compat",
    "download_and_tokenize", "ChunkedDataset", "create_dataloaders",
    "MultiSourceChunkedDataset", "create_multi_source_dataloaders",
]

import json
import bisect
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from tqdm import tqdm


# Tokenizer
_tokenizer_cache = None


def get_tokenizer(tokenizer_dir="tokenizer"):
    """Load tokenizer: custom BPE (tokenizers) by default, GPT-2 if tokenizer_dir='gpt2'."""
    global _tokenizer_cache
    if _tokenizer_cache is not None:
        return _tokenizer_cache

    # Custom BPE (default) - simpler, smaller vocab, no transformers dependency
    from tokenizers import Tokenizer
    tok_path = Path(tokenizer_dir) / "tetra_tokenizer.json"
    if tok_path.exists():
        _tokenizer_cache = Tokenizer.from_file(str(tok_path))
        return _tokenizer_cache

    # GPT-2 (transformers) - only if explicitly named "gpt2"
    if tokenizer_dir == "gpt2":
        try:
            from transformers import GPT2Tokenizer
            tok = GPT2Tokenizer.from_pretrained("gpt2")
            _tokenizer_cache = tok
            return tok
        except ImportError:
            print("Warning: transformers not installed, falling through to custom tokenizer")

    raise FileNotFoundError(
        f"Tokenizer not found at {tok_path}.\n"
        f"Run: python train_tokenizer.py or use --tokenizer-dir gpt2"
    )


class TokenizerWrapper:
    """Compatibility wrapper for tokenizers (both HuggingFace tokenizers & transformers).

    Provides unified interface:
        enc.encode(text) -> list[int]
        enc.decode(ids) -> str
        enc.eot_token -> int (eos_id)
        enc.n_vocab -> int
    """
    def __init__(self, tokenizer):
        self._tok = tokenizer
        # Detect tokenizer type
        if hasattr(tokenizer, "eos_token_id"):
            # transformers.PreTrainedTokenizer
            self._is_transformers = True
            self.eot_token = tokenizer.eos_token_id
            self.n_vocab = tokenizer.vocab_size
        else:
            # tokenizers.Tokenizer
            self._is_transformers = False
            vocab = tokenizer.get_vocab()
            self.eot_token = vocab.get("<EOS>", vocab.get("<eos>", 2))
            self.n_vocab = tokenizer.get_vocab_size()

    def encode(self, text):
        if self._is_transformers:
            return self._tok.encode(text)
        ids = self._tok.encode(text).ids
        if ids and ids[0] == self._tok.token_to_id("<BOS>"):
            ids = ids[1:]
        if ids and ids[-1] == self._tok.token_to_id("<EOS>"):
            ids = ids[:-1]
        return ids

    def encode_ordinary(self, text):
        return self.encode(text)

    def decode(self, ids):
        return self._tok.decode(ids)

    def token_to_id(self, token):
        if self._is_transformers:
            return self._tok.convert_tokens_to_ids(token)
        return self._tok.token_to_id(token)


def get_tokenizer_compat(tokenizer_dir="tokenizer"):
    """Return a TokenizerWrapper for unified tokenizer API."""
    raw = get_tokenizer(tokenizer_dir)
    return TokenizerWrapper(raw)


# Data Download & Tokenize

def download_and_tokenize(cache_dir="data", tokenizer_dir="tokenizer", max_stories=None):
    """Download TinyStories dataset, tokenize, and cache as .bin + metadata.json.

    Args:
        cache_dir: Directory for data storage (default: data/)
        tokenizer_dir: Directory containing tetra_tokenizer.json (default: tokenizer/)
        max_stories: Limit number of stories to process (default: all)

    Returns:
        tokens: numpy memmap of uint16 token ids
        metadata: dict with vocab_size, total_tokens
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Use custom tokenizer name so it doesn't collide with old gpt2 data
    tok = get_tokenizer(tokenizer_dir)
    tok_wrap = TokenizerWrapper(tok)

    bin_path = cache_path / "tinystories.bin"
    meta_path = cache_path / "metadata.json"

    if bin_path.exists() and meta_path.exists():
        print(f"Loading cached data from {bin_path}")
        tokens = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        with open(meta_path) as f:
            metadata = json.load(f)
        print(f"Tokens: {len(tokens):,} | Vocab: {metadata['vocab_size']}")
        return tokens, metadata

    # Find TinyStories text file (prefer V2 GPT-4)
    candidates = [
        "TinyStoriesV2-GPT4-train.txt",
        "TinyStories-train.txt",
    ]
    txt_path = None
    for name in candidates:
        p = cache_path / name
        if p.exists() and p.stat().st_size > 1e6:
            txt_path = p
            break

    if txt_path is None:
        print("Downloading TinyStoriesV2-GPT4-train.txt...")
        import requests
        url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
        r = requests.get(url, stream=True, timeout=(5, 30))
        r.raise_for_status()
        expected_size = int(r.headers.get("Content-Length", 0))
        txt_path = cache_path / "TinyStoriesV2-GPT4-train.txt"
        downloaded = 0
        with open(txt_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
        actual_mb = txt_path.stat().st_size / 1e6
        if expected_size and abs(downloaded - expected_size) > 1024:
            print(f"Warning: downloaded {downloaded} bytes, expected {expected_size}")
        print(f"Downloaded to {txt_path} ({actual_mb:.0f} MB)")

    print(f"Reading {txt_path} ({txt_path.stat().st_size / 1e6:.0f} MB)...")
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Splitting into stories...")
    stories = text.split("\n\n\n")
    stories = [s.strip() for s in stories if s.strip()]
    print(f"Found {len(stories):,} stories")

    if max_stories:
        stories = stories[:max_stories]
        print(f"Using first {max_stories:,} stories")

    eos_id = tok_wrap.eot_token

    print("Tokenizing with custom BPE tokenizer...")
    all_tokens = []
    for story in tqdm(stories, desc="Tokenize", unit="story"):
        token_ids = tok_wrap.encode(story)
        all_tokens.extend(token_ids)
        all_tokens.append(eos_id)

    tokens_array = np.array(all_tokens, dtype=np.uint16)
    total = len(tokens_array)
    print(f"Total: {len(stories):,} stories, {total:,} tokens")

    tokens_array.tofile(str(bin_path))
    mb = bin_path.stat().st_size / 1e6
    print(f"Saved to {bin_path} ({mb:.1f} MB)")

    metadata = {"vocab_size": tok_wrap.n_vocab, "total_tokens": total}
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return tokens_array, metadata


# Dataset

class ChunkedDataset(Dataset):
    """Non-overlapping chunks of tokens for fast shuffling."""
    def __init__(self, tokens, block_size):
        n = len(tokens)
        # Reserve 1 for y offset: max valid start_idx = n - block_size - 1
        valid_starts = n - block_size - 1  # extra -1 so y[start+block_size] is valid
        if valid_starts < 0:
            raise ValueError(f"Too few tokens ({n}) for block_size={block_size}")
        n_samples = valid_starts // block_size  # non-overlapping blocks
        n_usable = n_samples * block_size + 1  # +1 for y offset
        self.tokens = torch.from_numpy(tokens[:n_usable].astype(np.int64))
        self.block_size = block_size
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = self.tokens[start : start + self.block_size]
        y = self.tokens[start + 1 : start + self.block_size + 1]
        return x, y


def create_dataloaders(tokens, block_size=128, batch_size=8, val_split=0.05, num_workers=0, pin_memory=False):
    """Create train/validation DataLoaders from token array.

    Args:
        tokens: uint16 array of token ids
        block_size: context window per sample
        batch_size: samples per batch
        val_split: fraction of tokens for validation
        num_workers: DataLoader worker count
        pin_memory: pin memory for GPU transfer

    Returns:
        train_loader, val_loader
    """
    split_idx = int(len(tokens) * (1 - val_split))
    train_ds = ChunkedDataset(tokens[:split_idx], block_size)
    val_ds = ChunkedDataset(tokens[split_idx:], block_size)
    print(f"Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return train_loader, val_loader


# Multi-Source Dataset

class MultiSourceChunkedDataset(Dataset):
    """Multi-source chunked dataset with ratio-based sampling."""
    """Reads multiple .bin chunk files from different sources,
    samples according to configured ratios.

    Directory structure:
        data/
            manifest.json       # source metadata + ratios
            fineweb_0000.bin    # uint16 token chunks
            cosmopedia_0000.bin
            orca_0000.bin
            ...
    """
    def __init__(self, data_dir, block_size, val_split=0.05):
        data_dir = Path(data_dir)
        with open(data_dir / "manifest.json") as f:
            self.manifest = json.load(f)

        self.block_size = block_size
        self.sources = {}  # name -> {"chunks": [path, ...], "ratio": float, "offsets": [int]}
        self.total_chunks = 0
        self.rng = np.random.default_rng(42)

        # Load chunk files per source
        for src_name, src_info in self.manifest["sources"].items():
            chunks = sorted(data_dir.glob(f"{src_name}_*.bin"))
            if not chunks:
                print(f"WARNING: No chunks found for source '{src_name}'")
                continue

            # Compute token count per chunk
            chunk_entries = []
            for chunk_path in chunks:
                n_bytes = chunk_path.stat().st_size
                n_tokens = n_bytes // 2  # uint16 = 2 bytes
                n_blocks = n_tokens // block_size
                if n_blocks > 0:
                    chunk_entries.append({"path": chunk_path, "n_blocks": n_blocks})

            if not chunk_entries:
                continue

            self.sources[src_name] = {
                "chunks": chunk_entries,
                "ratio": src_info["ratio"],
                "cum_blocks": [],  # cumulative block count for index mapping
            }
            cum = 0
            for ce in chunk_entries:
                self.sources[src_name]["cum_blocks"].append(cum)
                cum += ce["n_blocks"]
            self.sources[src_name]["total_blocks"] = cum
            self.total_chunks += cum

        if not self.sources:
            raise RuntimeError(f"No valid chunks found in {data_dir}")

        print(f"MultiSourceChunkedDataset:")
        for name, src in self.sources.items():
            print(f"{name}: {src['total_blocks']:,} blocks ({len(src['chunks'])} chunks, ratio={src['ratio']:.0%})")
        print(f"Total: {self.total_chunks:,} blocks")

        # Split into train/val
        self.val_start = int(self.total_chunks * (1 - val_split))
        self.n_samples = self.val_start  # train portion

        # Cache np.memmap objects to avoid reopening files per __getitem__
        self._memmap_cache = {}

    def __len__(self):
        return self.n_samples

    def _get_source_for_index(self, global_idx):
        """Pick a source by ratio, then map global index to local block."""
        # Weighted random source selection (deterministic per sample for reproducibility)
        r = self.rng.random()
        cum = 0.0
        chosen = None
        for name, src in self.sources.items():
            cum += src["ratio"]
            if r < cum:
                chosen = name
                break
        if chosen is None:
            chosen = list(self.sources.keys())[-1]

        # Map to a block within this source (round-robin for coverage)
        src = self.sources[chosen]
        local_idx = global_idx % src["total_blocks"]

        # Find which chunk file this falls into via binary search
        chunk_entries = src["chunks"]
        cum_blocks = src["cum_blocks"]
        ci = bisect.bisect_right(cum_blocks, local_idx) - 1
        offset_in_chunk = (local_idx - cum_blocks[ci]) * self.block_size
        return chunk_entries[ci]["path"], offset_in_chunk

    def __getitem__(self, idx):
        if idx >= self.val_start:
            idx = idx + (self.total_chunks - self.val_start)

        chunk_path, offset = self._get_source_for_index(idx)
        path_str = str(chunk_path)
        if path_str not in self._memmap_cache:
            self._memmap_cache[path_str] = np.memmap(path_str, dtype=np.uint16, mode="r")
        tokens = self._memmap_cache[path_str]
        x = torch.from_numpy(tokens[offset:offset + self.block_size].astype(np.int64))
        y = torch.from_numpy(tokens[offset + 1:offset + self.block_size + 1].astype(np.int64))
        return x, y


def create_multi_source_dataloaders(
    data_dir, block_size=128, batch_size=8, val_split=0.05,
    num_workers=0, pin_memory=False,
):
    """Create train/val dataloaders from multi-source chunks (manifest.json format).

    Args:
        data_dir: directory with manifest.json and source_*.bin files
        block_size: context window per sample
        batch_size: samples per batch
        val_split: fraction of blocks for validation
        num_workers: DataLoader worker count
        pin_memory: pin memory for GPU transfer

    Returns:
        train_loader, val_loader
    """
    """Create train/val dataloaders from multi-source chunks."""
    ds = MultiSourceChunkedDataset(data_dir, block_size, val_split=val_split)
    n_val = ds.total_chunks - ds.val_start

    train_ds = ds
    val_ds = MultiSourceChunkedDataset(data_dir, block_size, val_split=val_split)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    print(f"Train samples: {len(train_ds):,} | Val samples: {n_val:,}")
    return train_loader, val_loader
