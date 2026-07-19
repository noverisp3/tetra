"""Data pipeline for Tetra -- tokenizer, dataset, dataloaders.

Uses a custom BPE tokenizer trained on TinyStories (not GPT-2/tiktoken).
Tokenizer is stored in tokenizer/tetra_tokenizer.json.
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
from tqdm import tqdm


# --- Tokenizer --------------------------------------------------------
_tokenizer_cache = None


def get_tokenizer(tokenizer_dir="tokenizer"):
    """Load the custom BPE tokenizer from disk (cached)."""
    global _tokenizer_cache
    if _tokenizer_cache is not None:
        return _tokenizer_cache

    from tokenizers import Tokenizer

    tok_path = Path(tokenizer_dir) / "tetra_tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_path}.\n"
            f"Run: python train_tokenizer.py"
        )

    _tokenizer_cache = Tokenizer.from_file(str(tok_path))
    return _tokenizer_cache


class TokenizerWrapper:
    """Compatibility wrapper for the HuggingFace tokenizers library.

    Provides the same interface we used with tiktoken:
        enc.encode(text) -> list[int]
        enc.decode(ids) -> str
        enc.eot_token -> int (eos_id)
        enc.n_vocab -> int
    """
    def __init__(self, tokenizer):
        self._tok = tokenizer
        vocab = tokenizer.get_vocab()
        self.eot_token = vocab.get("<EOS>", vocab.get("<eos>", 2))
        self.n_vocab = tokenizer.get_vocab_size()

    def encode(self, text):
        """Encode text, returning list of token IDs (no special tokens)."""
        # Disable post_processor (BOS/EOS) for raw encode
        ids = self._tok.encode(text).ids
        # Remove BOS/EOS if post_processor added them
        if ids and ids[0] == self._tok.token_to_id("<BOS>"):
            ids = ids[1:]
        if ids and ids[-1] == self._tok.token_to_id("<EOS>"):
            ids = ids[:-1]
        return ids

    def encode_ordinary(self, text):
        """Alias for encode() -- matches tiktoken API."""
        return self.encode(text)

    def decode(self, ids):
        """Decode token IDs to string."""
        return self._tok.decode(ids)

    def token_to_id(self, token):
        return self._tok.token_to_id(token)


def get_tokenizer_compat(tokenizer_dir="tokenizer"):
    """Return a TokenizerWrapper for backward-compatible API."""
    raw = get_tokenizer(tokenizer_dir)
    return TokenizerWrapper(raw)


# --- Data Download & Tokenize -----------------------------------------

def download_and_tokenize(cache_dir="data", tokenizer_dir="tokenizer", max_stories=None):
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
        print(f"  Tokens: {len(tokens):,} | Vocab: {metadata['vocab_size']}")
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
        print("ERROR: No TinyStories data found!")
        print(f"Please download TinyStoriesV2-GPT4-train.txt to {cache_path}/")
        print("  https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStoriesV2-GPT4-train.txt")
        raise FileNotFoundError("TinyStories data not found")

    print(f"Reading {txt_path} ({txt_path.stat().st_size / 1e6:.0f} MB)...")
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    print("Splitting into stories...")
    stories = text.split("\n\n\n")
    stories = [s.strip() for s in stories if s.strip()]
    print(f"  Found {len(stories):,} stories")

    if max_stories:
        stories = stories[:max_stories]
        print(f"  Using first {max_stories:,} stories")

    eos_id = tok_wrap.eot_token

    print("Tokenizing with custom BPE tokenizer...")
    all_tokens = []
    for story in tqdm(stories, desc="Tokenize", unit="story"):
        token_ids = tok_wrap.encode(story)
        all_tokens.extend(token_ids)
        all_tokens.append(eos_id)

    tokens_array = np.array(all_tokens, dtype=np.uint16)
    total = len(tokens_array)
    print(f"  Total: {len(stories):,} stories, {total:,} tokens")

    tokens_array.tofile(str(bin_path))
    mb = bin_path.stat().st_size / 1e6
    print(f"  Saved to {bin_path} ({mb:.1f} MB)")

    metadata = {"vocab_size": tok_wrap.n_vocab, "total_tokens": total}
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return tokens_array, metadata


# --- Dataset ----------------------------------------------------------

class ChunkedDataset(Dataset):
    """Non-overlapping chunks of tokens. Much faster to shuffle than sliding window."""
    def __init__(self, tokens, block_size):
        n = len(tokens) - (len(tokens) % block_size)
        self.tokens = torch.from_numpy(tokens[:n].astype(np.int64))
        self.block_size = block_size
        self.n_samples = n // block_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = self.tokens[start : start + self.block_size]
        y = self.tokens[start + 1 : start + self.block_size + 1]
        return x, y


def create_dataloaders(tokens, block_size=128, batch_size=8, val_split=0.05):
    split_idx = int(len(tokens) * (1 - val_split))
    train_ds = ChunkedDataset(tokens[:split_idx], block_size)
    val_ds = ChunkedDataset(tokens[split_idx:], block_size)
    print(f"Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
