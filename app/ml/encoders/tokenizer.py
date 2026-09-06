"""CLIP-compatible byte-level BPE tokenizer loaded from verified vocab files."""

from __future__ import annotations

import gzip
import html
import json
import re
from functools import lru_cache
from pathlib import Path

import torch

from app.constants import CLIP_TOKENIZER_CONTEXT_LEN

__all__ = ["CliPTokenizer", "basic_clean", "bytes_to_unicode", "whitespace_clean"]

_SOT = "<|startoftext|>"
_EOT = "<|endoftext|>"
_DEFAULT_SOT_ID = 49406
_DEFAULT_EOT_ID = 49407
_WHITESPACE = re.compile(r"\s+")


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """Build the reversible byte-to-unicode map used by CLIP/GPT-2 BPE.

    Returns:
        Mapping of 256 byte values to unicode characters.
    """
    values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    extra = [chr(value) for value in range(2**8) if value not in values]
    characters = [chr(value) for value in values] + extra
    return dict(zip(range(2**8), characters, strict=True))


def basic_clean(text: str) -> str:
    """Minimal HTML/entity cleanup used before tokenization.

    Parameters:
        text: Raw text.

    Returns:
        Cleaned text.
    """
    import ftfy  # type: ignore[import-not-found]

    return ftfy.fix_text(text)


def whitespace_clean(text: str) -> str:
    """Collapse whitespace runs to single spaces.

    Parameters:
        text: Input text.

    Returns:
        Cleaned text.
    """
    return _WHITESPACE.sub(" ", text).strip()


class CliPTokenizer:
    """Byte-level BPE tokenizer matching the CLIP vocabulary format.

    Loads `vocab.json` (token->id) and `merges.txt` (BPE merge rules) exported
    from the OpenAI CLIP ViT-B/16 checkpoint; behavior (SOT/EOT padding to a
    fixed 77-token context, lowercase, punctuation splitting, byte-level
    merges) is identical to the reference implementation.
    """

    __slots__ = (
        "bpe_ranks",
        "byte_decoder",
        "byte_encoder",
        "cache",
        "context_len",
        "decoder",
        "encoder",
        "eot_id",
        "eot_token",
        "merges",
        "sot_id",
        "sot_token",
    )

    def __init__(self, vocab_path: Path, merges_path: Path) -> None:
        """Load vocabulary and merges from disk.

        Parameters:
            vocab_path: Path to clip_bpe_vocab.json.
            merges_path: Path to clip_bpe_merges.txt.

        Raises:
            ValueError: When files are malformed or missing special tokens.
        """
        raw = vocab_path.read_text(encoding="utf-8")
        if vocab_path.suffix == ".gz":
            raw = gzip.decompress(raw.encode("utf-8")).decode("utf-8")
        encoder = json.loads(raw)
        if not isinstance(encoder, dict):
            raise TypeError("vocab file must contain a JSON object")
        self.encoder: dict[str, int] = {str(k): int(v) for k, v in encoder.items()}
        self.decoder: dict[int, str] = {v: k for k, v in self.encoder.items()}
        merges_text = merges_path.read_text(encoding="utf-8")
        if merges_path.suffix == ".gz":
            merges_text = gzip.decompress(merges_text.encode("utf-8")).decode("utf-8")
        merge_lines = [line for line in merges_text.split("\n") if line and not line.startswith("#")]
        self.merges = [tuple(line.split()) for line in merge_lines]
        if not all(len(pair) == 2 for pair in self.merges):
            raise ValueError("merges file contains malformed lines")
        self.bpe_ranks = {pair: index for index, pair in enumerate(self.merges)}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache: dict[str, str] = {}
        if _SOT not in self.encoder or _EOT not in self.encoder:
            raise ValueError("vocabulary is missing CLIP special tokens")
        self.sot_token = _SOT
        self.eot_token = _EOT
        self.sot_id = self.encoder[_SOT]
        self.eot_id = self.encoder[_EOT]
        self.context_len = CLIP_TOKENIZER_CONTEXT_LEN

    def _bpe(self, token: str) -> str:
        """Apply BPE merges to one whitespace-free token.

        Parameters:
            token: Byte-encoded token.

        Returns:
            The merged symbol string.
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = self._get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            index = 0
            while index < len(word):
                try:
                    first_hit = word.index(first, index)
                except ValueError:
                    new_word.extend(word[index:])
                    break
                new_word.extend(word[index:first_hit])
                index = first_hit
                if word[index] == first and index < len(word) - 1 and word[index + 1] == second:
                    new_word.append(first + second)
                    index += 2
                else:
                    new_word.append(word[index])
                    index += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    @staticmethod
    def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
        """Collect adjacent symbol pairs.

        Parameters:
            word: Symbol tuple.

        Returns:
            Set of adjacent pairs.
        """
        return {(word[i], word[i + 1]) for i in range(len(word) - 1)}

    def _tokenize_text(self, text: str) -> list[str]:
        """Run the CLIP text preprocessing + BPE pipeline.

        Parameters:
            text: Raw prompt text.

        Returns:
            BPE token strings (without SOT/EOT).
        """
        cleaned = whitespace_clean(basic_clean(html.unescape(text))).lower()
        cleaned = re.sub(
            r"([.,!?:;(){}\[\]\"'])",
            r" \1 ",
            cleaned,
        )
        cleaned = whitespace_clean(cleaned)
        tokens: list[str] = []
        for whitespace_token in cleaned.split(" "):
            if not whitespace_token:
                continue
            encoded = "".join(self.byte_encoder[byte] for byte in whitespace_token.encode("utf-8"))
            tokens.extend(self._bpe(encoded).split(" "))
        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode text to a fixed-length id list (SOT, tokens..., EOT, 0-pad).

        Parameters:
            text: Prompt text (empty string encodes to SOT+EOT, the true
                unconditional sequence).

        Returns:
            Exactly `context_len` integer ids.
        """
        ids = [self.sot_id]
        for token in self._tokenize_text(text)[: self.context_len - 2]:
            token_id = self.encoder.get(token)
            if token_id is not None:
                ids.append(token_id)
        ids.append(self.eot_id)
        while len(ids) < self.context_len:
            ids.append(0)
        return ids

    def encode_batch(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of prompts.

        Parameters:
            texts: Prompt strings.

        Returns:
            [B, context_len] long tensor.
        """
        return torch.tensor([self.encode(text) for text in texts], dtype=torch.long)

    def decode(self, ids: list[int]) -> str:
        """Decode ids back to text (debug helper).

        Parameters:
            ids: Token ids without padding.

        Returns:
            The decoded string.
        """
        tokens = [self.decoder.get(identifier, "") for identifier in ids]
        symbols = []
        for token in tokens:
            if token in (_SOT, _EOT, ""):
                continue
            symbols.extend(token.split(" "))
        bytes_out = bytes(self.byte_decoder.get(symbol, 32) for symbol in symbols)
        return bytes_out.decode("utf-8", errors="replace").strip()

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return len(self.encoder)
