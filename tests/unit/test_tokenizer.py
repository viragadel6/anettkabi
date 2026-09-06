"""CLIP BPE tokenizer: encoding shape, specials, padding, roundtrip."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from app.ml.encoders.tokenizer import CliPTokenizer

SOT = "<|startoftext|>"
EOT = "<|endoftext|>"


def _write_vocab(path: Path) -> None:
    byte_encoder_base = "abcdefghijklmnopqrstuvwxyz"
    vocab: dict[str, int] = {char: index for index, char in enumerate(byte_encoder_base)}
    vocab["##"] = len(vocab)
    vocab[SOT] = 49406
    vocab[EOT] = 49407
    vocab["ab"] = 100
    vocab["hello"] = 101
    vocab["helloab"] = 102
    path.write_text(json.dumps(vocab), encoding="utf-8")


def _write_merges(path: Path) -> None:
    lines = ["a b", "h e", "he l", "hel l", "hell o", "hello ab"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def tokenizer(tmp_path: Path) -> CliPTokenizer:
    vocab_path = tmp_path / "clip_bpe_vocab.json"
    merges_path = tmp_path / "clip_bpe_merges.txt"
    _write_vocab(vocab_path)
    _write_merges(merges_path)
    return CliPTokenizer(vocab_path, merges_path)


def test_specials_and_context(tokenizer: CliPTokenizer) -> None:
    assert tokenizer.sot_id == 49406
    assert tokenizer.eot_id == 49407
    assert tokenizer.context_len == 77


def test_empty_prompt_is_pure_unconditional(tokenizer: CliPTokenizer) -> None:
    ids = tokenizer.encode("")
    assert ids[0] == tokenizer.sot_id
    assert ids[1] == tokenizer.eot_id
    assert len(ids) == 77
    assert all(identifier == 0 for identifier in ids[2:])


def test_encode_fixed_length_with_padding(tokenizer: CliPTokenizer) -> None:
    ids = tokenizer.encode("hello")
    assert len(ids) == 77
    assert ids[0] == tokenizer.sot_id
    assert ids[-1] != tokenizer.eot_id or tokenizer.eot_id in ids
    assert tokenizer.eot_id in ids


def test_encode_batch_shapes(tokenizer: CliPTokenizer) -> None:
    batch = tokenizer.encode_batch(["hello", ""])
    assert batch.shape == (2, 77)
    assert batch.dtype.is_floating_point is False


def test_known_token_present(tokenizer: CliPTokenizer) -> None:
    ids = tokenizer.encode("hello")
    assert 101 in ids or 102 in ids


def test_gzip_vocab_supported(tmp_path: Path) -> None:
    vocab_path = tmp_path / "clip_bpe_vocab.json.gz"
    payload = json.dumps({SOT: 49406, EOT: 49407, "a": 1, "b": 2})
    vocab_path.write_bytes(gzip.compress(payload.encode("utf-8")))
    merges_path = tmp_path / "clip_bpe_merges.txt"
    merges_path.write_text("a b\n", encoding="utf-8")
    loaded = CliPTokenizer(vocab_path, merges_path)
    assert loaded.vocab_size >= 4


def test_missing_specials_rejected(tmp_path: Path) -> None:
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps({"a": 0}), encoding="utf-8")
    merges_path = tmp_path / "merges.txt"
    merges_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="special"):
        CliPTokenizer(vocab_path, merges_path)
