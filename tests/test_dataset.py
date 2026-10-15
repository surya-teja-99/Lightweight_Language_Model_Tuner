"""Unit tests for dataset loading, formatting, and label masking."""

import json

import pytest

from loraforge.dataset import (
    InstructionDataset,
    encode_example,
    format_prompt,
    load_pairs,
    split_train_eval,
)


class FakeTokenizer:
    """Deterministic char-level stand-in: no downloads, no torch."""

    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False,
                 truncation=False, max_length=None):
        ids = [(ord(c) % 997) + 1 for c in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def test_format_prompt_exact():
    assert format_prompt("Do X.") == "### Instruction:\nDo X.\n\n### Response:\n"


def test_load_pairs_roundtrip(tmp_path):
    p = tmp_path / "d.jsonl"
    rows = [{"instruction": "a?", "response": "b."},
            {"instruction": "c?", "response": "d."}]
    p.write_text("\n".join(json.dumps(r) for r in rows))
    assert load_pairs(p) == rows


def test_load_pairs_rejects_bad_rows(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"instruction": "a?"}) + "\n")  # no response
    with pytest.raises(ValueError, match="response"):
        load_pairs(p)


def test_load_pairs_rejects_empty_file(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("\n")
    with pytest.raises(ValueError, match="no usable rows"):
        load_pairs(p)


def test_encode_masks_prompt_region():
    tok = FakeTokenizer()
    # max_len must exceed the char-level encoding length (39 ids here);
    # with max_len=32 the whole example would truncate into the prompt region.
    enc = encode_example(tok, "hi", "there", max_len=64)
    prompt_len = len(tok(format_prompt("hi"),
                         add_special_tokens=False)["input_ids"])
    assert all(v == -100 for v in enc["labels"][:prompt_len])
    # response region keeps real ids
    assert any(v != -100 for v in enc["labels"][prompt_len:])
    # shapes line up and padding is consistent
    assert len(enc["input_ids"]) == len(enc["labels"]) == 64
    assert len(enc["attention_mask"]) == 64
    assert enc["input_ids"][-1] == tok.pad_token_id
    assert enc["labels"][-1] == -100
    assert enc["attention_mask"][-1] == 0


def test_split_deterministic_and_disjoint():
    pairs = [{"instruction": f"q{i}", "response": "r"} for i in range(50)]
    t1, e1 = split_train_eval(pairs, 0.2, seed=7)
    t2, e2 = split_train_eval(pairs, 0.2, seed=7)
    assert t1 == t2 and e1 == e2
    assert len(e1) == 10
    assert len(t1) == 40
    assert not (set(map(str, t1)) & set(map(str, e1)))


def test_instruction_dataset_shapes():
    import torch

    pairs = [{"instruction": "q?", "response": "a."} for _ in range(3)]
    ds = InstructionDataset(pairs, FakeTokenizer(), max_len=24)
    assert len(ds) == 3
    ex = ds[0]
    assert ex["input_ids"].shape == (24,)
    assert ex["input_ids"].dtype == torch.long
