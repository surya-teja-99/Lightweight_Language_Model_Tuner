"""Instruction dataset: loading, prompt formatting, label masking."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction: str) -> str:
    """Render the prompt prefix (everything the model must NOT be trained on)."""
    return PROMPT_TEMPLATE.format(instruction=instruction)


def load_pairs(path: str | Path) -> list[dict]:
    """Load and validate a JSONL file of {"instruction", "response"} rows."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: row must be a JSON object")
            for key in ("instruction", "response"):
                if key not in obj or not isinstance(obj[key], str) or not obj[key].strip():
                    raise ValueError(f"{path}:{lineno}: row needs non-empty {key!r}")
            rows.append({"instruction": obj["instruction"].strip(),
                         "response": obj["response"].strip()})
    if not rows:
        raise ValueError(f"{path}: no usable rows found")
    return rows


def split_train_eval(pairs: list[dict], eval_frac: float, seed: int):
    """Deterministic shuffle split; train and eval are disjoint."""
    idx = list(range(len(pairs)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_eval = max(1, int(len(pairs) * eval_frac))
    eval_idx = set(idx[:n_eval])
    train = [p for i, p in enumerate(pairs) if i not in eval_idx]
    ev = [p for i, p in enumerate(pairs) if i in eval_idx]
    return train, ev


def encode_example(tokenizer, instruction: str, response: str, max_len: int) -> dict:
    """Tokenize one pair; labels are -100 (ignored) over the prompt region."""
    prompt = format_prompt(instruction)
    full = prompt + response
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        full, add_special_tokens=False, truncation=True, max_length=max_len
    )["input_ids"]
    n_prompt = min(len(prompt_ids), len(full_ids))
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    pad_id = tokenizer.pad_token_id
    pad_len = max_len - len(full_ids)
    return {
        "input_ids": full_ids + [pad_id] * pad_len,
        "attention_mask": [1] * len(full_ids) + [0] * pad_len,
        "labels": labels + [-100] * pad_len,
    }


class InstructionDataset(Dataset):
    """Pre-tokenized instruction dataset (small data: encode once up front)."""

    def __init__(self, pairs: list[dict], tokenizer, max_len: int):
        self.examples = [
            {k: torch.tensor(v, dtype=torch.long)
             for k, v in encode_example(tokenizer, p["instruction"],
                                        p["response"], max_len).items()}
            for p in pairs
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        return self.examples[i]
