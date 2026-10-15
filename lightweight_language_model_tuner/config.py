"""Configuration dataclasses for Lightweight_Language_Model_Tuner."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class LoRAConfig:
    """LoRA adapter hyperparameters (PEFT)."""

    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    # GPT-2 style models fuse Q/K/V into one projection named "c_attn".
    target_modules: list = field(default_factory=lambda: ["c_attn"])


@dataclass
class TrainConfig:
    """End-to-end demo configuration. `steps` counts optimizer steps."""

    model_name: str = "distilgpt2"
    data_path: str = "data/sample_instructions.jsonl"
    eval_frac: float = 0.2
    seed: int = 42
    max_seq_len: int = 96
    batch_size: int = 2
    grad_accum: int = 2
    steps: int = 120
    lr: float = 5e-4
    warmup_frac: float = 0.1
    max_grad_norm: float = 1.0
    log_every: int = 10
    out_dir: str = "results"
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        d = dict(d)
        if "lora" in d and isinstance(d["lora"], dict):
            d["lora"] = LoRAConfig(**d["lora"])
        return cls(**d)

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
