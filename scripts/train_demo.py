#!/usr/bin/env python3
"""CLI: run the LoRA fine-tuning demo end to end."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loraforge.config import TrainConfig
from loraforge.train import train


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRAForge training demo")
    ap.add_argument("--data", default="data/sample_instructions.jsonl",
                    help="JSONL file of instruction/response pairs")
    ap.add_argument("--steps", type=int, default=120,
                    help="optimizer steps (demo default: 120)")
    ap.add_argument("--model", default="distilgpt2",
                    help="base model id on the Hugging Face Hub")
    ap.add_argument("--out", default="results",
                    help="output directory for artifacts")
    args = ap.parse_args()

    cfg = TrainConfig(data_path=args.data, steps=args.steps,
                      model_name=args.model, out_dir=args.out)
    meta = train(cfg)
    print(json.dumps({k: meta[k] for k in
                      ("trainable_params", "total_params",
                       "first_train_loss", "final_train_loss",
                       "wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
