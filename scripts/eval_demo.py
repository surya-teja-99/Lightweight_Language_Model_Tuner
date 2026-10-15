#!/usr/bin/env python3
"""CLI: evaluate the trained LoRA adapter (perplexity + generations)."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightweight_language_model_tuner.config import TrainConfig
from lightweight_language_model_tuner.evaluate import evaluate


def main() -> None:
    ap = argparse.ArgumentParser(description="Lightweight_Language_Model_Tuner evaluation")
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--out", default="results",
                    help="directory holding the trained adapter + eval split")
    args = ap.parse_args()

    cfg = TrainConfig(model_name=args.model, out_dir=args.out)
    report = evaluate(cfg)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
