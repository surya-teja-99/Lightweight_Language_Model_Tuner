#!/usr/bin/env python3
"""Download a slice of databricks-dolly-15k as instruction/response JSONL.

Usage:
    python data/download_dolly.py --n 1000 --out data/dolly_slice.jsonl

Requires the `datasets` package. Falls back gracefully if the Hub is
unreachable — the bundled data/sample_instructions.jsonl always works offline.
"""

import argparse
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="rows to take")
    ap.add_argument("--out", default="data/dolly_slice.jsonl")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("the 'datasets' package is required: pip install datasets")

    ds = load_dataset("databricks/databricks-dolly-15k",
                      split="train", streaming=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in ds:
            instruction = row["instruction"].strip()
            if row.get("context"):
                instruction += "\n\nContext:\n" + row["context"].strip()
            f.write(json.dumps({
                "instruction": instruction,
                "response": row["response"].strip(),
            }) + "\n")
            n += 1
            if n >= args.n:
                break
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
