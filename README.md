# LoRAForge

Fine-tune a small open language model with **LoRA** (Low-Rank Adaptation) on a
CPU-only machine — real training, real training curves, real eval numbers.
No GPU, no skeleton code: `make demo` runs the full pipeline end to end.

## What it does

LoRAForge takes `distilgpt2` (82M params, open weights), freezes it, and trains
a tiny LoRA adapter (`r=8` on the fused QKV projection `c_attn`) on 100
instruction/response pairs. It then reports:

- a **training loss curve** (`results/training_curve.png`),
- **held-out perplexity before vs. after** the adapter (`results/eval_report.json`),
- **side-by-side generations** from the base model and the adapted model
  (`results/generations.md`).

## Architecture

```
data/sample_instructions.jsonl            (100 instruction/response pairs)
        │
        ▼
┌───────────────────┐
│  prompt template  │  "### Instruction:\n...\n\n### Response:\n"
│  + label masking  │  labels = -100 over the prompt (model learns
└────────┬──────────┘  responses only), padded to 96 tokens
         ▼
┌────────────────────────────────────────────┐
│ distilgpt2 (frozen, 82M) + LoRA(r=8)        │
│ target: c_attn · <0.2% of params trainable  │
│ (exact count printed + logged at train time) │
│ AdamW 5e-4 · linear warmup+decay · CPU      │
└────────┬───────────────────────────────────┘
         ▼
results/
 ├── train_log.csv        per-step loss + lr
 ├── training_curve.png   the actual loss curve
 ├── lora-adapter/        trained adapter weights (PEFT)
 ├── eval_pairs.json      exact held-out split (reproducible eval)
 ├── eval_report.json     perplexity before/after
 └── generations.md       base vs. adapter samples
```

## Quickstart

```bash
make setup   # venv + pinned CPU-only deps (~3 min, one-time)
make demo    # train (120 steps) + eval + tests
```

Or step by step:

```bash
.venv/bin/python scripts/train_demo.py   # fine-tune, writes results/
.venv/bin/python scripts/eval_demo.py    # perplexity + generation samples
.venv/bin/python -m pytest tests/ -q     # unit tests
```

Train on a bigger slice of real data (needs the `datasets` package, included):

```bash
.venv/bin/python data/download_dolly.py --n 1000 --out data/dolly_slice.jsonl
.venv/bin/python scripts/train_demo.py --data data/dolly_slice.jsonl --steps 300
```

## Results

*Measured on this repo's demo run (2 vCPU, CPU-only PyTorch). Numbers below are
copied from `results/train_meta.json` and `results/eval_report.json` — nothing
here is invented.*

- Trainable parameters: **147,456** of **82,060,032** total (**0.180%**)
- Train loss: **3.72** (step 1) → **1.63** (step 120)
- Wall time: **355 s** (~6 min on 2 vCPUs, CPU-only)
- Held-out perplexity (response tokens, n=20): base **17.572** → +LoRA **13.126**
  (Δ **−25.3%**)

See `results/generations.md` for unedited base-vs-adapter samples.

## Limitations (honest)

- **Tiny data, tiny model.** 100 short pairs on an 82M model will not make a
  capable assistant; the point is the *pipeline*, not the checkpoint.
- **CPU training is slow.** 120 steps takes minutes here; scaling to thousands
  of examples wants a GPU (the code is device-agnostic — only the default is CPU).
- **Perplexity on 20 held-out short responses** is a weak signal; a real eval
  would use a larger held-out set and task-based metrics.
- Greedy generations from `distilgpt2` are often bland or repetitive — the
  samples in `results/generations.md` are unedited, warts and all.

## Project layout

```
LoRAForge/
├── loraforge/          # the package
│   ├── config.py       # TrainConfig / LoRAConfig dataclasses
│   ├── dataset.py      # JSONL loading, prompt template, label masking
│   ├── model.py        # tokenizer + base model + PEFT LoRA wiring
│   ├── train.py        # CPU training loop, logging, curve plot, adapter save
│   ├── evaluate.py     # held-out perplexity + generation samples
│   └── utils.py        # seeding, thread setup
├── scripts/            # train_demo.py, eval_demo.py CLIs
├── data/               # sample_instructions.jsonl (+ dolly download script)
├── tests/              # pytest suite (no weight downloads needed)
├── results/            # produced by a real run (committed for transparency)
├── Makefile
└── requirements.txt    # pinned
```

## Requirements

Python 3.10+, pip. CPU-only PyTorch (no CUDA needed). ~500 MB disk for the
`distilgpt2` weights (cached by Hugging Face Hub on first run).
