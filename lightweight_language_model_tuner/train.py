"""Training loop: LoRA fine-tune on CPU with a logged loss curve."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .dataset import InstructionDataset, load_pairs, split_train_eval
from .model import apply_lora, count_trainable, load_base_model, load_tokenizer
from .utils import set_seed, set_threads


def train(cfg: TrainConfig) -> dict:
    t0 = time.time()
    set_seed(cfg.seed)
    set_threads()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(cfg.data_path)
    train_pairs, eval_pairs = split_train_eval(pairs, cfg.eval_frac, cfg.seed)
    # Persist the exact eval split so evaluate.py scores the same rows.
    (out / "eval_pairs.json").write_text(json.dumps(eval_pairs, indent=2))

    tok = load_tokenizer(cfg.model_name)
    train_ds = InstructionDataset(train_pairs, tok, cfg.max_seq_len)
    gen = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                        generator=gen)

    model = load_base_model(cfg.model_name)
    model = apply_lora(model, cfg.lora)
    model.train()
    trainable, total = count_trainable(model)
    print(f"trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.3f}%)")

    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=cfg.lr)
    warmup = max(1, int(cfg.steps * cfg.warmup_frac))

    def lr_lambda(s: int) -> float:
        if s < warmup:
            return (s + 1) / warmup
        return max(0.0, (cfg.steps - s) / max(1, cfg.steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    log_rows: list[dict] = []
    it = iter(loader)
    opt.zero_grad()
    for step in range(1, cfg.steps + 1):
        acc_loss = 0.0
        for _ in range(cfg.grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            outputs = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"])
            (outputs.loss / cfg.grad_accum).backward()
            acc_loss += outputs.loss.item() / cfg.grad_accum
        torch.nn.utils.clip_grad_norm_(train_params, cfg.max_grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad()
        row = {"step": step, "loss": acc_loss,
               "lr": sched.get_last_lr()[0]}
        log_rows.append(row)
        if step % cfg.log_every == 0 or step == cfg.steps:
            print(f"step {step:>4}/{cfg.steps}  loss {acc_loss:.4f}  "
                  f"lr {row['lr']:.2e}", flush=True)

    wall_s = time.time() - t0

    # Artifacts: loss log, curve plot, adapter weights, run metadata.
    with open(out / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "loss", "lr"])
        w.writeheader()
        w.writerows(log_rows)

    steps = [r["step"] for r in log_rows]
    losses = [r["loss"] for r in log_rows]
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, losses, linewidth=1.5)
    plt.xlabel("optimizer step")
    plt.ylabel("train loss (mean NLL, response tokens)")
    plt.title(f"LoRA fine-tune: {cfg.model_name} "
              f"(r={cfg.lora.r}, lr={cfg.lr})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "training_curve.png", dpi=110)
    plt.close()

    model.save_pretrained(out / "lora-adapter")
    tok.save_pretrained(out / "lora-adapter")

    meta = {
        "config": cfg.to_dict(),
        "n_train": len(train_pairs),
        "n_eval": len(eval_pairs),
        "trainable_params": trainable,
        "total_params": total,
        "final_train_loss": log_rows[-1]["loss"],
        "first_train_loss": log_rows[0]["loss"],
        "wall_seconds": round(wall_s, 1),
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {wall_s:.1f}s — artifacts in {out}/")
    return meta
