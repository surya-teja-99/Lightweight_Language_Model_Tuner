"""Evaluation: held-out perplexity before/after LoRA + generation samples."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import TrainConfig
from .dataset import InstructionDataset, format_prompt
from .model import load_base_model, load_tokenizer
from .utils import set_seed, set_threads

N_GENERATION_PROMPTS = 3
MAX_NEW_TOKENS = 48


@torch.no_grad()
def perplexity(model, dataset, batch_size: int = 4) -> float:
    """Mean perplexity over response tokens only (labels != -100)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    nll_sum, tok_count = 0.0, 0
    for batch in loader:
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"])
        n = int((batch["labels"] != -100).sum())
        nll_sum += out.loss.item() * n
        tok_count += n
    return math.exp(nll_sum / max(1, tok_count))


@torch.no_grad()
def generate(model, tok, instruction: str) -> str:
    prompt = format_prompt(instruction)
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    gen = model.generate(
        ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    text = tok.decode(gen[0], skip_special_tokens=True)
    return text[len(prompt):] if text.startswith(prompt) else text


def evaluate(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    set_threads()
    out = Path(cfg.out_dir)

    eval_pairs = json.loads((out / "eval_pairs.json").read_text())
    tok = load_tokenizer(cfg.model_name)
    eval_ds = InstructionDataset(eval_pairs, tok, cfg.max_seq_len)

    # One model instance throughout: score base first, then wrap with LoRA.
    model = load_base_model(cfg.model_name)
    model.eval()
    ppl_base = perplexity(model, eval_ds)
    print(f"base eval perplexity:   {ppl_base:.2f}")
    base_gens = [generate(model, tok, p["instruction"])
                 for p in eval_pairs[:N_GENERATION_PROMPTS]]

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(out / "lora-adapter"))
    model.eval()
    ppl_lora = perplexity(model, eval_ds)
    print(f"LoRA eval perplexity:   {ppl_lora:.2f}")
    lora_gens = [generate(model, tok, p["instruction"])
                 for p in eval_pairs[:N_GENERATION_PROMPTS]]

    report = {
        "n_eval": len(eval_pairs),
        "eval_perplexity_base": round(ppl_base, 3),
        "eval_perplexity_lora": round(ppl_lora, 3),
        "perplexity_delta_pct": round(100 * (ppl_lora - ppl_base) / ppl_base, 2),
        "note": "perplexity over response tokens only, greedy decoding",
    }
    (out / "eval_report.json").write_text(json.dumps(report, indent=2))

    lines = ["# Generation samples (greedy, 48 new tokens)\n"]
    for i, p in enumerate(eval_pairs[:N_GENERATION_PROMPTS]):
        lines.append(f"## Prompt {i + 1}\n\n{p['instruction']}\n")
        lines.append(f"### Base ({cfg.model_name})\n\n{base_gens[i].strip()}\n")
        lines.append("### + LoRA adapter\n\n"
                     f"{lora_gens[i].strip()}\n")
        lines.append(f"Reference: {p['response']}\n")
    (out / "generations.md").write_text("\n".join(lines))
    print(f"eval artifacts in {out}/")
    return report
