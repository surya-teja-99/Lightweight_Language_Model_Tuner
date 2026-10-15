"""Model + tokenizer loading and LoRA application."""

from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import LoRAConfig


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 family has no pad token; reuse eos so batching works.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(model_name: str):
    return AutoModelForCausalLM.from_pretrained(model_name)


def apply_lora(model, lora_cfg: LoRAConfig):
    from peft import LoraConfig, TaskType, get_peft_model

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=lora_cfg.target_modules,
        # GPT-2's c_attn is a Conv1D layer (transposed weight layout).
        fan_in_fan_out=True,
        bias="none",
    )
    return get_peft_model(model, peft_cfg)


def count_trainable(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
