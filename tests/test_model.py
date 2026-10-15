"""Model tests that need no weight downloads.

- LoRA target modules are checked against a randomly-initialized model built
  from the downloaded *config* only (a few KB).
- The tokenizer test needs the small tokenizer files from the Hub.
"""

from transformers import AutoConfig, AutoModelForCausalLM

from loraforge.config import TrainConfig
from loraforge.model import load_tokenizer


def test_lora_targets_exist_in_architecture():
    cfg = TrainConfig()
    model_cfg = AutoConfig.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_config(model_cfg)  # no weights fetched
    leaf_names = {n.split(".")[-1] for n, _ in model.named_modules()}
    for target in cfg.lora.target_modules:
        assert target in leaf_names, f"{target!r} not a submodule of {cfg.model_name}"


def test_tokenizer_pad_defaults_to_eos():
    tok = load_tokenizer("distilgpt2")
    assert tok.pad_token_id == tok.eos_token_id
