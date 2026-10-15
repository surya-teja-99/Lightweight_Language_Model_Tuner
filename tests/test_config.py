"""Unit tests for configuration defaults and round-tripping."""

from lightweight_language_model_tuner.config import LoRAConfig, TrainConfig


def test_demo_defaults_sane():
    cfg = TrainConfig()
    assert cfg.model_name == "distilgpt2"
    assert cfg.steps > 0
    assert cfg.lr > 0
    assert cfg.lora.r > 0
    assert cfg.lora.target_modules, "need at least one LoRA target module"
    assert 0 < cfg.eval_frac < 1


def test_config_roundtrip(tmp_path):
    cfg = TrainConfig(steps=5, lora=LoRAConfig(r=4))
    p = tmp_path / "cfg.json"
    cfg.save(p)
    back = TrainConfig.load(p)
    assert back.steps == 5
    assert back.lora.r == 4
    assert back.model_name == cfg.model_name
