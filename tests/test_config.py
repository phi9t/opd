from pathlib import Path

import pytest

from opd.config import TrainConfig, load_config, validate_config


def test_load_tier_tiny():
    cfg = load_config(Path("configs/tier_tiny.yaml"))
    assert cfg.tier == "tiny"
    assert cfg.runtime == "local"
    assert cfg.device == "cpu"
    assert cfg.engine == "eager"
    assert cfg.loss_mode == "kl"


def test_validate_rejects_tiny_opd_rl():
    cfg = TrainConfig(
        tier="tiny",
        runtime="local",
        device="cpu",
        engine="eager",
        loss_mode="opd_rl",
        teacher_signal="sampled",
        max_steps=20,
        batch_size=4,
        max_new_tokens=32,
        lr=1e-4,
        seed=42,
        run_dir="runs",
        student_hidden_size=256,
        teacher_hidden_size=512,
        vocab_size=8192,
        prompts_path="data/prompts/tiny_stories.jsonl",
        tokenizer_dir="data/tokenizer/tiny",
    )
    with pytest.raises(ValueError, match="opd_rl"):
        validate_config(cfg)
