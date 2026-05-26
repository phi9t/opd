from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TrainConfig:
    tier: str
    runtime: str
    device: str
    engine: str
    loss_mode: str
    teacher_signal: str
    max_steps: int
    batch_size: int
    max_new_tokens: int
    lr: float
    seed: int
    run_dir: str
    student_hidden_size: int
    teacher_hidden_size: int
    vocab_size: int
    prompts_path: str
    tokenizer_dir: str
    log_token_samples: int = 4


def load_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text())
    return TrainConfig(
        tier=raw["tier"],
        runtime=raw["runtime"],
        device=raw["device"],
        engine=raw["engine"],
        loss_mode=raw.get("loss", {}).get("mode", "kl"),
        teacher_signal=raw.get("teacher", {}).get("signal", "sampled"),
        max_steps=int(raw.get("max_steps", 20)),
        batch_size=int(raw.get("batch_size", 4)),
        max_new_tokens=int(raw.get("max_new_tokens", 32)),
        lr=float(raw.get("lr", 1e-4)),
        seed=int(raw.get("seed", 42)),
        run_dir=raw.get("run_dir", "runs"),
        student_hidden_size=int(raw["model"]["student_hidden_size"]),
        teacher_hidden_size=int(raw["model"]["teacher_hidden_size"]),
        vocab_size=int(raw["model"]["vocab_size"]),
        prompts_path=raw["data"]["prompts_path"],
        tokenizer_dir=raw["data"]["tokenizer_dir"],
        log_token_samples=int(raw.get("log_token_samples", 4)),
    )


def validate_config(cfg: TrainConfig) -> None:
    if cfg.tier == "tiny" and cfg.runtime != "local":
        raise ValueError("tiny tier requires runtime=local in v1")
    if cfg.tier == "tiny" and cfg.device != "cpu":
        raise ValueError("tiny tier requires device=cpu in v1")
    if cfg.loss_mode == "opd_rl" and cfg.tier == "tiny":
        raise ValueError("opd_rl is M3; use qwen3_small tier")
