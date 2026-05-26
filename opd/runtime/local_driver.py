from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from opd.actor import StudentActor
from opd.batches import TrainBatch
from opd.config import TrainConfig
from opd.data.prompts import load_prompts
from opd.data.tokenizer_tiny import PAD_ID, TinyTokenizer
from opd.metrics import append_metrics_jsonl, write_step_json
from opd.models.synthetic_qwen3 import SyntheticQwen3Config
from opd.rollout import StudentRollout
from opd.teacher import TeacherScorer


def _state_dict_bytes(state_dict: dict) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())


def _pad_prompt_ids(encoded: list[list[int]]) -> torch.Tensor:
    max_len = max(len(ids) for ids in encoded)
    padded = [ids + [PAD_ID] * (max_len - len(ids)) for ids in encoded]
    return torch.tensor(padded, dtype=torch.long)


def _build_token_samples(
    prompt_texts: list[str],
    token_ids: torch.Tensor,
    per_token_kl: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
    n_keep: int,
) -> list[dict]:
    """Pick the top-`n_keep` sequences with the largest single-token |KL| spikes.

    Ranks by `per_token_kl.abs().amax(dim=1)` so spikes in either direction surface,
    matching the magnitude convention used by the explorer's TokenHeatmap. Returns a
    list of {"prompt", "tokens", "kl"} dicts trimmed to each sequence's unmasked length.
    """
    if n_keep <= 0:
        return []

    masked_kl = per_token_kl.abs() * response_mask
    per_seq_max = masked_kl.amax(dim=1) if masked_kl.numel() > 0 else torch.empty(0)
    k = min(n_keep, per_seq_max.numel())
    if k == 0:
        return []
    top_idx = torch.topk(per_seq_max, k=k).indices.tolist()

    samples: list[dict] = []
    for i in top_idx:
        length = int(response_mask[i].sum().item())
        if length == 0:
            continue
        ids = token_ids[i, :length].tolist()
        id_to_token = tokenizer._id_to_token
        tokens = [id_to_token[t] if t in id_to_token else f"‹{t}›" for t in ids]
        kls = per_token_kl[i, :length].tolist()
        samples.append(
            {
                "prompt": prompt_texts[i],
                "tokens": tokens,
                "kl": [float(x) for x in kls],
            }
        )
    return samples


class LocalDriver:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        self.run_dir = Path(cfg.run_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        config_path = self.run_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(asdict(cfg)), encoding="utf-8")

        self.tokenizer = TinyTokenizer(cfg.tokenizer_dir)
        self.prompts = load_prompts(cfg.prompts_path)

        student_cfg = SyntheticQwen3Config(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.student_hidden_size,
        )
        self.rollout = StudentRollout(
            student_cfg,
            device=cfg.device,
            max_new_tokens=cfg.max_new_tokens,
        )
        self.teacher = TeacherScorer(cfg, device=cfg.device)
        self.actor = StudentActor(cfg, device=cfg.device)

    def _sample_prompts(self, step: int) -> list[str]:
        n = len(self.prompts)
        return [
            self.prompts[(step * self.cfg.batch_size + i) % n]
            for i in range(self.cfg.batch_size)
        ]

    def _tokenize_prompts(self, texts: list[str]) -> torch.Tensor:
        encoded = [self.tokenizer.encode(text) for text in texts]
        return _pad_prompt_ids(encoded)

    def run(self) -> None:
        torch.manual_seed(self.cfg.seed)
        cfg = self.cfg

        for step in range(cfg.max_steps):
            texts = self._sample_prompts(step)
            prompt_ids = self._tokenize_prompts(texts)

            t0 = time.perf_counter()
            trajectory = self.rollout.generate(prompt_ids)
            gen_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            teacher_batch = self.teacher.score(trajectory)
            teacher_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            train_metrics = self.actor.train_step(
                TrainBatch(trajectory=trajectory, teacher=teacher_batch)
            )
            train_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            state_dict = self.actor.model.state_dict()
            sync_bytes = _state_dict_bytes(state_dict)
            self.rollout.model.load_state_dict(state_dict)
            sync_ms = (time.perf_counter() - t0) * 1000.0

            response_lengths = trajectory.token_ids.shape[1]
            num_tokens = int(trajectory.token_ids.numel())

            samples = _build_token_samples(
                prompt_texts=texts,
                token_ids=trajectory.token_ids.detach().cpu(),
                per_token_kl=train_metrics["per_token_kl"],
                response_mask=train_metrics["response_mask"],
                tokenizer=self.tokenizer,
                n_keep=cfg.log_token_samples,
            )

            payload = {
                "step": step,
                "gen_ms": gen_ms,
                "teacher_ms": teacher_ms,
                "train_ms": train_ms,
                "sync_ms": sync_ms,
                "sync_bytes": sync_bytes,
                "num_tokens": num_tokens,
                "mean_response_length": float(response_lengths),
                "mean_kl": train_metrics["mean_kl"],
                "loss": train_metrics["loss"],
                "grad_norm": train_metrics["grad_norm"],
                "loss_mode": cfg.loss_mode,
                "teacher_signal": cfg.teacher_signal,
                "samples": samples,
            }

            write_step_json(self.run_dir, step, payload)
            append_metrics_jsonl(self.run_dir, payload)
