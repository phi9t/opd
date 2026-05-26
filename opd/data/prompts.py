from __future__ import annotations

import json
from pathlib import Path


def load_prompts(path: str | Path, limit: int | None = None) -> list[str]:
    prompts: list[str] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(prompts) >= limit:
                break
            row = json.loads(line)
            prompts.append(row["prompt"])
    return prompts
