from __future__ import annotations

import json
from pathlib import Path

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
UNK_ID = 0
PAD_ID = 1


class TinyTokenizer:
    def __init__(self, tokenizer_dir: str | Path) -> None:
        path = Path(tokenizer_dir) / "vocab.json"
        with path.open(encoding="utf-8") as f:
            self._token_to_id: dict[str, int] = json.load(f)
        self._id_to_token = {i: t for t, i in self._token_to_id.items()}
        if self._token_to_id.get(UNK_TOKEN) != UNK_ID:
            raise ValueError(f"{UNK_TOKEN} must map to id {UNK_ID}")
        if self._token_to_id.get(PAD_TOKEN) != PAD_ID:
            raise ValueError(f"{PAD_TOKEN} must map to id {PAD_ID}")

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def encode(self, text: str) -> list[int]:
        return [self._token_to_id.get(word, UNK_ID) for word in text.lower().split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self._id_to_token.get(i, UNK_TOKEN) for i in ids)
