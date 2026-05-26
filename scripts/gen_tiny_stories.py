#!/usr/bin/env python3
"""Generate deterministic tiny story prompts and word-level vocab."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = ROOT / "data" / "prompts" / "tiny_stories.jsonl"
VOCAB_PATH = ROOT / "data" / "tokenizer" / "tiny" / "vocab.json"

MAX_VOCAB = 8192
NUM_STORIES = 200
SEED = 42

NAMES = [
    "Alex",
    "Blake",
    "Casey",
    "Drew",
    "Emery",
    "Finley",
    "Gray",
    "Harper",
    "Indigo",
    "Jordan",
    "Kai",
    "Lane",
    "Morgan",
    "Noel",
    "Oakley",
    "Parker",
    "Quinn",
    "River",
    "Sage",
    "Taylor",
]

THINGS = [
    "map",
    "key",
    "lantern",
    "feather",
    "shell",
    "coin",
    "compass",
    "crystal",
    "drum",
    "flute",
    "gem",
    "hat",
    "journal",
    "kite",
    "locket",
    "mirror",
    "necklace",
    "orb",
    "quill",
    "ring",
]

PLACES = [
    "forest",
    "meadow",
    "cave",
    "garden",
    "harbor",
    "hill",
    "island",
    "lake",
    "marsh",
    "mountain",
    "orchard",
    "pond",
    "river",
    "shore",
    "tower",
    "valley",
    "village",
    "woods",
    "yard",
    "zenith",
]

TEMPLATES = [
    "Once upon a time {name} found a {thing} in the {place}.",
    "One sunny day {name} discovered a {thing} near the {place}.",
    "Long ago {name} picked up a {thing} beside the {place}.",
    "In a quiet {place} {name} noticed a shiny {thing}.",
    "At dawn {name} followed a trail to the {place} and found a {thing}.",
]


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_vocab(stories: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for story in stories:
        counts.update(tokenize(story))

    vocab: dict[str, int] = {"<unk>": 0, "<pad>": 1}
    remaining = MAX_VOCAB - len(vocab)
    for word, _ in counts.most_common(remaining):
        if word not in vocab:
            vocab[word] = len(vocab)
    return vocab


def generate_stories() -> list[str]:
    rng = random.Random(SEED)
    stories: list[str] = []
    for i in range(NUM_STORIES):
        template = TEMPLATES[i % len(TEMPLATES)]
        story = template.format(
            name=rng.choice(NAMES),
            thing=rng.choice(THINGS),
            place=rng.choice(PLACES),
        )
        stories.append(story)
    return stories


def main() -> None:
    stories = generate_stories()
    vocab = build_vocab(stories)

    PROMPTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with PROMPTS_PATH.open("w", encoding="utf-8") as f:
        for story in stories:
            f.write(json.dumps({"prompt": story}) + "\n")

    with VOCAB_PATH.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote {len(stories)} prompts to {PROMPTS_PATH}")
    print(f"Wrote vocab ({len(vocab)} tokens) to {VOCAB_PATH}")


if __name__ == "__main__":
    main()
