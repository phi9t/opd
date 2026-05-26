from opd.data.prompts import load_prompts
from opd.data.tokenizer_tiny import TinyTokenizer


def test_load_prompts():
    prompts = load_prompts("data/prompts/tiny_stories.jsonl", limit=5)
    assert len(prompts) == 5


def test_tokenizer_roundtrip():
    tok = TinyTokenizer("data/tokenizer/tiny")
    ids = tok.encode("hello world")
    assert len(ids) >= 1
