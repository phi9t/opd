import torch

from opd.models.synthetic_qwen3 import SyntheticQwen3Config, SyntheticQwen3ForCausalLM


def test_forward_logits_shape():
    cfg = SyntheticQwen3Config(vocab_size=128, hidden_size=64, num_hidden_layers=2)
    m = SyntheticQwen3ForCausalLM(cfg)
    x = torch.randint(0, 128, (2, 8))
    logits = m(x).logits
    assert logits.shape == (2, 8, 128)


def test_logprobs_on_sampled_tokens():
    cfg = SyntheticQwen3Config(vocab_size=64, hidden_size=32, num_hidden_layers=2)
    m = SyntheticQwen3ForCausalLM(cfg)
    prompt = torch.tensor([[1, 2, 3]])
    cont = torch.tensor([[4, 5]])
    full = torch.cat([prompt, cont], dim=1)
    lp = m.logprobs_on_tokens(full, cont)
    assert lp.shape == (1, 2)
