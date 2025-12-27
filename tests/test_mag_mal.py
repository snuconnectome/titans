import torch
import pytest
from titans_pytorch.models.mag import MemoryAsGate
from titans_pytorch.models.mal import MemoryAsLayer

def test_mag_forward():
    dim = 32
    vocab_size = 100
    model = MemoryAsGate(dim=dim, vocab_size=vocab_size)
    x = torch.randint(0, vocab_size, (1, 10))
    output = model(x)
    assert output.shape == (1, 10, vocab_size)

def test_mal_forward():
    dim = 32
    vocab_size = 100
    model = MemoryAsLayer(dim=dim, vocab_size=vocab_size)
    x = torch.randint(0, vocab_size, (1, 10))
    output = model(x)
    assert output.shape == (1, 10, vocab_size)
