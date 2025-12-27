import torch
import pytest
from titans_pytorch.models.mal import MemoryAsLayer

def test_mal_forward_structure():
    dim = 32
    vocab_size = 100
    model = MemoryAsLayer(dim=dim, vocab_size=vocab_size, num_heads=4)
    
    x = torch.randint(0, vocab_size, (1, 20))
    output = model(x)
    assert output.shape == (1, 20, vocab_size)

def test_mal_long_sequence():
    dim = 32
    vocab_size = 100
    # Test for potential issues with combined persistent memory
    model = MemoryAsLayer(dim=dim, vocab_size=vocab_size)
    x = torch.randint(0, vocab_size, (1, 50))
    output = model(x)
    assert output.shape == (1, 50, vocab_size)
