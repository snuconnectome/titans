import torch
import pytest
from titans_pytorch.models.mag import MemoryAsGate
from titans_pytorch.memory.neural_memory import NeuralMemory

def test_mag_initialization():
    dim = 32
    vocab_size = 100
    model = MemoryAsGate(dim=dim, vocab_size=vocab_size, num_heads=4)
    assert isinstance(model.memory, NeuralMemory)

def test_mag_forward_shape():
    dim = 32
    vocab_size = 100
    seq_len = 50
    model = MemoryAsGate(dim=dim, vocab_size=vocab_size)
    
    x = torch.randint(0, vocab_size, (1, seq_len))
    output = model(x)
    
    # Output should correspond to input length (despite sliding window)
    assert output.shape == (1, seq_len, vocab_size)

def test_mag_sliding_window_logic():
    # Functional test to ensure it runs without error for various lengths
    dim = 32
    vocab_size = 100
    model = MemoryAsGate(dim=dim, vocab_size=vocab_size, window_size=10)
    
    # Sequence longer than window
    x = torch.randint(0, vocab_size, (1, 25))
    output = model(x)
    assert output.shape == (1, 25, vocab_size)
