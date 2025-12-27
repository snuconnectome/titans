import torch
import pytest
from titans_pytorch.models.mac import MemoryAsContext
from titans_pytorch.memory.neural_memory import NeuralMemory

def test_mac_initialization():
    dim = 32
    vocab_size = 100
    model = MemoryAsContext(dim=dim, vocab_size=vocab_size, num_heads=4)
    assert isinstance(model.memory, NeuralMemory)

def test_mac_segment_handling():
    # MAC processes data in segments.
    # We test if it correctly handles a sequence longer than one segment.
    dim = 32
    vocab_size = 100
    segment_len = 10
    total_len = 25 # 2 full segments + 1 partial
    
    model = MemoryAsContext(dim=dim, vocab_size=vocab_size, segment_len=segment_len)
    
    # Fake embedding input for simplicity or real integer input
    x = torch.randint(0, vocab_size, (1, total_len))
    
    output = model(x)
    
    # Output should match input length
    assert output.shape == (1, total_len, vocab_size) # Assuming projection to vocab
