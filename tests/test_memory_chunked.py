import torch
import pytest
from titans_pytorch.memory.neural_memory import NeuralMemory

def test_neural_memory_chunked_forward():
    dim = 16
    layer_width = 32
    memory = NeuralMemory(dim=dim, layer_width=layer_width)
    
    batch_size = 2
    seq_len = 20
    # Chunk size = 5, so 4 chunks
    chunk_size = 5
    
    x = torch.randn(batch_size, seq_len, dim)
    
    # Run Chunked
    out_chunked, state_chunked = memory.forward_chunked(x, chunk_size=chunk_size)
    assert out_chunked.shape == (batch_size, seq_len, dim)
    assert state_chunked.step == seq_len

def test_chunked_vs_sequential_stability():
    # While they won't match exactly, the graph should build and run.
    dim = 8
    memory = NeuralMemory(dim=dim)
    x = torch.randn(1, 10, dim)
    
    out_seq, _ = memory(x)
    out_chunk, _ = memory.forward_chunked(x, chunk_size=5)
    
    assert out_seq.shape == out_chunk.shape
