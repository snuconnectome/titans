import torch
import pytest
from titans_pytorch.memory.neural_memory import NeuralMemory

def test_dynamic_theta():
    """Verify that passing a larger theta results in larger parameter updates."""
    dim = 8
    memory = NeuralMemory(dim)
    batch_size = 1
    
    x = torch.randn(batch_size, dim)
    state = memory.get_initial_state(batch_size)
    
    # Run with small theta
    _, state_small = memory.forward_step(x, state, theta=0.001)
    
    # Run with large theta
    _, state_large = memory.forward_step(x, state, theta=1.0)
    
    # Compare change magnitude
    # We check the first weight: state[0]
    initial_w = state[0]
    
    diff_small = (state_small[0] - initial_w).abs().mean()
    diff_large = (state_large[0] - initial_w).abs().mean()
    
    assert diff_large > diff_small * 10, "Larger theta should produce significantly larger updates"

def test_surprise_gating():
    """Verify that gating scales the effective learning rate."""
    dim = 8
    # Initialize with specific surprise params
    memory = NeuralMemory(dim, surprise_tau=0.1, surprise_scale=0.1)
    batch_size = 1
    
    x = torch.randn(batch_size, dim)
    state = memory.get_initial_state(batch_size)
    
    # 1. Run without gating
    _, state_no_gate = memory.forward_step(x, state, theta=0.1, enable_gating=False)
    diff_no_gate = (state_no_gate[0] - state[0]).abs().mean()
    
    # 2. Run with gating
    # Depending on the loss and tau, gate will be < 1.0 or close to 0 or 1.
    # We just want to ensure it runs and modifies the update.
    _, state_gate = memory.forward_step(x, state, theta=0.1, enable_gating=True)
    diff_gate = (state_gate[0] - state[0]).abs().mean()
    
    # The update should likely be different
    assert not torch.isclose(diff_no_gate, diff_gate), "Gating should modify the update magnitude"
