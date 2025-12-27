import torch
import pytest
from titans_pytorch.memory.neural_memory import NeuralMemory

def test_inference_ablation():
    """Verify that enable_update=False prevents any parameter changes."""
    dim = 8
    memory = NeuralMemory(dim)
    batch_size = 1
    state = memory.get_initial_state(batch_size)
    x = torch.randn(batch_size, dim)
    
    # Run with update DISABLED
    _, state_no_update = memory.forward_step(x, state, enable_update=False)
    
    # Check strict equality (no change)
    for p_old, p_new in zip(state, state_no_update):
        assert torch.equal(p_old, p_new), "State should not change when enable_update=False"

    # Run with update ENABLED
    _, state_update = memory.forward_step(x, state, enable_update=True)
    
    # Check change
    has_change = False
    for p_old, p_new in zip(state, state_update):
        if not torch.allclose(p_old, p_new):
            has_change = True
            break
    assert has_change, "State SHOULD change when enable_update=True"

def test_memory_probe_concept():
    """
    Simulate a simplified 'probe' test.
    We check if the memory can 'hold' a value after an update.
    """
    dim = 8
    width = 16
    memory = NeuralMemory(dim, layer_width=width, learning_rate=0.1)
    batch_size = 1
    
    # Pattern to memorize: k -> v
    k_target = torch.randn(batch_size, dim)
    v_target = torch.randn(batch_size, dim)
    
    # We cheat and use k_target as input x, assuming W_k/W_v map it consistently
    # Ideally we'd set W_k=I, W_v=I for this specific test, but let's test end-to-end
    x = torch.randn(batch_size, dim)
    
    state = memory.get_initial_state(batch_size)
    
    # 1. Measure initial error
    # We need to project x to k, v
    k_proj = memory.W_k(x)
    v_proj = memory.W_v(x)
    pred_init = memory.functional_memory_forward(state, k_proj)
    err_init = (pred_init - v_proj).abs().mean()
    
    # 2. Update memory on x
    _, state_post = memory.forward_step(x, state)
    
    # 3. Measure error after update
    pred_post = memory.functional_memory_forward(state_post, k_proj)
    err_post = (pred_post - v_proj).abs().mean()
    
    assert err_post < err_init, f"Memory should reduce error on the memorized pattern. Init: {err_init}, Post: {err_post}"
