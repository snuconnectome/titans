import torch
import pytest
from titans_pytorch.memory.neural_memory import NeuralMemory
from titans_pytorch.memory.functional import mlp_forward
from torch.func import vmap

def test_neural_memory_initialization():
    dim = 64
    layer_width = 128
    memory = NeuralMemory(dim, layer_width=layer_width)
    assert memory.dim == dim
    assert memory.layer_width == layer_width

def test_neural_memory_forward_shape():
    batch_size = 2
    seq_len = 10
    dim = 32
    memory = NeuralMemory(dim)
    
    x = torch.randn(batch_size, seq_len, dim)
    output, state = memory(x)
    
    assert output.shape == (batch_size, seq_len, dim)

def test_neural_memory_has_parameters():
    dim = 32
    memory = NeuralMemory(dim)
    params = list(memory.parameters())
    assert len(params) > 0

def test_memory_update_mechanism():
    dim = 8
    width = 16
    memory = NeuralMemory(dim, layer_width=width)
    
    batch_size = 1
    seq_len = 5
    x = torch.randn(batch_size, seq_len, dim)
    
    state = memory.get_initial_state(batch_size, device=x.device)
    x_t = x[:, 0, :] 
    output, next_state = memory.forward_step(x_t, state)
    
    k_t = torch.nn.functional.normalize(memory.W_k(x_t), p=2, dim=-1)
    v_t = memory.W_v(x_t)
    
    def compute_loss(st, k, v):
        def single_loss(p, ki, vi):
            vp = mlp_forward(p, ki)
            return torch.nn.functional.mse_loss(vp, vi)
        return vmap(single_loss)(st.params, k, v).mean()
        
    loss_initial = compute_loss(state, k_t, v_t)
    loss_next = compute_loss(next_state, k_t, v_t)
    
    # The loss should decrease after an update step on the same datum
    assert loss_next < loss_initial
