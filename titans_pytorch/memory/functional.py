import torch
import torch.nn.functional as F
from torch.func import grad, functional_call, vmap

def mlp_forward(params, x):
    """
    Pure functional MLP forward pass.
    params: dict of tensors (w1, b1, w2, b2, ...)
    x: (Dim,) or (Batch, Dim)
    """
    # Assuming params keys follow standard nn.Sequential/nn.Module naming
    # or a specific custom naming convention.
    
    # Simple 2-layer MLP implementation for Neural Memory
    # This reflects the structure in the previous implementations
    w1, b1 = params['init_w1'], params['init_b1']
    w2, b2 = params['init_w2'], params['init_b2']
    
    h = F.linear(x, w1, b1)
    h = F.relu(h)
    return F.linear(h, w2, b2)

def compute_loss(params, k, v):
    """
    associative memory loss: ||M(k) - v||^2
    """
    v_pred = mlp_forward(params, k)
    return F.mse_loss(v_pred, v)

def update_step_functional(params, momentum_buffer, k, v, lr, momentum_factor, decay):
    """
    Performs a single functional update step.
    Returns: (new_params, new_momentum_buffer, loss, grads)
    """
    loss = compute_loss(params, k, v)
    grads = grad(compute_loss)(params, k, v)
    
    new_params = {}
    new_momentum = {}
    
    for name, p in params.items():
        g = grads[name]
        m = momentum_buffer[name]
        
        # S_t = eta * S_{t-1} - theta * grad
        nm = momentum_factor * m - lr * g
        
        # M_t = (1 - alpha) * M_{t-1} + S_t
        np = (1.0 - decay) * p + nm
        
        new_params[name] = np
        new_momentum[name] = nm
        
    return new_params, new_momentum, loss, grads
