import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, Any
from torch.func import vmap, grad, functional_call

from titans_pytorch.utils import exists, default, l2_normalize
from titans_pytorch.memory.functional import mlp_forward

class NeuralMemoryState:
    def __init__(self, params: Dict[str, Tensor], momentum: Dict[str, Tensor], step: int = 0):
        self.params = params
        self.momentum = momentum
        self.step = step

    def detach(self):
        return NeuralMemoryState(
            params={k: v.detach() for k, v in self.params.items()},
            momentum={k: v.detach() for k, v in self.momentum.items()},
            step=self.step
        )

class NeuralMemory(nn.Module):
    def __init__(
        self, 
        dim: int, 
        layer_width: int = 128, 
        learning_rate: float = 0.01, 
        momentum: float = 0.9, 
        decay: float = 0.0,
        surprise_tau: float = 0.1,
        surprise_scale: float = 0.1,
        learnable_gates: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.layer_width = layer_width
        self.base_lr = learning_rate
        self.base_momentum = momentum
        self.base_decay = decay
        self.learnable_gates = learnable_gates
        
        # Surprise Gating Parameters
        self.surprise_tau = surprise_tau
        self.surprise_scale = surprise_scale
        
        # Projections
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_q = nn.Linear(dim, dim, bias=False)
        
        # Memory Network Initial Parameters
        self.init_params = nn.ParameterDict({
            'init_w1': nn.Parameter(torch.randn(layer_width, dim) * 0.02),
            'init_b1': nn.Parameter(torch.zeros(layer_width)),
            'init_w2': nn.Parameter(torch.randn(dim, layer_width) * 0.02),
            'init_b2': nn.Parameter(torch.zeros(dim))
        })

        if learnable_gates:
            self.alpha_proj = nn.Linear(dim, 1) # Forgetting
            self.eta_proj = nn.Linear(dim, 1)   # Momentum
            self.theta_proj = nn.Linear(dim, 1) # Learning Rate
            
            # Init biases to match base values approximately
            nn.init.constant_(self.alpha_proj.bias, -5.0) # Start with low forgetting
            nn.init.constant_(self.eta_proj.bias, 2.0)    # Start with high momentum (sigmoid(2) ~ 0.88)
            nn.init.constant_(self.theta_proj.bias, -4.6) # Start with 0.01 (sigmoid(-4.6) ~ 0.01)

    def get_initial_state(self, batch_size: int, device: torch.device) -> NeuralMemoryState:
        # Per-sample parameters
        params = {
            k: v.unsqueeze(0).expand(batch_size, -1, -1) if v.dim() == 2 else v.unsqueeze(0).expand(batch_size, -1)
            for k, v in self.init_params.items()
        }
        momentum = {k: torch.zeros_like(v) for k, v in params.items()}
        return NeuralMemoryState(params, momentum)

    def _compute_gates(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.learnable_gates:
            alpha = torch.sigmoid(self.alpha_proj(x))
            eta = torch.sigmoid(self.eta_proj(x))
            theta = torch.sigmoid(self.theta_proj(x))
        else:
            alpha = torch.full((x.shape[0], 1), self.base_decay, device=x.device)
            eta = torch.full((x.shape[0], 1), self.base_momentum, device=x.device)
            theta = torch.full((x.shape[0], 1), self.base_lr, device=x.device)
        return alpha, eta, theta

    def forward_step(
        self, 
        x_t: Tensor, 
        state: NeuralMemoryState, 
        enable_update: bool = True,
        stop_grad_k: bool = False,
        stop_grad_v: bool = False
    ) -> Tuple[Tensor, NeuralMemoryState]:
        
        batch_size = x_t.shape[0]
        q = l2_normalize(self.W_q(x_t))
        k = l2_normalize(self.W_k(x_t))
        v = self.W_v(x_t)
        
        alpha, eta, theta = self._compute_gates(x_t)
        
        # 1. Retrieval (Batched functional call)
        # We need to vmap the functional_call over the batch dimension of params and q
        def single_retrieve(p, query):
            return mlp_forward(p, query)
        
        y_pred = vmap(single_retrieve)(state.params, q)
        
        # 2. Update
        if enable_update:
            k_in = k.detach() if stop_grad_k else k
            v_in = v.detach() if stop_grad_v else v
            
            def compute_single_loss(p, ki, vi):
                vp = mlp_forward(p, ki)
                return F.mse_loss(vp, vi)
            
            # vmap the gradient computation
            def single_grad(p, ki, vi):
                return grad(compute_single_loss)(p, ki, vi)
            
            grads = vmap(single_grad)(state.params, k_in, v_in)
            
            new_params = {}
            new_momentum = {}
            
            for name in state.params.keys():
                p = state.params[name]
                m = state.momentum[name]
                g = grads[name]
                
                # S_t = eta * S_{t-1} - theta * grad
                nm = eta.view(batch_size, 1, 1) * m - theta.view(batch_size, 1, 1) * g if m.dim() == 3 else eta * m - theta * g
                
                # M_t = (1 - alpha) * M_{t-1} + S_t
                np = (1.0 - alpha.view(batch_size, 1, 1)) * p + nm if p.dim() == 3 else (1.0 - alpha) * p + nm
                
                new_params[name] = np
                new_momentum[name] = nm
            
            new_state = NeuralMemoryState(new_params, new_momentum, state.step + 1)
        else:
            new_state = state

        return y_pred, new_state

    def forward(self, x: Tensor, state: Optional[NeuralMemoryState] = None) -> Tuple[Tensor, NeuralMemoryState]:
        batch_size, seq_len, _ = x.shape
        if state is None:
            state = self.get_initial_state(batch_size, x.device)
        
        outputs = []
        for t in range(seq_len):
            y_t, state = self.forward_step(x[:, t, :], state)
            outputs.append(y_t)
            
        return torch.stack(outputs, dim=1), state
