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

    def forward_chunked(
        self, 
        x: Tensor, 
        chunk_size: int = 64,
        state: Optional[NeuralMemoryState] = None
    ) -> Tuple[Tensor, NeuralMemoryState]:
        """
        Optimized Chunkwise Forward Pass.
        
        Strategy:
        1. Split sequence into chunks.
        2. Within each chunk, assume memory parameters are constant (Staleness).
        3. Parallelize Retrieval and Gradient Computation using vmap.
        4. Aggregate gradients and perform a single update at the end of the chunk.
        
        Speedup: O(T) -> O(T/C) sequential steps.
        """
        from titans_pytorch.utils import chunk_sequence, unchunk_sequence
        
        batch_size, seq_len, _ = x.shape
        if state is None:
            state = self.get_initial_state(batch_size, x.device)
            
        # 1. Chunk input
        x_chunks = chunk_sequence(x, chunk_size) # (B, NumChunks, ChunkSize, D)
        num_chunks = x_chunks.shape[1]
        
        outputs_list = []
        
        # Define functional closures for vmap
        def compute_chunk_retrieval(p, q_chunk):
            # p: params, q_chunk: (ChunkIdx, Dim) -- wait, vmap over B and ChunkTime?
            # actually we vmap over Batch, and within batch simple matmul if q is (ChunkSize, Dim)
            # but mlp_forward expects (Dim,) or (Batch, Dim).
            # Let's use vmap over time as well inside.
            return vmap(lambda _q: mlp_forward(p, _q))(q_chunk)

        def compute_chunk_grads(p, k_chunk, v_chunk):
             # p: params, k,v: (ChunkSize, Dim)
             # Compute gradients for EACH step in chunk
             def step_grad(ki, vi):
                 return grad(lambda _p, _k, _v: F.mse_loss(mlp_forward(_p, _k), _v))(p, ki, vi)
             return vmap(step_grad)(k_chunk, v_chunk)

        # Main Loop over Chunks (Reduced sequential steps)
        for i in range(num_chunks):
            x_c = x_chunks[:, i, :, :] # (B, C, D)
            curr_chunk_size = x_c.shape[1]
            
            # Precompute Q, K, V for entire chunk
            # x_c is (B, C, D) -> Linear layers apply to last dim, so output (B, C, D)
            q_c = l2_normalize(self.W_q(x_c))
            k_c = l2_normalize(self.W_k(x_c))
            v_c = self.W_v(x_c)
            
            # Gates (B, C, 1)
            alpha_c, eta_c, theta_c = self._compute_gates(x_c)
            
            # --- 1. Parallel Retrieval ---
            # vmap over Batch dimension
            # Input to vmap: (params_batch, q_batch)
            #   params_batch: dict of (B, ...)
            #   q_batch: (B, C, D)
            y_chunk = vmap(compute_chunk_retrieval)(state.params, q_c) # -> (B, C, D)
            outputs_list.append(y_chunk)
            
            # --- 2. Parallel Gradient Computation ---
            # vmap over Batch dimension
            # Returns dict of grads per param: each (B, C, Shape...)
            chunk_grads = vmap(compute_chunk_grads)(state.params, k_c, v_c)
            
            # --- 3. Aggregated Update ---
            # We need to aggregate the gradients and gate values over the chunk.
            # Simple approach: Mean/Sum gradients, Mean gates?
            # Or weighted sum?
            # Titans paper suggests "Surprise S_t" is added. 
            # If we sum the updates: M_new = M_old + Sum(Updates)
            
            new_params = {}
            new_momentum = {}
            
            for name in state.params.keys():
                p = state.params[name] # (B, ...)
                m = state.momentum[name] # (B, ...)
                g_seq = chunk_grads[name] # (B, C, ...)
                
                # Expand dims for broadcasting if needed
                # g_seq is (B, C, P_dim...)
                
                # Aggregate Gates and Grads
                # We want effectively: S_t = eta * S_{t-1} - theta * g_t
                # But parallel approximation: just sum(-theta * g_t) and handle momentum globally?
                # Or simplier: Average Alpha, Average Theta, Average Grad
                
                # Let's implement Average Aggregation for stability
                grad_avg = g_seq.mean(dim=1) # (B, ...)
                theta_avg = theta_c.mean(dim=1).view(batch_size, *([1]*(p.dim()-1))) # (B, 1...)
                alpha_avg = alpha_c.mean(dim=1).view(batch_size, *([1]*(p.dim()-1)))
                eta_avg = eta_c.mean(dim=1).view(batch_size, *([1]*(p.dim()-1)))
                
                # Momentum Update (Chunk level)
                # S_chunk = eta * S_prev - theta * grad_avg
                nm = eta_avg * m - theta_avg * grad_avg
                
                # Param Update
                # M_chunk = (1 - alpha) * M_prev + S_chunk
                np = (1.0 - alpha_avg) * p + nm
                
                new_params[name] = np
                new_momentum[name] = nm
            
            state = NeuralMemoryState(new_params, new_momentum, state.step + curr_chunk_size)
            
        # Reassemble output
        output_full = torch.cat(outputs_list, dim=1)
        
        # Unpad if needed (handled by slicing)
        if seq_len != output_full.shape[1]:
             output_full = output_full[:, :seq_len, :]
             
        return output_full, state
