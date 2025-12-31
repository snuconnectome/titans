import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class GatedLinearAttention(nn.Module):
    """
    Minimal Gated Linear Attention (GLA) implementation.
    Reference: https://arxiv.org/abs/2312.06635
    
    Structure:
    - Data-dependent gates for decay.
    - Linear Attention with recurrent state.
    """
    def __init__(self, d_model=128, n_head=4, gate_nonlinearity='sigmoid'):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.g_proj = nn.Linear(d_model, d_model, bias=False) # Decay gate
        
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.grp_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (Batch, Time, Dim)
        """
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_head)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_head)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_head)
        g = rearrange(self.g_proj(x), 'b t (h d) -> b h t d', h=self.n_head)
        
        # Data-dependent decay
        # alpha_t = sigmoid(g_t)
        alpha = torch.sigmoid(g)
        
        # Parallel Linear Attention with Decay (Simplified recurrence)
        # H_t = alpha_t * H_{t-1} + K_t^T V_t
        # Y_t = Q_t H_t
        # Ideally, we implement this with a parallel scan (associative scan).
        # For prototype/benchmark with short seq_len, sequential loop is fine.
        # But let's try a simple cumulative interaction approach for "Linear" property.
        
        # Sequential Recurrence for correctness (Linear Scan)
        h_state = torch.zeros(B, self.n_head, self.head_dim, self.head_dim, device=x.device)
        y_list = []
        
        for t in range(T):
            # Update state with decay
            # a_t = alpha[:, :, t].unsqueeze(-1).unsqueeze(-1) # Incorrect
            a_t = alpha[:, :, t].unsqueeze(-1) # (B, H, D, 1)
            k_t = k[:, :, t].unsqueeze(-1) # (B, H, D, 1)
            v_t = v[:, :, t].unsqueeze(-2) # (B, H, 1, D)
            
            kv = torch.matmul(k_t, v_t) # (B, H, D, D) outer product
            
            h_state = a_t * h_state + kv
            
            # Read
            # y_t = q_t @ h_t
            q_t = q[:, :, t].unsqueeze(-2) # (B, H, 1, D)
            y_t = torch.matmul(q_t, h_state).squeeze(-2) # (B, H, D)
            y_list.append(y_t)
            
        y = torch.stack(y_list, dim=2) # (B, H, T, D)
        y = rearrange(y, 'b h t d -> b t (h d)')
        
        return self.out_proj(self.grp_norm(y))
