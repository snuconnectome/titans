import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class RG_LRU(nn.Module):
    """
    Recurrent Gated Linear Recurrent Unit (Simplified).
    Core component of Griffin logic: 1D recurrence.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.input_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim) # Recurrence gate
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, T, D)
        u = self.input_proj(x)
        g = torch.sigmoid(self.gate_proj(x)) # Decay gate
        
        # Simple element-wise recurrence
        # h_t = g_t * h_{t-1} + (1 - g_t) * u_t
        
        h_state = torch.zeros_like(u[:, 0, :])
        h_list = []
        
        for t in range(u.shape[1]):
            g_t = g[:, t, :]
            u_t = u[:, t, :]
            h_state = g_t * h_state + (1 - g_t) * u_t
            h_list.append(h_state)
            
        h = torch.stack(h_list, dim=1)
        return self.out_proj(h)

class LocalAttention(nn.Module):
    """
    Local Sliding Window Attention.
    """
    def __init__(self, dim, window_size=4, n_head=4):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.n_head = n_head
        self.head_dim = dim // n_head
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b t (c h d) -> c b h t d', c=3, h=self.n_head)
        
        # Sliding Window Mask
        # We can implement this naively by masking the attention scores
        attn_scores = torch.einsum('bhid, bhjd -> bhij', q, k) / (self.head_dim ** 0.5)
        
        # Causal mask AND Window mask
        # i >= j (causal) AND i - j < window_size
        mask = torch.ones(T, T, device=x.device)
        mask = torch.tril(mask, diagonal=0)
        mask = torch.triu(mask, diagonal=-(self.window_size - 1))
        
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        y = torch.einsum('bhij, bhjd -> bhid', attn_probs, v)
        y = rearrange(y, 'b h t d -> b t (h d)')
        return self.out_proj(y)

class Griffin(nn.Module):
    """
    Minimal Griffin block (Hybrid).
    Reference: https://arxiv.org/abs/2402.19427
    Combines Recurrent Block (RG-LRU) and Local Attention.
    """
    def __init__(self, d_model=128, depth=2, window_size=4):
        super().__init__()
        dim = d_model
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # Alternating or stacked? Paper uses a mix. 
            # We'll simple stack Recurrent -> Local Attn for the block.
            self.layers.append(nn.ModuleDict({
                'recurrence': RG_LRU(dim),
                'norm1': nn.LayerNorm(dim),
                'local_attn': LocalAttention(dim, window_size=window_size),
                'norm2': nn.LayerNorm(dim),
                'mlp': nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim)),
                'norm3': nn.LayerNorm(dim)
            }))

    def forward(self, x):
        for layer in self.layers:
            # 1. Recurrent Block Residual
            res = x
            x = layer['recurrence'](layer['norm1'](x)) + res
            
            # 2. Local Attention Residual
            res = x
            x = layer['local_attn'](layer['norm2'](x)) + res
            
            # 3. MLP Residual
            res = x
            x = layer['mlp'](layer['norm3'](x)) + res
            
        return x
