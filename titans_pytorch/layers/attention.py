import torch
import torch.nn as nn

class SlidingWindowAttention(nn.Module):
    """
    Standard Multihead Attention but with a sliding window causal mask.
    """
    def __init__(self, dim, num_heads=4, window_size=128):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
    def forward(self, x):
        """
        x: (Batch, SeqLen, Dim)
        """
        B, N, D = x.shape
        
        # Create Sliding Window Mask
        # A position i can attend to j if i - window_size <= j <= i
        # Torch implementation of attn_mask:
        # 2D mask (N, N) where mask[i, j] = True (keep) or False (discard/inf)
        # PyTorch standard: True means ignore (bool) or add float(-inf)
        # But wait, is_causal=True handles upper triangle.
        # We need to additionally mask out the "too far past".
        
        # We construct a float mask
        mask = torch.triu(torch.ones(N, N, device=x.device) * float('-inf'), diagonal=1) # Upper triangle for causal
        
        # Lower triangle filtering for sliding window
        # We want to MASK OUT if j < i - window_size
        # So we want to keep if i - window_size <= j
        # indices: row i, col j
        # condition for -inf: i - j > window_size
        
        indices = torch.arange(N, device=x.device)
        rows = indices.unsqueeze(1)
        cols = indices.unsqueeze(0)
        
        # Mask where distance is too large
        too_far_mask = (rows - cols) > self.window_size
        mask[too_far_mask] = float('-inf')
        
        output, _ = self.attn(x, x, x, attn_mask=mask, is_causal=False) # We handled causal manually in the mask
        return output
