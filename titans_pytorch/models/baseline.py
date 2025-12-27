import torch
import torch.nn as nn

class StandardTransformer(nn.Module):
    def __init__(self, dim, vocab_size, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(dim, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        # Create causal mask
        N = x.shape[1]
        mask = torch.triu(torch.ones(N, N, device=x.device) * float('-inf'), diagonal=1)
        
        out = self.transformer(x, mask=mask, is_causal=True)
        return self.out_proj(out)
