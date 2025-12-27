import torch
import torch.nn as nn
from titans_pytorch.memory.neural_memory import NeuralMemory
from titans_pytorch.layers.attention import SlidingWindowAttention

class MemoryAsLayer(nn.Module):
    def __init__(self, dim, vocab_size, num_heads=4, window_size=128, num_persistent_tokens=16, memory_width=128):
        super().__init__()
        self.dim = dim
        self.num_persistent_tokens = num_persistent_tokens
        
        self.embedding = nn.Embedding(vocab_size, dim)
        self.persistent_memory = nn.Parameter(torch.randn(num_persistent_tokens, dim) * 0.02)
        
        self.memory = NeuralMemory(dim, layer_width=memory_width)
        self.swa = SlidingWindowAttention(dim, num_heads=num_heads, window_size=window_size)
        
        self.out_proj = nn.Linear(dim, vocab_size)
        self.norm_mem = nn.LayerNorm(dim)
        self.norm_attn = nn.LayerNorm(dim)

    def forward(self, x):
        B, N = x.shape
        x_emb = self.embedding(x)
        
        P_batch = self.persistent_memory.unsqueeze(0).expand(B, -1, -1)
        x_seq = torch.cat([P_batch, x_emb], dim=1) 
        
        mem_out, _ = self.memory(x_seq)
        mem_out = self.norm_mem(mem_out)
        
        swa_out = self.swa(mem_out)
        swa_out = self.norm_attn(swa_out)
        
        final_seq = swa_out[:, -N:, :]
        logits = self.out_proj(final_seq)
        return logits
