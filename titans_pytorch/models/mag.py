import torch
import torch.nn as nn
from titans_pytorch.memory.neural_memory import NeuralMemory
from titans_pytorch.layers.attention import SlidingWindowAttention

class MemoryAsGate(nn.Module):
    def __init__(self, dim, vocab_size, num_heads=4, window_size=128, num_persistent_tokens=16, memory_width=128):
        super().__init__()
        self.dim = dim
        self.num_persistent_tokens = num_persistent_tokens
        
        self.embedding = nn.Embedding(vocab_size, dim)
        self.persistent_memory = nn.Parameter(torch.randn(num_persistent_tokens, dim) * 0.02)
        
        self.memory = NeuralMemory(dim, layer_width=memory_width)
        self.swa = SlidingWindowAttention(dim, num_heads=num_heads, window_size=window_size)
        
        self.out_proj = nn.Linear(dim, vocab_size)
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_mem = nn.LayerNorm(dim)

    def forward(self, x):
        B, N = x.shape
        x_emb = self.embedding(x)
        
        # Branch 1: Sliding Window Attention
        P_batch = self.persistent_memory.unsqueeze(0).expand(B, -1, -1)
        swa_input = torch.cat([P_batch, x_emb], dim=1)
        swa_out_full = self.swa(swa_input)
        y = swa_out_full[:, -N:, :]
        
        # Branch 2: Neural Memory (Unpack output and state)
        mem_out, _ = self.memory(x_emb)
        
        y_norm = self.norm_attn(y)
        m_norm = self.norm_mem(mem_out)
        
        combined = y_norm * m_norm
        logits = self.out_proj(combined)
        return logits

