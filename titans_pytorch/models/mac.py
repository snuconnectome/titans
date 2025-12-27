import torch
import torch.nn as nn
from titans_pytorch.memory.neural_memory import NeuralMemory, NeuralMemoryState
from titans_pytorch.utils import l2_normalize
from torch.func import vmap

class MemoryAsContext(nn.Module):
    def __init__(self, dim, vocab_size, num_heads=4, segment_len=128, num_persistent_tokens=16, memory_width=128,
                 shared_kv_proj=False):
        super().__init__()
        self.dim = dim
        self.segment_len = segment_len
        self.num_persistent_tokens = num_persistent_tokens
        
        # Token Embedding
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Persistent Memory (Learnable Parameters)
        self.persistent_memory = nn.Parameter(torch.randn(num_persistent_tokens, dim) * 0.02)
        
        # Long-term Neural Memory
        self.memory = NeuralMemory(dim, layer_width=memory_width)
        
        # Short-term Memory (Attention)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        # WK/WV Shared Projection Logic (Option A1)
        if shared_kv_proj:
            if hasattr(self.attention, 'in_proj_weight') and self.attention.in_proj_weight is not None:
                qkv_weight = self.attention.in_proj_weight
                k_weight = qkv_weight[dim:2*dim, :]
                v_weight = qkv_weight[2*dim:3*dim, :]
                with torch.no_grad():
                    self.memory.W_k.weight.copy_(k_weight)
                    self.memory.W_v.weight.copy_(v_weight)

        self.stop_grad_k = True 
        self.stop_grad_v = False
            
        # Output projection
        self.out_proj = nn.Linear(dim, vocab_size)

    def forward(self, x):
        B, N = x.shape
        x_emb = self.embedding(x)
        
        # Initialize Memory State
        memory_state = self.memory.get_initial_state(B, x.device)
        
        # Segment the input
        pad_len = (self.segment_len - (N % self.segment_len)) % self.segment_len
        if pad_len > 0:
            padding = torch.zeros(B, pad_len, self.dim, device=x.device)
            x_padded = torch.cat([x_emb, padding], dim=1)
        else:
            x_padded = x_emb
            
        num_segments = x_padded.shape[1] // self.segment_len
        final_outputs = []
        
        for i in range(num_segments):
            start_idx = i * self.segment_len
            end_idx = start_idx + self.segment_len
            segment = x_padded[:, start_idx:end_idx, :] # (B, S, D)
            
            # 1. Retrieve History from Memory (using current state)
            # Parallel retrieval for the whole segment
            q_segment = l2_normalize(self.memory.W_q(segment))
            
            def single_retrieve(p, query):
                from titans_pytorch.memory.functional import mlp_forward
                return mlp_forward(p, query)
            
            h_t = vmap(single_retrieve)(memory_state.params, q_segment)
            
            # 2. Construct Attention Input
            P_batch = self.persistent_memory.unsqueeze(0).expand(B, -1, -1)
            combined_input = torch.cat([P_batch, h_t, segment], dim=1)
            
            # 3. Attention
            total_attn_len = combined_input.shape[1]
            attn_mask = torch.triu(torch.ones(total_attn_len, total_attn_len), diagonal=1).bool().to(x.device)
            
            attn_output, _ = self.attention(combined_input, combined_input, combined_input, attn_mask=attn_mask, is_causal=True)
            segment_output = attn_output[:, -self.segment_len:, :]
            
            # 4. Update Memory using segment tokens
            for t in range(self.segment_len):
                token_to_memorize = segment_output[:, t, :]
                _, memory_state = self.memory.forward_step(
                    token_to_memorize, 
                    memory_state, 
                    stop_grad_k=self.stop_grad_k, 
                    stop_grad_v=self.stop_grad_v
                )
            
            # 5. Final Gating (Eq 25)
            q_out = l2_normalize(self.memory.W_q(segment_output))
            mem_out = vmap(single_retrieve)(memory_state.params, q_out)
            
            segment_final = segment_output * mem_out # Element-wise product
            final_outputs.append(segment_final)

        full_output = torch.cat(final_outputs, dim=1)
        full_output = full_output[:, :N, :]
        logits = self.out_proj(full_output)
        
        return logits
