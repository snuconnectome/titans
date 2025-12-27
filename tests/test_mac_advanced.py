import torch
from titans_pytorch.models.mac import MemoryAsContext

def test_mac_shared_projection_init():
    dim = 32
    model = MemoryAsContext(dim=dim, vocab_size=100, shared_kv_proj=True)
    
    # Check if weights are initialized same as attention
    qkv = model.attention.in_proj_weight
    k_att = qkv[dim:2*dim, :]
    
    # Memory K projection
    k_mem = model.memory.W_k.weight
    
    # Assert they are equal immediately after init
    assert torch.allclose(k_att, k_mem), "Memory K projection should be initialized from Attention K projection"
