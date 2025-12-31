
import torch.nn as nn
from titans_pytorch.neuro.models import TitanNeuro, Simple3DEncoder, Simple3DDecoder
from titans_pytorch.neuro.models_swift import SwiFTWrapper
from titans_pytorch.neuro.models_neurostorm import NeuroSTORMWrapper
from titans_pytorch.neuro.baselines.mamba2 import Mamba2
from titans_pytorch.neuro.baselines.gla import GatedLinearAttention
from titans_pytorch.neuro.baselines.griffin import Griffin

class TitanBaseline(nn.Module):
    """
    Generic wrapper for sequence models (Mamba, GLA, Griffin) to be compatible with TitanNeuro 
    encoder/decoder structure for fMRI tasks.
    """
    def __init__(self, mixer_cls, mixer_kwargs={}, spatial_shape=(32,32,32), hidden_dim=128):
        super().__init__()
        self.encoder = Simple3DEncoder(channels=1, hidden_dim=hidden_dim, input_spatial_shape=spatial_shape)
        self.memory = mixer_cls(d_model=hidden_dim, **mixer_kwargs)
        self.decoder = Simple3DDecoder(
            hidden_dim=hidden_dim, 
            channels=1, 
            output_shape=spatial_shape,
            encoder_flat_size=self.encoder.flat_size,
            encoder_final_shape=self.encoder.final_shape
        )

    def forward(self, x, use_chunked=False, chunk_size=64):
        # x: (B, T, C, D, H, W)
        B, T, C, D, H, W = x.shape
        x_flat = x.view(B*T, C, D, H, W)
        embeddings = self.encoder(x_flat)
        embeddings = embeddings.view(B, T, -1)
        
        # Mixer (Sequence Model)
        # Assumes mixer takes (B, L, D) and returns (B, L, D) or (B, L, D), state
        memory_out = self.memory(embeddings)
        
        # If tuple returned (like mamba returns y, h), take first element
        if isinstance(memory_out, tuple):
            memory_out = memory_out[0]
            
        memory_out_flat = memory_out.view(B*T, -1)
        pred_flat = self.decoder(memory_out_flat)
        pred = pred_flat.view(B, T, C, D, H, W)
        return pred

def get_model(model_name, spatial_shape=(32,32,32), hidden_dim=64):
    if model_name == 'swift':
        return SwiFTWrapper(spatial_shape=spatial_shape, hidden_dim=hidden_dim)
    elif model_name == 'neurostorm':
        return NeuroSTORMWrapper(spatial_shape=spatial_shape, hidden_dim=hidden_dim)
    elif model_name == 'mamba2':
        return TitanBaseline(Mamba2, mixer_kwargs={'d_state': 16, 'd_conv': 4, 'expand': 2}, spatial_shape=spatial_shape, hidden_dim=hidden_dim)
    elif model_name == 'gla':
        return TitanBaseline(GatedLinearAttention, mixer_kwargs={'n_head': 4}, spatial_shape=spatial_shape, hidden_dim=hidden_dim)
    elif model_name == 'griffin':
        return TitanBaseline(Griffin, mixer_kwargs={'depth': 2}, spatial_shape=spatial_shape, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")
