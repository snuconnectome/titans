import torch
import torch.nn as nn
from titans_pytorch.neuro.neurostorm import NeuroSTORMMAE as NeuroSTORMOriginal

class NeuroSTORMWrapper(nn.Module):
    """
    Wrapper for NeuroSTORM (BrainT5) to match Titans-Neuro benchmark interface.
    """
    def __init__(self, spatial_shape=(32,32,32), hidden_dim=64):
        super().__init__()
        
        self.model = NeuroSTORMOriginal(
            img_size=(spatial_shape[0], spatial_shape[1], spatial_shape[2], 16),
            in_chans=1,
            embed_dim=hidden_dim,
            window_size=(4, 4, 4, 4), 
            first_window_size=(4, 4, 4, 4),
            patch_size=(4, 4, 4, 1),
            depths=[2, 2],
            num_heads=[4, 8],
            c_multiplier=2,
            spatial_dims=4,
            drop_path_rate=0.1,
        )
        
        # Issue: RuntimeError: mat1 and mat2 shapes cannot be multiplied (1024x64 and 16x64)
        # 1024 is likely B*N (batch size * num_patches). 16x64 is the weight (16=out_features, 64=in_features).
        # Wait, linear layer weight is (out, in). So it expects (..., in).
        # (1024x64) x (16x64).T -> (1024x64) x (64x16) -> (1024x16). This should work if in_features=64.
        
        # Let's inspect `self.model.decoder_pred`.
        # In original code:
        # self.decoder_pred = nn.Linear(embed_dim * 2 ** (len(depths) - 1) // 8, patch_size[0] ** 3 * in_chans, bias=True)
        # embed_dim=64, depths=[2,2] (len=2).
        # embed_dim * 2^(1) // 8 = 64 * 2 // 8 = 16.
        # So in_features is 16.
        
        # But our input `x_dec` coming from `norm_up` has shape with last dim = 64 (embed_dim).
        # norm_up = norm_layer(embed_dim) -> so output is 64 dim.
        
        # NeuroSTORM Logic seems to assume reduction in channel dim or different architecture flow.
        # Let's look at `PatchExpanding`:
        # self.expand = nn.Linear(dim, c_multiplier * c_multiplier * dim, bias=False)
        # rearrange(x, 'B D H W T (P1 P2 P3 C) -> B (D P1) (H P2) (W P3) T C', ...)
        # It expands spatial dims and reduces channel dim by factor.
        
        # The issue is `decoder_pred` input dimension mismatch.
        # We will redefine `decoder_pred` to match our actual feature dimension (64).
        
        # Calculate correct input dim for final projection
        # Based on trace, input is 64.
        current_dim = hidden_dim # 64
        output_dim = 4*4*4*1 # patch_size^3 * in_chans = 64
        
        self.model.decoder_pred = nn.Linear(current_dim, output_dim, bias=True)
        
    def forward(self, x):
        # x: (B, T, C, D, H, W) -> (B, C, D, H, W, T)
        x_in = x.permute(0, 2, 3, 4, 5, 1)
        
        # 1. Patch Embed
        x_emb = self.model.patch_embed(x_in)
        # x_emb = self.model.pos_drop(x_emb)
        
        # 2. Encoder
        for i in range(self.model.num_layers):
            x_emb = self.model.pos_embeds[i](x_emb)
            x_emb = self.model.layers[i](x_emb.contiguous())
            
        # 3. Decoder
        # Reshape for decoder
        x_dec = x_emb.permute(0, 2, 3, 4, 5, 1) # B D H W T C
        x_dec = self.model.first_patch_expanding(x_dec)
        x_dec = x_dec.permute(0, 5, 1, 2, 3, 4) # B C D H W T
        
        for layer in self.model.layers_up:
            x_dec = layer(x_dec)
            
        x_dec = x_dec.permute(0, 2, 3, 4, 5, 1) # B D H W T C
        x_dec = self.model.norm_up(x_dec)
        
        # Linear projection
        x_dec = self.model.decoder_pred(x_dec)
        
        # Reshape to Image
        C = self.model.in_chans
        P1, P2, P3 = self.model.patch_size[:3]
        
        # Shape: B D H W T (P1*P2*P3*C)
        # Target: B C (D*P1) (H*P2) (W*P3) T
        
        x_out = x_dec.permute(0, 5, 1, 2, 3, 4) # B (feat) D H W T
        
        B, Feat, D, H, W, T = x_out.shape
        # Feat = P1*P2*P3*C = 64
        
        x_out = x_out.view(B, C, P1, P2, P3, D, H, W, T)
        # Permute to put spatial dims together
        x_out = x_out.permute(0, 1, 5, 2, 6, 3, 7, 4, 8) 
        # B, C, D, P1, H, P2, W, P3, T
        
        x_out = x_out.reshape(B, C, D*P1, H*P2, W*P3, T)
        
        last_frame = x_out[..., -1] # (B, C, D, H, W)
        out = last_frame.unsqueeze(1) # (B, 1, C, D, H, W)
        
        return out
