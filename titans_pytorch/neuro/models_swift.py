import torch
import torch.nn as nn
from titans_pytorch.neuro.swift import SwiFT as SwiFTOriginal

class SwiFTWrapper(nn.Module):
    """
    Wrapper for SwiFT to match Titans-Neuro benchmark interface.
    """
    def __init__(self, spatial_shape=(32,32,32), hidden_dim=64):
        super().__init__()
        
        # Configure SwiFT to match input shape
        # Input: (B, T, C, D, H, W)
        # SwiFT expects: (B, C, D, H, W, T)
        
        self.model = SwiFTOriginal(
            img_size=(spatial_shape[0], spatial_shape[1], spatial_shape[2], 16), # T=16 fixed window
            in_chans=1,
            embed_dim=hidden_dim,
            window_size=(4, 4, 4, 4), # Adjusted for small input
            first_window_size=(4, 4, 4, 4),
            patch_size=(4, 4, 4, 1), # Keep time resolution?
            depths=[2, 2],
            num_heads=[4, 8],
            c_multiplier=2,
            spatial_dims=4
        )
        
        # SwiFT output needs to be projected back to image space
        # SwiFT reduces spatial dims by factor of patch_size * 2^(num_layers-1)?
        # Let's check output shape.
        # With embed_dim=64, c_multiplier=2, layers=2
        # Layer 0: dim=64
        # Layer 1: dim=128
        # Output dim = 128 * 2 (c_multiplier from last layer transition?) -> No, typically last layer dim
        
        # Correction based on error:
        # "weight of size [256, 128, 4, 4, 4], expected input to have 256 channels, but got 128 channels"
        # ConvTranspose3d(in_channels, out_channels, ...)
        # The weight shape is [in_channels, out_channels, k, k, k] for ConvTranspose3d?
        # Actually ConvTranspose3d weight is [in_channels, out_channels/groups, k, k, k]
        # Wait, weight shape [256, 128, ...] means in=256, out=128.
        # But error says "expected input to have 256 channels, but got 128 channels".
        # So input has 128 channels.
        # So in_channels of first ConvTranspose3d should be 128.
        
        # SwiFT logic:
        # embed_dim=64
        # layer 0: 64
        # layer 1: 128 (embed_dim * c_multiplier^1)
        # So output has 128 channels.
        
        final_dim = hidden_dim * 2 # 128
        
        self.decoder_head = nn.Sequential(
            # Input: (B, 128, D/?, H/?, W/?)
            nn.ConvTranspose3d(final_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(hidden_dim // 2, 1, kernel_size=4, stride=2, padding=1)
        )
        
    def forward(self, x):
        # x: (B, T, C, D, H, W) -> (B, C, D, H, W, T)
        x_in = x.permute(0, 2, 3, 4, 5, 1)
        
        # SwiFT Forward
        # Returns: (B, C_out, D', H', W', T')
        features = self.model(x_in)
        
        # Take the last time point feature to predict next step
        # features shape: (B, C_last, D_last, H_last, W_last, T_last)
        last_feat = features[..., -1] # (B, C, D, H, W)
        
        # Decode
        out = self.decoder_head(last_feat) # (B, 1, D, H, W)
        
        # Add Time dim to match (B, T_out=1, C, D, H, W)
        out = out.unsqueeze(1) 
        
        return out
