"""
Patchembedding module
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from typing import Sequence, Type, Union

from monai.utils import ensure_tuple_rep

class PatchEmbed(nn.Module):
    """
    4D Image to Patch Embedding
    """

    def __init__(
        self,
        img_size: Sequence[int],
        patch_size: Sequence[int],
        in_chans: int = 1,
        embed_dim: int = 48,
        norm_layer: Type[LayerNorm] = None,
        flatten: bool = True,
        spatial_dims: int = 4,
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            patch_size: dimension of patch size.
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            norm_layer: normalization layer.
            flatten: whether to flatten the output.
            spatial_dims: spatial dimension.
        """

        super().__init__()
        
        self.img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.grid_size = tuple(i // p for i, p in zip(self.img_size, self.patch_size))
        self.num_patches = np.prod(self.grid_size)
        self.flatten = flatten

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=self.patch_size[:3], stride=self.patch_size[:3])

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # x: B, C, D, H, W, T
        B, C, D, H, W, T = x.shape
        
        # Merge Time into Batch or treat as spatial?
        # SwiFT expects 4D patches but Conv3d is 3D.
        # Typically Swin-4D implementations handle Time as another dimension or 
        # use Conv3d on (D, H, W) and share across T, or use Conv3d with Time depth.
        
        # Based on the provided code usage: 
        # x = self.patch_embed(x) -> (B, EmbedDim, D', H', W', T)
        
        # Let's assume we process each time point independently for embedding spatial features
        # OR use a 3D conv on spatial dims and keep time.
        
        # Reshape: (B*T, C, D, H, W)
        x_flat = x.permute(0, 5, 1, 2, 3, 4).reshape(-1, C, D, H, W)
        
        x_emb = self.proj(x_flat) # (B*T, EmbedDim, D/P, H/P, W/P)
        
        # Reshape back: (B, T, EmbedDim, D', H', W') -> (B, EmbedDim, D', H', W', T)
        _, ED, DP, HP, WP = x_emb.shape
        x_emb = x_emb.view(B, T, ED, DP, HP, WP).permute(0, 2, 3, 4, 5, 1)
        
        if self.flatten:
            x_emb = x_emb.flatten(2).transpose(1, 2)  # B T*D*H*W C
            
        if self.norm is not None:
            if self.flatten:
                x_emb = self.norm(x_emb)
            else:
                # Permute to apply norm on channel dim if needed, usually LayerNorm is last dim
                pass
                
        return x_emb


