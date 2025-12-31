"""
mamba2-minimal
==============

A minimal, single-file implementation of the Mamba-2 model in PyTorch.
Adapted from: https://github.com/tommyip/mamba2-minimal
"""

import math
import json
from dataclasses import dataclass
from typing import Iterable, NamedTuple, TypeAlias, cast, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import LongTensor, Tensor, nn

Device: TypeAlias = Union[str, torch.device, None]

@dataclass
class Mamba2Config:
    d_model: int  # model dimension (D)
    n_layer: int = 1  # number of Mamba-2 layers
    d_state: int = 64  # state dimension (N)
    d_conv: int = 4  # convolution kernel size
    expand: int = 2  # expansion factor (E)
    headdim: int = 32  # head dimension (P)
    chunk_size: int = 10  # matrix partition size (Q)
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // self.headdim

class InferenceCache(NamedTuple):
    conv_state: Tensor  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: Tensor  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, args: Mamba2Config, device: Device = None):
        return InferenceCache(
            torch.zeros(
                batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
            ),
            torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device
            ),
        )

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def silu(x):
    return x * F.sigmoid(x)

def segsum(x: Tensor, device: Device = None) -> Tensor:
    """Stable segment sum calculation."""
    T = x.size(-1)
    if device is None:
        device = x.device
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
    """Structed State Space Duality (SSD) - the core of Mamba-2"""
    # Pad if sequence length is not divisible by chunk_size
    T_orig = x.shape[1]
    if T_orig % chunk_size != 0:
        pad_len = chunk_size - (T_orig % chunk_size)
        x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        A = F.pad(A, (0, 0, 0, pad_len))
        B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
    
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    
    # Truncate to original length
    return Y[:, :T_orig], final_state

class Mamba2(nn.Module):
    def __init__(self, d_model=128, d_state=64, d_conv=4, expand=2, headdim=32, chunk_size=10, device: Device = None):
        super().__init__()
        # Adapting to match the TDD interface (kwargs in __init__)
        self.args = Mamba2Config(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand, 
            headdim=headdim,
            chunk_size=chunk_size
        )
        self.args.__post_init__()
        self.device = device
        args = self.args

        # Order: (z, x, B, C, dt)
        d_in_proj = 2 * args.d_inner + 2 * args.d_state + args.nheads
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False, device=device)

        conv_dim = args.d_inner + 2 * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
            device=device,
        )

        self.dt_bias = nn.Parameter(torch.empty(args.nheads, device=device))
        self.A_log = nn.Parameter(torch.empty(args.nheads, device=device))
        self.D = nn.Parameter(torch.empty(args.nheads, device=device))
        self.norm = RMSNorm(args.d_inner, device=device)
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False, device=device)
        
        # Init weights (simple init)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.normal_(self.dt_bias, std=0.02)
        nn.init.normal_(self.A_log, mean=math.log(0.5))
        nn.init.normal_(self.D, std=1.0)


    def forward(self, u: Tensor, h: Union[InferenceCache, None] = None):
        """
        u: (batch, seqlen, d_model)
        """
        A = -torch.exp(self.A_log)  # (nheads,)
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # Pad or truncate xBC seqlen to d_conv
        xBC_t = rearrange(xBC, "b l d -> b d l")
        # Ensure sufficient padding for conv if seqlen < d_conv
        # But for conv1d with padding, it should be fine. 
        # The original code pads manualy for inference cache alignment.
        # We simplify for batch training mode.
        
        xBC_conv = self.conv1d(xBC_t)[:, :, : u.shape[1]]
        xBC = silu(rearrange(xBC_conv, "b d l -> b l d"))
        
        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        
        y, ssm_state = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.args.chunk_size,
            device=self.device,
        )
        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        # h = InferenceCache(conv_state, ssm_state) # Skipping inference cache for now
        return y
