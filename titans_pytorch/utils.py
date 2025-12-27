import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, TypeVar

T = TypeVar("T")

def exists(val: Optional[T]) -> bool:
    return val is not None

def default(val: Optional[T], default_val: T) -> T:
    return val if exists(val) else default_val

def l2_normalize(x: Tensor, dim: int = -1, eps: float = 1e-12) -> Tensor:
    return F.normalize(x, p=2, dim=dim, eps=eps)

def chunk_sequence(x: Tensor, chunk_size: int) -> Tensor:
    batch, seq_len, dim = x.shape
    if seq_len % chunk_size != 0:
        padding = chunk_size - (seq_len % chunk_size)
        x = F.pad(x, (0, 0, 0, padding))
        seq_len = x.shape[1]
    num_chunks = seq_len // chunk_size
    return x.reshape(batch, num_chunks, chunk_size, dim)

def unchunk_sequence(x: Tensor, original_len: Optional[int] = None) -> Tensor:
    batch, num_chunks, chunk_size, dim = x.shape
    x = x.reshape(batch, (num_chunks * chunk_size), dim)
    if exists(original_len):
        x = x[:, :original_len, :]
    return x

def compute_beta_cumulative(alpha: Tensor) -> Tensor:
    one_minus_alpha = 1 - alpha
    return torch.cumprod(one_minus_alpha, dim=1)
