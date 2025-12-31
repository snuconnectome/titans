import torch
import torch.nn.functional as F
from torch import Tensor

def representational_similarity_analysis(
    activations_a: Tensor, 
    activations_b: Tensor, 
    method: str = 'correlation'
) -> float:
    """
    Computes the RSA score between two sets of representations (e.g., Model Memory vs Brain fMRI).
    
    Args:
        activations_a: (N, Dim_A) - e.g., Titan Memory States
        activations_b: (N, Dim_B) - e.g., Flattened Brain Voxels
        method: 'correlation' or 'euclidean'
        
    Returns:
        similarity_score: scalar representing the correlation between the RDMs.
    """
    
    # 1. Normalize
    a_norm = F.normalize(activations_a, p=2, dim=1)
    b_norm = F.normalize(activations_b, p=2, dim=1)
    
    # 2. Compute RDMs (Dissimilarity = 1 - Correlation)
    # Correlation Matrix = X @ X.T
    rdm_a = 1 - (a_norm @ a_norm.T)
    rdm_b = 1 - (b_norm @ b_norm.T)
    
    # 3. Flatten upper triangle (excluding diagonal) to compare the structures
    n = rdm_a.shape[0]
    tri_u_indices = torch.triu_indices(n, n, offset=1)
    
    vec_a = rdm_a[tri_u_indices[0], tri_u_indices[1]]
    vec_b = rdm_b[tri_u_indices[0], tri_u_indices[1]]
    
    # 4. Compute Spearman or Pearson correlation between the two geometry vectors
    # Simple Pearson: Cosine similarity of centered vectors
    vec_a_centered = vec_a - vec_a.mean()
    vec_b_centered = vec_b - vec_b.mean()
    
    rsa_score = F.cosine_similarity(vec_a_centered.unsqueeze(0), vec_b_centered.unsqueeze(0))
    
    return rsa_score.item()

def compute_mse(pred: Tensor, target: Tensor) -> float:
    """
    Computes Mean Squared Error between prediction and target.
    """
    return F.mse_loss(pred, target).item()

def compute_voxel_correlation(pred: Tensor, target: Tensor) -> float:
    """
    Computes average voxel-wise correlation over time.
    Inputs: (Batch, Time, ...)
    """
    # Flatten to (Batch, Time, Voxels)
    B, T = pred.shape[:2]
    pred_flat = pred.reshape(B, T, -1)
    target_flat = target.reshape(B, T, -1)
    
    # Centering
    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)
    
    pred_centered = pred_flat - pred_mean
    target_centered = target_flat - target_mean
    
    # Cosine similarity along Time dimension
    # (B, Voxels)
    numerator = (pred_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt((pred_centered ** 2).sum(dim=1)) * torch.sqrt((target_centered ** 2).sum(dim=1))
    
    # Avoid div by zero
    correlation = numerator / (denominator + 1e-8)
    
    return correlation.mean().item()
