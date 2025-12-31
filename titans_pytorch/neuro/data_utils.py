import torch
import numpy as np
from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
    """
    Slices long 4D fMRI sequence into training windows.
    Input: (Time, C, D, H, W)
    Returns: 
        x: (Window_Size, C, D, H, W)
        y: (Horizon, C, D, H, W) - Targets immediately following x
    """
    def __init__(self, data_tensor, window_size=16, horizon=3, transform=None):
        self.data = data_tensor
        self.window_size = window_size
        self.horizon = horizon
        self.transform = transform
        
        # Calculate number of valid windows
        # Total T
        # We need T >= t + window + horizon
        # Max t = T - window - horizon
        self.num_samples = self.data.shape[0] - self.window_size - self.horizon + 1
        
    def __len__(self):
        return max(0, self.num_samples)
    
    def __getitem__(self, idx):
        # Slice data
        start = idx
        mid = start + self.window_size
        end = mid + self.horizon
        
        x = self.data[start:mid]
        y = self.data[mid:end]
        
        # Apply Augmentation (Transform)
        # Transform should apply consistent spatial transform to both x and y
        if self.transform:
            # Stack x and y to apply same transform
            # (W+H, C, D, H, W)
            combined = torch.cat([x, y], dim=0)
            combined = self.transform(combined)
            x = combined[:self.window_size]
            y = combined[self.window_size:]
            
        return x, y

class RandomRotate3D:
    """
    Random 90 degree rotations for 3D augmentation.
    """
    def __call__(self, x):
        # x: (Time, C, D, H, W)
        # Randomly choose axes to rotate
        k = np.random.randint(0, 4) # 0, 1, 2, 3 rotations
        if k == 0: return x
        
        dims = np.random.choice([2, 3, 4], size=2, replace=False)
        return torch.rot90(x, k, dims=dims.tolist())

class RandomFlip3D:
    """
    Random flip along D, H, W axes.
    """
    def __call__(self, x):
        # x: (Time, C, D, H, W)
        if np.random.rand() < 0.5:
            x = torch.flip(x, [2]) # Flip D
        if np.random.rand() < 0.5:
            x = torch.flip(x, [3]) # Flip H
        if np.random.rand() < 0.5:
            x = torch.flip(x, [4]) # Flip W
        return x

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

