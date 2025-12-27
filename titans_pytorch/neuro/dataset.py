import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MockRaidersDataset(Dataset):
    """
    Simulates the structure of the Haxby Lab Raiders fMRI dataset.
    Used for TDD and verification when full data is not available.
    
    Shape: (Time, Channels, Depth, Height, Width)
    SwiFT expects: (B, C, D, H, W, T) or similar. Titans usually expects (B, T, Feature).
    
    We will output: (Time, Channels, D, H, W) for a single subject run.
    """
    def __init__(self, num_timepoints=100, spatial_shape=(32, 32, 32), channels=1):
        self.num_timepoints = num_timepoints
        self.spatial_shape = spatial_shape
        self.channels = channels
        
        # synthetic data: [T, C, D, H, W]
        self.data = torch.randn(num_timepoints, channels, *spatial_shape)
        
        # Simulate some temporal correlation (HRF-like)
        for t in range(1, num_timepoints):
            self.data[t] = 0.8 * self.data[t-1] + 0.2 * torch.randn(channels, *spatial_shape)

    def __len__(self):
        # In a real scenario, this might be number of subjects or runs.
        # Here we treat the dataset as providing ONE sequence (the movie).
        # But PyTorch Dataset usually returns *samples*.
        # For fMRI language modeling, we might slice the long sequence into windows.
        return 1

    def get_full_sequence(self):
        return self.data

class RealRaidersDataset(Dataset):
    """
    Loader for actual NIfTI files from Raiders dataset.
    Requires 'nibabel' and actual files.
    """
    def __init__(self, data_dir, mask_path=None):
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
        try:
            import nibabel as nib
            self.nib = nib
        except ImportError:
            raise ImportError("Please install nibabel: pip install nibabel")
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]
        img = self.nib.load(path)
        data = img.get_fdata() # (X, Y, Z, T)
        
        # Convert to Tensor (T, C, D, H, W)
        # 1. Transpose: T, X, Y, Z
        data = np.transpose(data, (3, 0, 1, 2))
        data = torch.from_numpy(data).float()
        
        # 2. Add Channel dim: T, 1, X, Y, Z
        data = data.unsqueeze(1)
        
        return data
