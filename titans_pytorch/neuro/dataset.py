import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob

class MockBudapestDataset(Dataset):
    """
    Simulates the structure of OpenNeuro ds003017 (The Grand Budapest Hotel).
    Shape: (Time, Channels, Depth, Height, Width)
    """
    def __init__(self, num_timepoints=100, spatial_shape=(32, 32, 32), channels=1):
        self.num_timepoints = num_timepoints
        self.spatial_shape = spatial_shape
        self.channels = channels
        self.data = torch.randn(num_timepoints, channels, *spatial_shape)

    def __len__(self):
        return 1

    def get_full_sequence(self):
        return self.data

class BudapestDataset(Dataset):
    """
    Loader for OpenNeuro ds003017 BIDS dataset.
    Expected structure: sub-XX/func/sub-XX_task-movie_bold.nii.gz
    """
    def __init__(self, root_dir, subject_id='01', task_name='movie', chunk_size=None):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.task_name = task_name
        self.chunk_size = chunk_size
        
        # BIDS pattern matcher
        # ds003017/sub-01/func/sub-01_task-movie_bold.nii.gz (Adjust based on actual file names found)
        # Search for files
        search_pattern = os.path.join(root_dir, f"sub-{subject_id}", "func", f"*task-{task_name}*_bold.nii.gz")
        self.files = sorted(glob.glob(search_pattern))
        
        if not self.files:
            print(f"Warning: No files found for pattern {search_pattern}")
        
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
        
        # Optional: Chunking logic if the sequence is too long for memory?
        # But Titans is designed for infinite context, so we might return the whole thing
        # or handle slicing in a collate_fn if making batches of subsequences.
        
        return data
