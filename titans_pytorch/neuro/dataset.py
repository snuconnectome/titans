import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import h5py
import librosa

class NaturalLanguageDataset(Dataset):
    """
    Loader for OpenNeuro ds003020 (Natural Language fMRI).
    Connects preprocessed fMRI (.hf5) with audio stimuli (.wav).
    """
    def __init__(
        self, 
        root_dir="/home/juke/git/ds003020", 
        subject_id='UTS01', 
        tr=2.0,
        window_size=16,
        extract_audio_features=True
    ):
        self.root_dir = root_dir
        self.subject_id = subject_id
        self.tr = tr
        self.window_size = window_size
        self.extract_audio_features = extract_audio_features
        
        # 1. Locate preprocessed fMRI data
        self.preproc_dir = os.path.join(root_dir, "derivative", "preprocessed_data", subject_id)
        self.fmri_files = sorted(glob.glob(os.path.join(self.preproc_dir, "*.hf5")))
        
        # 2. Locate stimuli
        self.stim_dir = os.path.join(root_dir, "stimuli")
        
        # 3. Match stories (files)
        self.samples = []
        for f in self.fmri_files:
            story_name = os.path.basename(f).replace(".hf5", "")
            audio_path = os.path.join(self.stim_dir, f"{story_name}.wav")
            
            if os.path.exists(audio_path):
                self.samples.append({
                    "story": story_name,
                    "fmri_path": f,
                    "audio_path": audio_path
                })
            else:
                print(f"Warning: Audio not found for {story_name} at {audio_path}")

    def __len__(self):
        return len(self.samples)

    def _get_fmri_data(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            # Assuming standard structure in ds003020 hf5 files
            # Typically 'data' or subject-named key
            keys = list(f.keys())
            data = f[keys[0]][:] # (Time, Voxels)
        return torch.from_numpy(data).float()

    def _extract_features(self, audio_path, num_trs):
        # Placeholder for real audio feature extraction (e.g., Log-Mel or Wav2Vec)
        # Here we use simple MFCCs as a baseline for actual data
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Target duration in seconds
        target_duration = num_trs * self.tr
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40) # (40, Frames)
        
        # Resample/Average to match TRs
        # This is a simplification: in reality, we align with the 2.0s TR
        feat_dim = mfcc.shape[0]
        features = np.zeros((num_trs, feat_dim))
        
        frames_per_tr = int(sr * self.tr / 512) # Approximate hop length
        for i in range(num_trs):
            start = i * frames_per_tr
            end = (i + 1) * frames_per_tr
            if start < mfcc.shape[1]:
                features[i] = mfcc[:, start:end].mean(axis=1)
                
        return torch.from_numpy(features).float()

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load fMRI (Target for Encoding, Input for Decoding)
        fmri = self._get_fmri_data(sample['fmri_path'])
        
        # Extract/Load Stimulus (Input for Encoding, Target for Decoding)
        stimulus = self._extract_features(sample['audio_path'], fmri.shape[0])
        
        # Basic alignment check
        min_len = min(fmri.shape[0], stimulus.shape[0])
        fmri = fmri[:min_len]
        stimulus = stimulus[:min_len]
        
        return {
            "story": sample['story'],
            "brain": fmri,      # (Time, Voxels)
            "stimulus": stimulus # (Time, FeatDim)
        }

class MockBudapestDataset(Dataset):
    """
    Simulates the structure of OpenNeuro ds003017 (The Grand Budapest Hotel).
    Shape: (Time, Channels, Depth, Height, Width)
    
    Real Data Specs (TR=2s):
    - Run 1: 598s (~299 vols)
    - Run 2: 498s (~249 vols)
    - Run 3: 535s (~267 vols)
    - Run 4: 618s (~309 vols)
    - Run 5: 803s (~401 vols)
    """
    def __init__(self, num_timepoints=299, spatial_shape=(32, 32, 32), channels=1):
        self.num_timepoints = num_timepoints
        self.spatial_shape = spatial_shape
        self.channels = channels
        # Simulate normalized fMRI data (mean 0, std 1)
        self.data = torch.randn(num_timepoints, channels, *spatial_shape)

    def __len__(self):
        return 1

    def get_full_sequence(self):
        return self.data
    
    def __getitem__(self, idx):
        # Returns the full sequence for the 'subject'
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
        
        return data
