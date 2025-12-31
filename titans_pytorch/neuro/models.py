import torch
import torch.nn as nn
from titans_pytorch.memory.neural_memory import NeuralMemory
from titans_pytorch.models.mac import MemoryAsContext

class Simple3DEncoder(nn.Module):
    def __init__(self, channels=1, hidden_dim=64, input_spatial_shape=(32,32,32)):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv3d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        
        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, channels, *input_spatial_shape)
            out = self.conv_net(dummy)
            self.flat_size = out.numel() // out.shape[0]
            self.final_shape = out.shape[1:] # C, D, H, W after conv
            
        self.fc = nn.Linear(self.flat_size, hidden_dim)

    def forward(self, x):
        features = self.conv_net(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)

class Simple3DDecoder(nn.Module):
    def __init__(self, hidden_dim=64, channels=1, output_shape=(32,32,32), encoder_flat_size=None, encoder_final_shape=None):
        super().__init__()
        self.output_shape = output_shape
        
        # Use encoder info if provided, else define default (assuming 32x32x32 input logic if not)
        # But to be safe, let's rely on passed args or Recalculate if needed.
        # For this prototype, we'll assume we can inverse the Encoder logic or just use ConvTranspose.
        
        self.encoder_final_shape = encoder_final_shape if encoder_final_shape else (64, 4, 4, 4)
        self.flat_size = encoder_flat_size if encoder_flat_size else 64*4*4*4
        
        self.input_linear = nn.Linear(hidden_dim, self.flat_size)
        
        self.net = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, channels, kernel_size=4, stride=2, padding=1),
        )
        
    def forward(self, x):
        x = self.input_linear(x)
        x = x.view(-1, *self.encoder_final_shape)
        return self.net(x)

class TitanNeuro(nn.Module):
    """
    Titan V3 for 4D fMRI.
    Structure:
    1. Spatial Encoder: 3D Vol -> Feature Vector (per timestep)
    2. Temporal Memory: Titan Neural Memory (sequence modeling)
    3. Spatial Decoder: Feature Vector -> Predicted 3D Vol (Next timestep)
    """
    def __init__(self, spatial_shape=(32,32,32), hidden_dim=128):
        super().__init__()
        self.encoder = Simple3DEncoder(channels=1, hidden_dim=hidden_dim, input_spatial_shape=spatial_shape)
        
        # Using Titan V3 Neural Memory
        self.memory = NeuralMemory(dim=hidden_dim, layer_width=hidden_dim*2)
        
        self.decoder = Simple3DDecoder(
            hidden_dim=hidden_dim, 
            channels=1, 
            output_shape=spatial_shape,
            encoder_flat_size=self.encoder.flat_size,
            encoder_final_shape=self.encoder.final_shape
        )
        
    def forward(self, x, use_chunked=False, chunk_size=64):
        """
        x: (Batch, Time, Channels, D, H, W)
        Returns: Prediction (Batch, Time, Channels, D, H, W) - Next Step Prediction
        """
        B, T, C, D, H, W = x.shape
        
        # 1. Spatial Encoding (Frame by Frame)
        # Flatten batch and time for encoder
        x_flat = x.view(B*T, C, D, H, W)
        embeddings = self.encoder(x_flat) # (B*T, hidden_dim)
        embeddings = embeddings.view(B, T, -1) # (B, T, hidden_dim)
        
        # 2. Titan Neural Memory (Temporal Modeling)
        # We model the sequence of brain states
        # Output is (B, T, hidden_dim)
        if use_chunked:
            memory_out, _ = self.memory.forward_chunked(embeddings, chunk_size=chunk_size)
        else:
            memory_out, _ = self.memory(embeddings)
        
        # 3. Spatial Decoding (Predict Next Brain State)
        memory_out_flat = memory_out.view(B*T, -1)
        pred_flat = self.decoder(memory_out_flat)
        pred = pred_flat.view(B, T, C, D, H, W)
        
        return pred

class TitanBrainEncoding(nn.Module):
    """
    Task 2: Brain Encoding (Stimulus -> Brain)
    Predicts fMRI responses from stimulus features using Titan Neural Memory.
    """
    def __init__(self, stimulus_dim=40, voxel_dim=50000, hidden_dim=128):
        super().__init__()
        self.projector = nn.Linear(stimulus_dim, hidden_dim)
        self.memory = NeuralMemory(dim=hidden_dim, layer_width=hidden_dim*2)
        self.predictor = nn.Linear(hidden_dim, voxel_dim)

    def forward(self, stimulus, use_chunked=False, chunk_size=64):
        """
        stimulus: (Batch, Time, FeatDim)
        Returns: brain_pred: (Batch, Time, VoxelDim)
        """
        # 1. Project to hidden space
        x = self.projector(stimulus)
        
        # 2. Temporal modeling with Titans
        if use_chunked:
            x, _ = self.memory.forward_chunked(x, chunk_size=chunk_size)
        else:
            x, _ = self.memory(x)
            
        # 3. Predict voxels
        brain_pred = self.predictor(x)
        return brain_pred

class TitanSemanticDecoding(nn.Module):
    """
    Task 3: Semantic Decoding (Brain -> Stimulus)
    Predicts stimulus features from fMRI responses using Titan Neural Memory.
    """
    def __init__(self, voxel_dim=50000, stimulus_dim=40, hidden_dim=128):
        super().__init__()
        self.projector = nn.Linear(voxel_dim, hidden_dim)
        self.memory = NeuralMemory(dim=hidden_dim, layer_width=hidden_dim*2)
        self.predictor = nn.Linear(hidden_dim, stimulus_dim)

    def forward(self, brain, use_chunked=False, chunk_size=64):
        """
        brain: (Batch, Time, VoxelDim)
        Returns: stimulus_pred: (Batch, Time, FeatDim)
        """
        # 1. Project to hidden space
        x = self.projector(brain)
        
        # 2. Temporal modeling with Titans
        if use_chunked:
            x, _ = self.memory.forward_chunked(x, chunk_size=chunk_size)
        else:
            x, _ = self.memory(x)
            
        # 3. Predict stimulus features
        stimulus_pred = self.predictor(x)
        return stimulus_pred
