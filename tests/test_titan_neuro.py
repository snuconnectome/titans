import torch
import pytest
from titans_pytorch.neuro.models import TitanNeuro
from titans_pytorch.neuro.dataset import MockRaidersDataset

def test_titan_neuro_forward_shape():
    # Setup
    B = 1
    T = 10
    spatial_shape = (32, 32, 32)
    model = TitanNeuro(spatial_shape=spatial_shape, hidden_dim=64)
    
    # Input: (B, T, C, D, H, W)
    x = torch.randn(B, T, 1, *spatial_shape)
    
    # Forward
    pred = model(x)
    
    # Check shape
    assert pred.shape == x.shape

def test_titan_neuro_overfit_mock():
    # Can the model learn a simple pattern from Mock Data?
    dataset = MockRaidersDataset(num_timepoints=10, spatial_shape=(16, 16, 16)) # Smaller for speed
    data = dataset.get_full_sequence().unsqueeze(0) # Add Batch Dim: (1, 10, 1, 16, 16, 16)
    
    model = TitanNeuro(spatial_shape=(16, 16, 16), hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    
    # Target: Next Step Prediction (Shifted Input)
    # Input: x[0...T-1], Target: x[1...T]
    input_seq = data[:, :-1]
    target_seq = data[:, 1:]
    
    initial_loss = 0
    final_loss = 0
    
    for i in range(5): # 5 Steps
        optimizer.zero_grad()
        pred = model(input_seq)
        loss = loss_fn(pred, target_seq)
        loss.backward()
        optimizer.step()
        
        if i == 0: initial_loss = loss.item()
        final_loss = loss.item()
        
    print(f"Initial Neuro Loss: {initial_loss}, Final: {final_loss}")
    assert final_loss < initial_loss
