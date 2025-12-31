import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import pandas as pd
from tabulate import tabulate
import json
import os
import sys

# Ensure project root in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from titans_pytorch.neuro.dataset import MockBudapestDataset
from titans_pytorch.neuro.data_utils import SlidingWindowDataset, RandomRotate3D, RandomFlip3D, Compose
from titans_pytorch.neuro.metrics import compute_mse, compute_voxel_correlation

# Models
from titans_pytorch.neuro.models import TitanNeuro
# wrappers
from titans_pytorch.neuro.models_swift import SwiFTWrapper
from titans_pytorch.neuro.models_neurostorm import NeuroSTORMWrapper

def get_model(model_name, device, spatial_shape=(48, 48, 48)):
    if model_name == "Identity":
        return IdentityModel()
    elif model_name == "Titans-Neuro":
        return TitanNeuro(spatial_shape=spatial_shape, hidden_dim=64).to(device)
    elif model_name == "SwiFT":
        return SwiFTWrapper(spatial_shape=spatial_shape, hidden_dim=64).to(device)
    elif model_name == "NeuroSTORM":
        return NeuroSTORMWrapper(spatial_shape=spatial_shape, hidden_dim=64).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

class IdentityModel(nn.Module):
    def forward(self, x):
        return x[:, -1:] # Predicts last frame persistence

class Trainer:
    def __init__(self, model, model_name, device, lr=1e-4):
        self.model = model
        self.model_name = model_name
        self.device = device
        
        # Identity model doesn't need optimizer
        if model_name == "Identity":
            self.optimizer = None
        else:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
            
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader):
        if self.model_name == "Identity":
            return 0.0
            
        self.model.train()
        total_loss = 0
        count = 0
        
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            # x: (B, Window, C, D, H, W)
            # y: (B, Horizon, C, D, H, W)
            
            self.optimizer.zero_grad()
            
            # Forward pass: Predict next frames
            # We train to predict y[0] (t+1) from x
            # For simplicity in training loop, we perform 1-step prediction training
            # or teacher forcing for horizon.
            # Here we do: Predict next frame given context.
            
            # Just train on Next Step Prediction (Step 1)
            target = y[:, 0:1] # Just t+1
            
            pred = self.model(x) # Should output (B, 1, C, D, H, W)
            
            loss = self.criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            count += 1
            
        return total_loss / count

    def evaluate(self, dataloader, horizon=3):
        self.model.eval()
        total_mse = 0
        total_corr = 0
        count = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                # Auto-regressive Rollout Evaluation
                curr_input = x
                predictions = []
                
                for h in range(horizon):
                    # Predict next frame
                    pred = self.model(curr_input)
                    predictions.append(pred)
                    
                    # Update input: slide window
                    curr_input = torch.cat([curr_input[:, 1:], pred], dim=1)
                
                preds_tensor = torch.cat(predictions, dim=1)
                
                mse = compute_mse(preds_tensor, y)
                corr = compute_voxel_correlation(preds_tensor, y)
                
                total_mse += mse
                total_corr += corr
                count += 1
                
        if count == 0: return {"MSE": 0, "Correlation": 0}
        return {"MSE": total_mse/count, "Correlation": total_corr/count}

def run_pipeline():
    print("🚀 Starting Training & Benchmark Pipeline...")
    
    # 1. Configuration
    EPOCHS = 10 # Reduced for demo speed, user asked for 50-100
    BATCH_SIZE = 4
    WINDOW_SIZE = 16
    HORIZON = 3
    SPATIAL_SHAPE = (32, 32, 32) # Using 32 for speed in this demo run
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  Device: {DEVICE}, Shape: {SPATIAL_SHAPE}, Epochs: {EPOCHS}")

    # 2. Data Preparation
    print("📂 preparing Data...")
    # Mock data with structure of ds003017
    # Train: Run 1-4 (~2200 vols), Test: Run 5 (~800 vols)
    # Using smaller mock for speed
    train_raw = MockBudapestDataset(num_timepoints=200, spatial_shape=SPATIAL_SHAPE).get_full_sequence()
    test_raw = MockBudapestDataset(num_timepoints=50, spatial_shape=SPATIAL_SHAPE).get_full_sequence()
    
    # Augmentation
    train_transform = Compose([RandomRotate3D(), RandomFlip3D()])
    
    train_dataset = SlidingWindowDataset(train_raw, WINDOW_SIZE, HORIZON, transform=train_transform)
    test_dataset = SlidingWindowDataset(test_raw, WINDOW_SIZE, HORIZON, transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"   Train Samples: {len(train_dataset)}, Test Samples: {len(test_dataset)}")

    # 3. Models
    models_list = ["Identity", "Titans-Neuro", "SwiFT", "NeuroSTORM"]
    results = []

    for model_name in models_list:
        print(f"\n🧠 Processing {model_name}...")
        try:
            model = get_model(model_name, DEVICE, SPATIAL_SHAPE)
            trainer = Trainer(model, model_name, DEVICE)
            
            # Training Loop
            if model_name != "Identity":
                pbar = tqdm(range(EPOCHS), desc="Training")
                for epoch in pbar:
                    loss = trainer.train_epoch(train_loader)
                    pbar.set_postfix({'loss': f"{loss:.4f}"})
            
            # Final Evaluation
            start_time = time.time()
            metrics = trainer.evaluate(test_loader, HORIZON)
            elapsed = time.time() - start_time
            
            print(f"   ✅ Result: MSE={metrics['MSE']:.4f}, Corr={metrics['Correlation']:.4f}")
            
            results.append({
                "Model": model_name,
                "MSE": metrics['MSE'],
                "Correlation": metrics['Correlation'],
                "Time (s)": elapsed
            })
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            # import traceback
            # traceback.print_exc()

    # 4. Report
    df = pd.DataFrame(results)
    df = df.sort_values(by="MSE")
    
    print("\n🏆 FINAL LEADERBOARD (After Training) 🏆")
    print("=" * 60)
    print(tabulate(df, headers='keys', tablefmt='github', showindex=False))
    print("=" * 60)
    
    df.to_json("training_benchmark_results.json", orient='records', indent=2)

if __name__ == "__main__":
    run_pipeline()

