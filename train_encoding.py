import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
from tqdm import tqdm
import json
import pandas as pd

from titans_pytorch.neuro.dataset import NaturalLanguageDataset
from titans_pytorch.neuro.models import TitanBrainEncoding

# Identity Model
class IdentityEncoding(nn.Module):
    def __init__(self, stimulus_dim, voxel_dim):
        super().__init__()
        # Simple mean projection as baseline
        self.proj = nn.Linear(stimulus_dim, voxel_dim)
        
    def forward(self, x, **kwargs):
        # x: (B, T, D)
        return self.proj(x)

# Placeholder Wrappers for other models (SwiFT, NeuroSTORM)
# Since they are 3D-native, we need to adapt them for Voxel Encoding (1D vector output)
# Or we can just use simple MLP/Transformer baselines for "vector" encoding tasks.
class TransformerEncoding(nn.Module):
    def __init__(self, stimulus_dim, voxel_dim, hidden_dim=128):
        super().__init__()
        self.proj = nn.Linear(stimulus_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(hidden_dim, voxel_dim)
        
    def forward(self, x, **kwargs):
        x = self.proj(x)
        x = self.transformer(x)
        return self.head(x)

class MambaEncoding(nn.Module):
    def __init__(self, stimulus_dim, voxel_dim, hidden_dim=128):
        super().__init__()
        self.proj = nn.Linear(stimulus_dim, hidden_dim)
        try:
            from mamba_ssm import Mamba
            self.mamba = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
        except ImportError:
            print("Warning: mamba_ssm not found, using Linear fallback")
            self.mamba = nn.Linear(hidden_dim, hidden_dim)
            
        self.head = nn.Linear(hidden_dim, voxel_dim)
        
    def forward(self, x, **kwargs):
        x = self.proj(x)
        x = self.mamba(x)
        return self.head(x)

def get_model(name, s_dim, v_dim, h_dim):
    if name == "Identity":
        return IdentityEncoding(s_dim, v_dim)
    elif name == "Transformer":
        return TransformerEncoding(s_dim, v_dim, h_dim)
    elif name == "NeuroSTORM": # Mamba-based
        return MambaEncoding(s_dim, v_dim, h_dim)
    elif name == "Titans-Neuro":
        return TitanBrainEncoding(s_dim, v_dim, h_dim)
    else:
        raise ValueError(f"Unknown model: {name}")

def train_encoding(args):
    print(f"🚀 Task 2: Brain Encoding - 4 Model Comparison")
    print(f"   Subject: {args.subject_id}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")

    # 1. Dataset
    dataset = NaturalLanguageDataset(
        root_dir=args.data_root, 
        subject_id=args.subject_id
    )
    
    if len(dataset) == 0:
        print("❌ No matching fMRI/Audio samples found. Check paths.")
        return

    first_sample = dataset[0]
    voxel_dim = first_sample['brain'].shape[1]
    stimulus_dim = first_sample['stimulus'].shape[1]
    
    print(f"   Voxel Dimension: {voxel_dim}")
    print(f"   Stimulus Dimension: {stimulus_dim}")

    # 2. Models to Benchmark
    models_list = ["Identity", "Transformer", "NeuroSTORM", "Titans-Neuro"]
    results = []
    
    for model_name in models_list:
        print(f"\n🧠 Training {model_name}...")
        
        try:
            model = get_model(model_name, stimulus_dim, voxel_dim, args.hidden_dim).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
            criterion = nn.MSELoss()
            
            history = {"loss": []}
            
            # Training Loop
            model.train()
            start_train = time.time()
            
            for epoch in range(args.epochs):
                epoch_loss = 0
                count = 0
                
                # Simple batching: 1 story per batch
                for sample in dataset:
                    stim = sample['stimulus'].unsqueeze(0).to(device)
                    brain = sample['brain'].unsqueeze(0).to(device)
                    
                    optimizer.zero_grad()
                    
                    # Use chunked only for Titans if needed, others are standard
                    kwargs = {}
                    if model_name == "Titans-Neuro":
                        kwargs = {'use_chunked': True, 'chunk_size': args.chunk_size}
                        
                    pred = model(stim, **kwargs)
                    
                    loss = criterion(pred, brain)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    count += 1
                
                avg_loss = epoch_loss / count
                history["loss"].append(avg_loss)
                print(f"   Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f}")
                
            elapsed = time.time() - start_train
            final_loss = history["loss"][-1]
            
            print(f"   ✅ Done. Final Loss: {final_loss:.6f} ({elapsed:.1f}s)")
            
            results.append({
                "Model": model_name,
                "Final Loss": final_loss,
                "Time (s)": elapsed,
                "History": history["loss"]
            })
            
            # Save Model
            os.makedirs('checkpoints/encoding', exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/encoding/{model_name}_{args.subject_id}.pt")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()

    # 3. Save Results for Dashboard
    df = pd.DataFrame(results)
    df.to_json("encoding_benchmark_results.json", orient="records", indent=2)
    print("\n🏆 Encoding Leaderboard saved to encoding_benchmark_results.json")
    print(df[["Model", "Final Loss", "Time (s)"]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/juke/git/ds003020')
    parser.add_argument('--subject_id', type=str, default='UTS01')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--chunk_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    args = parser.parse_args()
    train_encoding(args)

