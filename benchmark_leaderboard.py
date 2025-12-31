import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tabulate import tabulate
import time
from pathlib import Path
import json
import sys
import os

# Ensure the project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from titans_pytorch.neuro.models import TitanNeuro
from titans_pytorch.neuro.dataset import MockBudapestDataset
from titans_pytorch.neuro.metrics import compute_mse, compute_voxel_correlation, representational_similarity_analysis

# Optional: Try to import SwiFT wrapper
try:
    from titans_pytorch.neuro.models_swift import SwiFTWrapper
    HAS_SWIFT = True
except ImportError:
    HAS_SWIFT = False

# Optional: Try to import NeuroSTORM wrapper
try:
    from titans_pytorch.neuro.models_neurostorm import NeuroSTORMWrapper
    HAS_NEUROSTORM = True
except ImportError as e:
    print(f"⚠️ NeuroSTORM not available: {e}")
    HAS_NEUROSTORM = False


def evaluate_model(model, dataloader, device, steps=10, horizon=3):
    """
    Evaluates model on Multi-step Trajectory Prediction (k=3)
    """
    model.eval()
    total_mse = 0
    total_corr = 0
    count = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= steps: break
            
            # batch: (B, T_total, C, D, H, W)
            # T_total must be > input_len + horizon
            x = batch.to(device)
            B, T_total, C, D, H, W = x.shape
            
            input_len = 16
            if T_total < input_len + horizon:
                continue

            # Context: 0 ~ 15 (16 frames)
            context = x[:, :input_len]
            
            # Target: 16, 17, 18 (3 frames)
            targets = x[:, input_len:input_len+horizon]
            
            # Auto-regressive Rollout
            curr_input = context
            predictions = []
            
            for h in range(horizon):
                # Predict next frame
                # Model output should be (B, 1, C, D, H, W) or sequence
                # For this benchmark, we assume model takes sequence and outputs next frame(s)
                # or we just use the last prediction.
                
                pred_seq = model(curr_input) # Expected (B, T_in, ...) or (B, 1, ...)
                
                # Take the last frame prediction
                if pred_seq.shape[1] == curr_input.shape[1]:
                    next_frame = pred_seq[:, -1:].clone()
                else:
                    next_frame = pred_seq[:, -1:].clone() # Assume model outputs only next frames
                
                predictions.append(next_frame)
                
                # Append to input for next step (Sliding window)
                # Remove first frame, add predicted frame
                curr_input = torch.cat([curr_input[:, 1:], next_frame], dim=1)

            # Stack predictions: (B, Horizon, C, D, H, W)
            preds_tensor = torch.cat(predictions, dim=1)
            
            # Metrics
            mse = compute_mse(preds_tensor, targets)
            corr = compute_voxel_correlation(preds_tensor, targets)
            
            total_mse += mse
            total_corr += corr
            count += 1
            
    if count == 0:
        return {"MSE": 999.0, "Correlation": 0.0}
        
    return {
        "MSE": total_mse / count,
        "Correlation": total_corr / count
    }

class IdentityModel(nn.Module):
    def forward(self, x):
        # Predicts next state is same as LAST input state
        return x[:, -1:]

def run_benchmark():
    print("🧠 Starting Titans Neuro Benchmark (Multi-step Horizon k=3)...")
    
    # 1. Setup Data
    print("📂 Loading Dataset (Mock)...")
    # T=50 is enough for few batches of 16+3
    # Use 48x48x48 to approximate realistic low-res fMRI (standard is often 64 or 96)
    # 16 is too small for hierarchical models like SwiFT
    spatial_shape = (48, 48, 48) 
    dataset = MockBudapestDataset(num_timepoints=50, spatial_shape=spatial_shape) 
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  Device: {device}")

    results = []

    # 2. Define Models to Benchmark
    models_to_test = [
        ("Identity (Baseline)", IdentityModel()),
        ("Titans-Neuro (Zero-Shot)", TitanNeuro(spatial_shape=spatial_shape, hidden_dim=64).to(device))
    ]
    
    if HAS_SWIFT:
        try:
            swift_model = SwiFTWrapper(spatial_shape=spatial_shape, hidden_dim=64).to(device)
            models_to_test.append(("SwiFT (SOTA)", swift_model))
        except Exception as e:
            print(f"Could not init SwiFT: {e}")
            
    if HAS_NEUROSTORM:
        try:
            # Note: NeuroSTORM might need mamba_ssm which is tricky to install. 
            # The code has a fallback though.
            neurostorm_model = NeuroSTORMWrapper(spatial_shape=spatial_shape, hidden_dim=64).to(device)
            models_to_test.append(("NeuroSTORM (SOTA)", neurostorm_model))
        except Exception as e:
            print(f"Could not init NeuroSTORM: {e}")

    # 3. Run Evaluation
    for name, model in models_to_test:
        print(f"🚀 Evaluating {name}...")
        start_time = time.time()
        
        try:
            metrics = evaluate_model(model, loader, device, steps=5, horizon=3)
            elapsed = time.time() - start_time
            
            results.append({
                "Model": name,
                "MSE": f"{metrics['MSE']:.4f}",
                "Correlation": f"{metrics['Correlation']:.4f}",
                "Time (s)": f"{elapsed:.2f}"
            })
        except Exception as e:
            print(f"❌ Failed {name}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Generate Leaderboard
    df = pd.DataFrame(results)
    
    print("\n🏆 NEURO-BENCHMARK LEADERBOARD (Horizon=6s) 🏆")
    print("=" * 60)
    print(tabulate(df, headers='keys', tablefmt='github', showindex=False))
    print("=" * 60)
    
    output_path = Path("benchmark_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_benchmark()
