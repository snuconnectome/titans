import torch
import time
import numpy as np
from titans_pytorch.neuro.models import TitanNeuro

def benchmark():
    print("="*60)
    print("🚀 Titan-Neuro Performance Benchmark (Synthetic Data)")
    print("="*60)

    # 1. Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True # Optimize conv speeds

    # 2. Synthetic Data Params
    B, T = 1, 512 # Fairly long sequence
    C, D, H, W = 1, 32, 32, 32 
    hidden_dim = 128
    
    print(f"\nConfiguration:")
    print(f"  Batch: {B}, Time: {T}")
    print(f"  Volume: {D}x{H}x{W}")
    print(f"  Memory Dim: {hidden_dim}")

    # 3. Instantiate Model
    model = TitanNeuro(spatial_shape=(D,H,W), hidden_dim=hidden_dim).to(device)
    model.eval() # Eval mode mainly

    # Create dummy input
    x = torch.randn(B, T, C, D, H, W).to(device)

    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        _ = model(x[:, :16]) 

    # --- Benchmark Sequential ---
    print("\n[Method 1] Sequential (Original)")
    start_time = time.time()
    if device.type == 'cuda': torch.cuda.synchronize()
    
    with torch.no_grad():
        _ = model(x, use_chunked=False)
        
    if device.type == 'cuda': torch.cuda.synchronize()
    end_time = time.time()
    seq_time = end_time - start_time
    print(f"  Time: {seq_time:.4f} sec")
    print(f"  FPS : {T / seq_time:.2f} volumes/sec")

    # --- Benchmark Chunked (Parallel) ---
    chunk_sizes = [32, 64, 128]
    
    for cs in chunk_sizes:
        print(f"\n[Method 2] Chunked Parallel (Chunk={cs})")
        start_time = time.time()
        if device.type == 'cuda': torch.cuda.synchronize()
        
        with torch.no_grad():
            _ = model(x, use_chunked=True, chunk_size=cs)
            
        if device.type == 'cuda': torch.cuda.synchronize()
        end_time = time.time()
        chunk_time = end_time - start_time
        speedup = seq_time / chunk_time
        print(f"  Time: {chunk_time:.4f} sec")
        print(f"  FPS : {T / chunk_time:.2f} volumes/sec")
        print(f"  Speedup: {speedup:.2f}x")

    print("\nBenchmark Complete.")

if __name__ == "__main__":
    benchmark()
