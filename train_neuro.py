import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time

from titans_pytorch.neuro.models import TitanNeuro
from titans_pytorch.neuro.dataset import BudapestDataset, MockBudapestDataset

def train(args):
    print(f"🚀 Starting Titan-Neuro Training")
    print(f"   Dataset: {args.dataset}")
    print(f"   Batch: {args.batch_size}, Chunk: {args.chunk_size}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # 1. Dataset
    if args.dataset == 'mock':
        dataset = MockBudapestDataset(num_timepoints=args.seq_len, spatial_shape=(32,32,32))
    else:
        # Assumes DataLad info is present or files exist
        dataset = BudapestDataset(root_dir=args.data_root)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 2. Model
    model = TitanNeuro(spatial_shape=(32,32,32), hidden_dim=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            # batch: (B, T, C, D, H, W)
            batch = batch.to(device)
            
            # Next-Volume Prediction
            # Input: x[0 : T-1]
            # Target: x[1 : T]
            input_seq = batch[:, :-1]
            target_seq = batch[:, 1:]
            
            optimizer.zero_grad()
            
            # Forward (Chunked for speed)
            pred = model(input_seq, use_chunked=True, chunk_size=args.chunk_size)
            
            loss = criterion(pred, target_seq)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.6f}")
        
        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.6f}. Time: {elapsed:.2f}s")
        
        # Checkpoint
        if (epoch + 1) % 5 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            ckpt_path = f"checkpoints/titan_neuro_e{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mock', choices=['mock', 'openneuro'])
    parser.add_argument('--data_root', type=str, default='./ds003017')
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--chunk_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    args = parser.parse_args()
    train(args)
