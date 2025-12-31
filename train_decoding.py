import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
from tqdm import tqdm

from titans_pytorch.neuro.dataset import NaturalLanguageDataset
from titans_pytorch.neuro.models import TitanSemanticDecoding

def train_decoding(args):
    print(f"🚀 Task 3: Semantic Decoding Training (Natural Language)")
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

    # Helper to get dims from first sample
    first_sample = dataset[0]
    voxel_dim = first_sample['brain'].shape[1]
    stimulus_dim = first_sample['stimulus'].shape[1]
    
    print(f"   Voxel Dimension: {voxel_dim}")
    print(f"   Stimulus Dimension: {stimulus_dim}")

    # 2. Model
    model = TitanSemanticDecoding(
        voxel_dim=voxel_dim, 
        stimulus_dim=stimulus_dim, 
        hidden_dim=args.hidden_dim
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss() # Predicting MFCC features
    
    # 3. Training Loop
    model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        start_time = time.time()
        
        pbar = tqdm(dataset, desc=f"Epoch {epoch}")
        for sample in pbar:
            brain = sample['brain'].unsqueeze(0).to(device)    # (1, T, Voxels)
            stim = sample['stimulus'].unsqueeze(0).to(device)  # (1, T, Feats)
            
            optimizer.zero_grad()
            
            # Predict stimulus from brain
            pred = model(brain, use_chunked=True, chunk_size=args.chunk_size)
            
            loss = criterion(pred, stim)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.6f}"})
            
        avg_loss = epoch_loss / len(dataset)
        elapsed = time.time() - start_time
        print(f"✨ Epoch {epoch} Done. Avg Loss: {avg_loss:.6f}. Time: {elapsed:.2f}s")
        
        # Checkpoint
        if (epoch + 1) % 5 == 0:
            os.makedirs('checkpoints/decoding', exist_ok=True)
            ckpt_path = f"checkpoints/decoding/titan_decoding_{args.subject_id}_e{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/home/juke/git/ds003020')
    parser.add_argument('--subject_id', type=str, default='UTS01')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--chunk_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    args = parser.parse_args()
    train_decoding(args)

