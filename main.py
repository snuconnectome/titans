import torch
import torch.nn as nn
import torch.optim as optim
from titans_pytorch.models.mac import MemoryAsContext
from titans_pytorch.models.baseline import StandardTransformer
import random

def generate_associative_data(batch_size, seq_len, vocab_size, num_pairs=2):
    """
    Generate synthetic associative recall task.
    Sequence: [k1, v1, k2, v2, ..., k1, ?]
    Goal: Predict v1 given k1 at the end.
    """
    data = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = data.clone()
    
    # Needs implementation logic, but for simple seq modeling 
    # we can use standard random data and see if models converge aka "memorize".
    # For a stronger test, we enforce structure.
    # [A, B, ..., A] -> Target B
    
    # Let's stick to standard next-token prediction on random data 
    # (checking ability to memorize recent context) first.
    
    # Shift targets for next-token prediction
    inputs = data[:, :-1]
    targets = data[:, 1:]
    return inputs, targets

def train_step(model, optimizer, criterion, x, y):
    optimizer.zero_grad()
    output = model(x)
    
    # Reshape for loss
    # output: (B, N, V), y: (B, N)
    loss = criterion(output.transpose(1, 2), y)
    loss.backward()
    optimizer.step()
    return loss.item()

def run_experiment():
    dim = 64
    vocab_size = 128
    batch_size = 16
    seq_len = 64
    steps = 100
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Models
    mac = MemoryAsContext(dim, vocab_size, num_heads=4, segment_len=16).to(device)
    transformer = StandardTransformer(dim, vocab_size, num_heads=4).to(device)
    
    opt_mac = optim.AdamW(mac.parameters(), lr=1e-3)
    opt_tf = optim.AdamW(transformer.parameters(), lr=1e-3)
    
    criterion = nn.CrossEntropyLoss()
    
    print("Starting Training...")
    for i in range(steps):
        x, y = generate_associative_data(batch_size, seq_len, vocab_size)
        x, y = x.to(device), y.to(device)
        
        loss_mac = train_step(mac, opt_mac, criterion, x, y)
        loss_tf = train_step(transformer, opt_tf, criterion, x, y)
        
        if i % 10 == 0:
            print(f"Step {i}: MAC Loss: {loss_mac:.4f}, TF Loss: {loss_tf:.4f}")
            
    print("Experiment Complete.")

if __name__ == "__main__":
    run_experiment()
