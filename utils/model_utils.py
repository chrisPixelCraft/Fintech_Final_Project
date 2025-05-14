import torch
import os

def save_checkpoint(state, filename: str = 'checkpoint.pth'):
    """Save training checkpoint"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename: str):
    """Load training checkpoint"""
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
    return model, optimizer
