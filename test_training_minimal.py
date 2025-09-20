#!/usr/bin/env python3
"""
Minimal test to isolate training issues
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Create a MUCH simpler model to test training
class SimpleChessNet(nn.Module):
    def __init__(self):
        super(SimpleChessNet, self).__init__()
        # Just 3 layers instead of 30!
        self.conv1 = nn.Conv2d(119, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_linear = nn.Linear(2 * 8 * 8, 4672)
        
        # Value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_linear = nn.Linear(1 * 8 * 8, 1)
        
    def forward(self, x):
        # Simple forward pass
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Policy head
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_linear(policy)
        
        # Value head
        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value_pred = torch.tanh(self.value_linear(value))
        
        return policy_logits, value_pred

# Simple dataset
class SimpleDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Random board input
        board_tensor = torch.randn(119, 8, 8)
        
        # Random target policy (20 legal moves)
        target_policy = torch.zeros(4672)
        legal_indices = torch.randperm(4672)[:20]
        target_policy[legal_indices] = 1.0 / 20
        
        # Legal mask
        legal_mask = torch.zeros(4672, dtype=torch.bool)
        legal_mask[legal_indices] = True
        
        # Random target value
        target_value = torch.tensor([np.random.choice([-1.0, 1.0])], dtype=torch.float32)
        
        return board_tensor, target_policy, legal_mask, target_value

def test_simple_training():
    print("=== MINIMAL TRAINING TEST ===")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleChessNet().to(device)
    dataset = SimpleDataset(100)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    for epoch in range(5):
        total_loss = 0
        total_policy_loss = 0 
        total_value_loss = 0
        num_batches = 0
        
        for batch_idx, (board_tensors, target_policies, legal_masks, target_values) in enumerate(loader):
            board_tensors = board_tensors.to(device)
            target_policies = target_policies.to(device)
            legal_masks = legal_masks.to(device)
            target_values = target_values.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            policy_logits, value_pred = model(board_tensors)
            
            # Loss computation (same as training script)
            masked_logits = policy_logits.clone()
            masked_logits[~legal_masks] = -100.0
            
            log_probs = F.log_softmax(masked_logits, dim=1)
            normalized_target = target_policies / (target_policies.sum(dim=1, keepdim=True) + 1e-8)
            policy_loss = -(normalized_target * log_probs).sum(dim=1).mean()
            
            value_loss = F.mse_loss(value_pred.squeeze(), target_values.squeeze())
            total_loss_batch = policy_loss + value_loss
            
            # Backward pass
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_policy = total_policy_loss / num_batches
        avg_value = total_value_loss / num_batches
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Policy={avg_policy:.4f}, Value={avg_value:.4f}")

if __name__ == "__main__":
    test_simple_training()