# PenuX_small.py
# Updated lightweight version for small dataset (e.g. 57 samples)
# Author: Netanel Stern
# Date: May 2026

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

class RationalPolynomialNeuron(nn.Module):
    """Novel RPN activation function"""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.1))
        self.b = nn.Parameter(torch.tensor(0.1))
        self.lambda_ = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x):
        # n(a,b,λ,z) = λ(z + a z³) / (1 + b z²)
        numerator = self.lambda_ * (x + self.a * x ** 3)
        denominator = 1 + self.b * x ** 2
        return numerator / denominator


class PenuX_Small(nn.Module):
    """
    Lightweight PenuX model optimized for small datasets
    """
    def __init__(self, input_size=45, num_classes=3, dropout_rate=0.5):
        super().__init__()
        
        # Bi-LSTM Encoder (reduced size)
        self.bi_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # Rational Polynomial Neuron
        self.rpn = RationalPolynomialNeuron()
        
        # Refinement LSTM (further reduced)
        self.refine_lstm = nn.LSTM(
            input_size=128,  # 64 * 2 (bidirectional)
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            dropout=0.4
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        bi_out, _ = self.bi_lstm(x)
        bi_out = self.rpn(bi_out)
        
        refine_out, _ = self.refine_lstm(bi_out)
        # Use last hidden state
        last_hidden = refine_out[:, -1, :]
        
        logits = self.classifier(last_hidden)
        return logits


# ====================== Training Setup ======================

def get_model_and_optimizer():
    model = PenuX_Small(input_size=45, num_classes=3, dropout_rate=0.5)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=5e-4, 
        weight_decay=1e-3
    )
    
    # Focal Loss
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.5, alpha=None):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha
        
        def forward(self, inputs, targets):
            ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            return focal_loss.mean()
    
    criterion = FocalLoss(gamma=2.5)
    
    return model, optimizer, criterion


# Example usage
if __name__ == "__main__":
    print("PenuX_Small model initialized successfully!")
    print("Total parameters:", 
          sum(p.numel() for p in PenuX_Small().parameters() if p.requires_grad))
    
    # Example dummy data (replace with real data)
    # X: (samples, time_steps, features), y: (samples,)
    # model, optimizer, criterion = get_model_and_optimizer()
