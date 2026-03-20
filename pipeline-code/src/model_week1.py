"""
Updated 3DES Attack Model - WEEK 1 FIX

Changes from original:
1. Support for configurable num_classes (64 for S-Box inputs)
2. Added class weighting for imbalanced datasets
3. Optimized for S-Box INPUT prediction
4. Better regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASCADModel(nn.Module):
    """
    ASCAD CNN Architecture with 64-class output for S-Box inputs.
    """
    def __init__(self, input_dim=500, num_classes=64):
        super(ASCADModel, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=11, stride=1, padding=5)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=11, stride=1, padding=5)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=11, stride=1, padding=5)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv1d(256, 512, kernel_size=11, stride=1, padding=5)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # Calculate flattened size
        dummy_x = torch.zeros(1, 1, input_dim)
        with torch.no_grad():
            x = self.pool1(torch.relu(self.bn1(self.conv1(dummy_x))))
            x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
            x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
            self._to_linear = x.numel()
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(1024, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.5)
        
        # Output layer
        self.fc3 = nn.Linear(1024, num_classes)
        
        # Spatial Dropout
        self.drop_conv = nn.Dropout(p=0.3)
        
    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        
        # Conv blocks
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.drop_conv(x)
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        x = self.drop_conv(x)
        
        # Flatten
        x = x.view(-1, self._to_linear)
        
        # FC blocks
        x = self.dropout1(F.relu(self.bn5(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn6(self.fc2(x))))
        x = self.fc3(x)
        
        return x


def get_model(input_dim=500, num_classes=64):
    """Create ASCAD model"""
    return ASCADModel(input_dim, num_classes)


def compute_class_weights(labels, num_classes=64):
    """Compute class weights for imbalanced data"""
    import numpy as np
    
    unique, counts = np.unique(labels, return_counts=True)
    weights = np.ones(num_classes)
    weights[unique] = 1.0 / counts
    weights = weights / weights.mean()
    
    return torch.from_numpy(weights).float()
