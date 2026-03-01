
import torch
import torch.nn as nn
import torch.nn.functional as F

class ZaidNet(nn.Module):
    """
    ZaidNet Architecture for Side-Channel Analysis.
    Based on: "Methodology for efficient CNN architectures in SCA" (Zaid et al., TCHES 2020)
    https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA
    
    Optimized for high accuracy (>99%) with fewer parameters than ASCAD.
    """
    def __init__(self, input_dim=1500, num_classes=16):
        super(ZaidNet, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2 = nn.Conv1d(16, 32, kernel_size=11, stride=1, padding=5)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3 = nn.Conv1d(32, 64, kernel_size=11, stride=1, padding=5)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # Block 4
        self.conv4 = nn.Conv1d(64, 128, kernel_size=11, stride=1, padding=5)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # Flatten
        # Calculate size dynamically based on input_dim
        # 1500 -> 750 -> 375 -> 187 -> 93
        final_dim = input_dim // 16 
        self._to_linear = 128 * final_dim 
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x shape: (Batch, features)
        if x.ndim == 2:
            x = x.unsqueeze(1)
            
        # CNN Blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # MLP Blocks
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

def get_model(input_dim=1500, num_classes=16):
    return ZaidNet(input_dim, num_classes)
