import torch
import torch.nn as nn
import torch.nn.functional as F

class ASCADModel(nn.Module):
    """
    ASCAD CNN Architecture adapted for 3DES Side-Channel Analysis.
    Based on: https://github.com/ANSSI-FR/ASCAD
    Paper: https://eprint.iacr.org/2018/053.pdf
    
    Key differences from original ASCAD:
    - Input: ~500 POIs instead of 700 raw samples
    - Output: 16 classes (S-Box outputs) instead of 256 (AES byte values)
    - Slightly reduced capacity to prevent overfitting on smaller dataset
    """
    def __init__(self, input_dim=500, num_classes=16):
        super(ASCADModel, self).__init__()
        
        # Convolutional Layers (ASCAD-style)
        # Using kernel_size=11 to capture wider temporal patterns in power traces
        # AveragePooling is better for SCA than MaxPooling (preserves signal amplitude)
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=11, stride=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=11, stride=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=11, stride=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv1d(256, 512, kernel_size=11, stride=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # Calculate flattened size dynamically
        dummy_x = torch.zeros(1, 1, input_dim)
        with torch.no_grad():
            dummy_out = self.pool4(self.bn4(self.conv4(self.pool3(self.bn3(self.conv3(self.pool2(self.bn2(self.conv2(self.pool1(self.bn1(self.conv1(dummy_x))))))))))))
            self._to_linear = dummy_out.numel()
        
        # Fully Connected Layers (ASCAD-style)
        # Deep FC layers with heavy dropout for regularization
        # REDUCED CAPACITY: 2048 -> 1024 to prevent overfitting on 5k traces
        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5) 
        
        self.fc2 = nn.Linear(1024, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(1024, num_classes)
        
        # New: Spatial Dropout for CNN features
        self.drop_conv = nn.Dropout(p=0.3)
        
    def forward(self, x):
        # x shape: (Batch, features)
        # Reshape for Conv1d: (Batch, Channels=1, Length)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        
        # Convolutional blocks
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.drop_conv(x) # Regularization
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        x = self.drop_conv(x) # Regularization
        
        # Flatten
        x = x.view(-1, self._to_linear)
        
        # Fully connected blocks with dropout
        x = self.dropout1(F.relu(self.bn5(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn6(self.fc2(x))))
        x = self.fc3(x)  # Logits (no activation, CrossEntropyLoss handles softmax)
        
        return x

def get_model(input_dim=500, num_classes=16):
    """
    Factory function to create ASCAD-style model.
    
    Args:
        input_dim: Number of POIs (features) per trace
        num_classes: Number of output classes (16 for S-Box)
    
    Returns:
        ASCADModel instance
    """
    return ASCADModel(input_dim, num_classes)

