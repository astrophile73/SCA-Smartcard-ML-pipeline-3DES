
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


class ZaidNetSharedBackbone(nn.Module):
    """
    Multi-key-type ZaidNet with shared backbone and per-key-type heads.
    
    Architecture:
    - Shared backbone: Conv1D layers (extract general power-to-leakage patterns)
    - Per-key-type heads: FC layers (specialized for KENC/KMAC/KDEK)
    
    Gap #2 Implementation: Transfer Learning
    Enables:
    1. Train on KENC with full backbone
    2. Fine-tune on KMAC with frozen backbone (0.0001 LR)
    3. Fine-tune on KDEK with frozen backbone (0.0001 LR)
    → Higher accuracy on KMAC/KDEK due to inherited feature extraction
    """
    def __init__(self, input_dim=1500, num_classes=16):
        super(ZaidNetSharedBackbone, self).__init__()
        
        # SHARED BACKBONE: Conv layers (power-agnostic feature extraction)
        # These layers learn: aligned_power_samples → intermediate_features
        # Shared across all key types to reduce overfitting
        
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
        
        # Calculate flattened size
        final_dim = input_dim // 16
        self._to_linear = 128 * final_dim
        
        # SHARED FC layers (general feature learning)
        self.fc_shared1 = nn.Linear(self._to_linear, 1024)
        self.fc_shared2 = nn.Linear(1024, 512)
        self.dropout_shared = nn.Dropout(0.3)
        
        # KEY-TYPE SPECIFIC HEADS: Each key type gets dedicated classification layer
        # These layers learn: shared_features → S-box_output_for_this_key_type
        self.fc_kenc = nn.Linear(512, num_classes)
        self.fc_kmac = nn.Linear(512, num_classes)
        self.fc_kdek = nn.Linear(512, num_classes)
        
        self.dropout_head = nn.Dropout(0.3)

    def forward_backbone(self, x):
        """Extract shared features from input (used during transfer learning)."""
        if x.ndim == 2:
            x = x.unsqueeze(1)
            
        # Shared CNN backbone
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Shared FC layers
        x = self.dropout_shared(F.relu(self.fc_shared1(x)))
        x = self.dropout_shared(F.relu(self.fc_shared2(x)))
        return x

    def forward(self, x, key_type="kenc"):
        """
        Forward pass with key-type-specific head.
        
        Args:
            x: Input tensor (batch_size, features)
            key_type: One of "kenc", "kmac", "kdek" (default: "kenc")
        
        Returns:
            Logits for 16 S-box classes
        """
        shared_features = self.forward_backbone(x)
        
        # Apply key-type-specific head
        if key_type.lower() == "kenc":
            logits = self.fc_kenc(self.dropout_head(shared_features))
        elif key_type.lower() == "kmac":
            logits = self.fc_kmac(self.dropout_head(shared_features))
        elif key_type.lower() == "kdek":
            logits = self.fc_kdek(self.dropout_head(shared_features))
        else:
            raise ValueError(f"Unknown key_type: {key_type}")
        
        return logits

    def freeze_backbone(self):
        """Freeze shared backbone (for transfer learning fine-tuning)."""
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.conv2.parameters():
            param.requires_grad = False
        for param in self.conv3.parameters():
            param.requires_grad = False
        for param in self.conv4.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for param in self.bn2.parameters():
            param.requires_grad = False
        for param in self.bn3.parameters():
            param.requires_grad = False
        for param in self.bn4.parameters():
            param.requires_grad = False
        for param in self.fc_shared1.parameters():
            param.requires_grad = False
        for param in self.fc_shared2.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze shared backbone (enable full training)."""
        for param in self.parameters():
            param.requires_grad = True


def get_model(input_dim=1500, num_classes=16, use_shared_backbone=False):
    """
    Factory function for model creation.
    
    Args:
        input_dim: Input feature dimension (default: 1500 POIs)
        num_classes: Number of output classes (default: 16 S-box values)
        use_shared_backbone: If True, return ZaidNetSharedBackbone (for transfer learning)
                           If False, return standard ZaidNet (for backward compatibility)
    
    Returns:
        Model instance
    """
    if use_shared_backbone:
        return ZaidNetSharedBackbone(input_dim, num_classes)
    return ZaidNet(input_dim, num_classes)
