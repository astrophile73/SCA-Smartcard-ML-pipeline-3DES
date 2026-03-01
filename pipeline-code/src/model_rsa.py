"""
RSA Model Architecture for Side-Channel Analysis
Predicts RSA CRT components byte-by-byte
"""
import torch
import torch.nn as nn

class RSABytePredictor(nn.Module):
    """
    Predicts a single byte (0-255) of an RSA component
    """
    def __init__(self, input_dim=200):
        super(RSABytePredictor, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Output: 256 classes (one per byte value)
        self.classifier = nn.Linear(256, 256)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)


class RSAComponentModel(nn.Module):
    """
    Predicts all 128 bytes of an RSA CRT component
    Uses 128 separate byte predictors
    """
    def __init__(self, input_dim=200, num_bytes=128):
        super(RSAComponentModel, self).__init__()
        
        self.num_bytes = num_bytes
        
        # Shared feature extraction
        self.shared_features = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Separate head for each byte
        self.byte_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256)  # 256 classes per byte
            )
            for _ in range(num_bytes)
        ])
        
    def forward(self, x):
        # Shared features
        features = self.shared_features(x)
        
        # Predict each byte
        outputs = []
        for byte_predictor in self.byte_predictors:
            byte_output = byte_predictor(features)
            outputs.append(byte_output)
        
        return outputs  # List of 128 tensors, each (batch_size, 256)


def create_rsa_model(input_size=200, num_bytes=128):
    """
    Factory function to create RSA model
    """
    return RSAComponentModel(input_dim=input_size, num_bytes=num_bytes)


if __name__ == "__main__":
    # Test
    model = create_rsa_model(input_size=200, num_bytes=128)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    x = torch.randn(4, 200)
    outputs = model(x)
    print(f"Output: {len(outputs)} byte predictions")
    print(f"Each prediction shape: {outputs[0].shape}")
