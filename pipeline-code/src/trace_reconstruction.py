"""
Gap #3: Trace Reconstruction Defense

Defensive mechanism to detect spoofed/synthetic power traces.

Theory: Given a recovered key, reconstruct what the power trace should look like
if it came from authentic hardware. Compare with original trace to detect spoofing.

Two approaches:
1. Autoencoder/VAE: Learn compressed representation during profiling
   - Encode original trace → latent space
   - Decode latent → reconstructed trace
   - Metric: MSE(original, reconstructed) indicates authenticity
   
2. Gradient-based inversion: Find trace that maximizes model confidence for key
   - Given recovered key k, optimize trace t to maximize P(k|t)
   - If P(k|t) much higher for synthetic t than genuine t, indicates spoofing

Implementation: Foundation classes for both approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from src.utils import setup_logger

logger = setup_logger("trace_reconstruction")


class TraceAutoencoder(nn.Module):
    """
    Autoencoder for trace reconstruction and anomaly detection.
    
    Architecture:
    - Encoder: Conv1D layers → latent bottleneck (64-dim)
    - Decoder: Transpose Conv1D → reconstruct original trace
    
    Training: Learn to reconstruct genuine traces
    Testing: High reconstruction error → probably spoofed trace
    """
    
    def __init__(self, input_dim=1500, latent_dim=64):
        super(TraceAutoencoder, self).__init__()
        
        # ENCODER: Compress trace to latent representation
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        
        # Calculate size after encoding
        # 1500 -> 750 -> 375 -> 187
        encoded_size = input_dim // 8
        self.fc_encode = nn.Linear(64 * encoded_size, latent_dim)
        
        # DECODER: Reconstruct trace from latent
        self.fc_decode = nn.Linear(latent_dim, 64 * encoded_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=11, stride=2, padding=5, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=11, stride=2, padding=5, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=11, stride=2, padding=5, output_padding=1),
        )
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoded_size = encoded_size

    def encode(self, x):
        """Compress trace to latent space."""
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        latent = self.fc_encode(x)
        return latent

    def decode(self, latent):
        """Reconstruct trace from latent."""
        x = self.fc_decode(latent)
        x = x.view(x.size(0), 64, self.encoded_size)
        x = self.decoder(x)
        return x

    def forward(self, x):
        """Full autoencoder: compress and reconstruct."""
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        # Trim to input dimension if needed
        if reconstructed.shape[-1] > self.input_dim:
            reconstructed = reconstructed[..., :self.input_dim]
        return reconstructed, latent


class TraceReconstructionDetector:
    """
    Detector for spoofed/synthetic power traces.
    
    Usage:
    1. Train on genuine traces: detector.train(genuine_traces)
    2. Test on unknown traces: authenticity_score = detector.detect(test_trace)
    
    Authenticity score ∈ [0, 1]:
    - High (>0.8): Likely genuine (low reconstruction error)
    - Medium (0.5-0.8): Uncertain (moderate reconstruction error)
    - Low (<0.5): Likely spoofed (high reconstruction error)
    """
    
    def __init__(self, input_dim=1500, latent_dim=64, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.autoencoder = TraceAutoencoder(input_dim, latent_dim).to(self.device)
        self.input_dim = input_dim
        
        # Calibration: store reconstruction errors from training data for thresholding
        self.train_errors = []
        self.error_threshold_95 = None  # 95th percentile of training errors
        self.error_mean = None
        self.error_std = None

    def train(self, traces, epochs=20, batch_size=32, learning_rate=0.001):
        """
        Train autoencoder on genuine traces.
        
        Args:
            traces: (n_traces, n_samples) genuine power traces
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Adam learning rate
        """
        logger.info(f"Training trace reconstruction detector on {len(traces)} traces...")
        
        X = torch.from_numpy(traces).float().to(self.device)
        if X.ndim == 2:
            X = X.unsqueeze(1)
        
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                
                reconstructed, _ = self.autoencoder(batch)
                loss = criterion(reconstructed, batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / (len(X) // batch_size):.4f}")
        
        # Calibrate thresholds on training data
        self.autoencoder.eval()
        with torch.no_grad():
            errors = []
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                reconstructed, _ = self.autoencoder(batch)
                mse = F.mse_loss(reconstructed, batch, reduction='none').mean(dim=(1, 2))
                errors.extend(mse.cpu().numpy())
        
        self.train_errors = np.array(errors)
        self.error_threshold_95 = np.percentile(self.train_errors, 95)
        self.error_mean = np.mean(self.train_errors)
        self.error_std = np.std(self.train_errors)
        
        logger.info(f"Autoencoder training complete. Calibration error threshold: {self.error_threshold_95:.4f}")

    def detect(self, trace) -> Tuple[float, dict]:
        """
        Analyze trace authenticity.
        
        Args:
            trace: (n_samples,) single power trace
        
        Returns:
            authenticity_score: ∈ [0, 1], higher = more likely genuine
            metadata: Dict with error statistics for debugging
        """
        # Ensure detector is trained
        assert self.error_mean is not None and self.error_std is not None, \
            "Detector not trained. Call train() before detect()."
        
        X = torch.from_numpy(trace).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        self.autoencoder.eval()
        with torch.no_grad():
            reconstructed, latent = self.autoencoder(X)
            mse = F.mse_loss(reconstructed, X).item()
        
        # Normalize error to [0, 1] based on training calibration
        if self.error_std > 1e-6:
            z_score = (mse - self.error_mean) / self.error_std
        else:
            z_score = 0
        
        # Convert z-score to authenticity probability [0, 1]
        # Assumes normal distribution: z = -2 → ~97% probability genuine
        from scipy.stats import norm
        authenticity = norm.cdf(-z_score)  # Lower error → higher authenticity
        authenticity = np.clip(authenticity, 0, 1)
        
        metadata = {
            "reconstruction_error": mse,
            "error_mean": self.error_mean,
            "error_std": self.error_std,
            "error_threshold_95": self.error_threshold_95,
            "z_score": z_score,
            "authenticity_score": float(authenticity),
        }
        
        return float(authenticity), metadata

    def save(self, path):
        """Save trained autoencoder and calibration data."""
        torch.save({
            "autoencoder_state": self.autoencoder.state_dict(),
            "error_mean": self.error_mean,
            "error_std": self.error_std,
            "error_threshold_95": self.error_threshold_95,
        }, path)
        logger.info(f"Saved trace reconstruction detector to {path}")

    def load(self, path):
        """Load trained autoencoder and calibration data."""
        checkpoint = torch.load(path, map_location=self.device)
        self.autoencoder.load_state_dict(checkpoint["autoencoder_state"])
        self.error_mean = checkpoint["error_mean"]
        self.error_std = checkpoint["error_std"]
        self.error_threshold_95 = checkpoint["error_threshold_95"]
        logger.info(f"Loaded trace reconstruction detector from {path}")


def compute_trace_authenticity_scores(test_traces, detector, batch_size=32) -> np.ndarray:
    """
    Compute authenticity scores for a batch of traces.
    
    Args:
        test_traces: (n_traces, n_samples) test traces
        detector: Trained TraceReconstructionDetector instance
        batch_size: Process in batches for memory efficiency
    
    Returns:
        authenticity_scores: (n_traces,) scores in [0, 1]
    """
    scores = []
    for i in range(0, len(test_traces), batch_size):
        batch = test_traces[i:i+batch_size]
        for trace in batch:
            score, _ = detector.detect(trace)
            scores.append(score)
    
    return np.array(scores)


def flag_suspicious_traces(test_traces, detector, authenticity_threshold=0.6) -> np.ndarray:
    """
    Identify traces that may be spoofed based on reconstruction error.
    
    Args:
        test_traces: (n_traces, n_samples) test traces
        detector: Trained TraceReconstructionDetector instance
        authenticity_threshold: Traces with score < threshold flagged as suspicious
    
    Returns:
        suspicious_flags: (n_traces,) boolean array, True = suspicious
    """
    scores = compute_trace_authenticity_scores(test_traces, detector)
    return scores < authenticity_threshold
