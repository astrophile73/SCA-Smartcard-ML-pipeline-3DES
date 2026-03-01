import numpy as np
from scipy.signal import correlate
from src.utils import setup_logger

logger = setup_logger("preprocess")

def normalize_traces(traces: np.ndarray) -> np.ndarray:
    """
    Normalize traces (Z-score normalization per trace).
    """
    mean = np.mean(traces, axis=1, keepdims=True)
    std = np.std(traces, axis=1, keepdims=True)
    # Avoid div by zero
    std[std == 0] = 1.0
    return (traces - mean) / std

def align_traces(traces: np.ndarray, reference_idx=0, max_shift=100, reference_trace=None) -> np.ndarray:
    """
    Align traces using cross-correlation using explicit reference or index.
    """
    if reference_trace is not None:
        ref_trace = reference_trace
    else:
        ref_trace = traces[reference_idx]
        
    aligned_traces = np.zeros_like(traces, dtype=np.float32)
    # If using internal index, copy it first. If external, we align everything.
    if reference_trace is None:
        aligned_traces[reference_idx] = ref_trace
    
    n_samples = traces.shape[1]
    
    logger.info("Aligning traces using FFT-based Cross-Correlation...")
    try:
        from scipy.signal import fftconvolve
        
        # Optimization: Use a smaller window for alignment to speed up
        # 131k samples is too big for full convolution.
        # We focus on the first 10,000 samples where the trigger/op usually starts.
        align_window = min(10000, n_samples)
        ref_segment = ref_trace[:align_window]
        
        # Reverse reference for convolution -> correlation
        ref_segment_rev = ref_segment[::-1]
        
        for i in range(len(traces)):
            if reference_trace is None and i == reference_idx:
                aligned_traces[i] = traces[i]
                continue
            
            trace_segment = traces[i][:align_window]
            
            # Cross-correlate
            corr = fftconvolve(trace_segment, ref_segment_rev, mode='same')
            
            # Find peak
            peak_idx = np.argmax(corr)
            
            # Calculate shift
            # In 'same' mode, the peak for zero shift is at center
            center = len(corr) // 2
            shift = peak_idx - center
            
            # Apply shift
            if abs(shift) > max_shift:
                # If shift is huge, it might be a false positive or just too far.
                # We clamp or skip. Let's clamp.
                shift = int(np.sign(shift) * max_shift)
                
            # Shift the FULL trace
            # Positive shift means trace was "late" (peak to the right), so we shift left?
            # Standard correlation: if peak is at lag L, then f(t) ~ g(t+L).
            if shift != 0:
                aligned_traces[i] = np.roll(traces[i], -shift)
            else:
                aligned_traces[i] = traces[i]

            if i % 100 == 0:
                logger.debug(f"Aligned trace {i} with shift {shift}")
                
        return aligned_traces
    except ImportError:
        logger.warning("scipy not found, skipping alignment.")
        return traces
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        return traces


def extract_window(traces: np.ndarray, start: int, end: int) -> np.ndarray:
    if start < 0: start = 0
    if end > traces.shape[1]: end = traces.shape[1]
    return traces[:, start:end]
