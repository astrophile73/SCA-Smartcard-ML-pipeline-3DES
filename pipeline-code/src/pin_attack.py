
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utils import setup_logger

logger = setup_logger("pin_attack")

def run_pin_attack():
    # 1. Load Data
    # Priority 1: Check "Test Trace with Verify Command (Visa Card)" directory
    data_dir = "Test Trace with Verify Command (Visa Card)"
    npz_file = "verify_trace.npz"
    
    traces = None
    verify_cmd = None
    
    # Try Directory first
    if os.path.exists(data_dir):
        logger.info(f"Loading data from directory: {data_dir}...")
        try:
            t_path = os.path.join(data_dir, "trace_data.npy")
            v_path = os.path.join(data_dir, "Verify_command.npy")
            if os.path.exists(t_path) and os.path.exists(v_path):
                traces = np.load(t_path)
                verify_cmd = np.load(v_path)
                logger.info(f"Loaded from directory. Traces: {traces.shape}")
        except Exception as e:
            logger.error(f"Failed to load from directory: {e}")

    # Try NPZ file if Directory failed
    if traces is None and os.path.exists(npz_file):
        logger.info(f"Loading data from file: {npz_file}...")
        try:
            data = np.load(npz_file)
            if 'trace_data' in data and 'Verify_command' in data:
                traces = data['trace_data']
                verify_cmd = data['Verify_command']
                logger.info(f"Loaded from NPZ. Traces: {traces.shape}")
            else:
                logger.warning("NPZ missing required keys.")
        except Exception as e:
            logger.error(f"Failed to load from NPZ: {e}")

    if traces is None or verify_cmd is None:
        logger.error("Could not load traces or verify commands from any source.")
        return
    
    # Handle 0-d array (scalar)
    if verify_cmd.ndim == 0:
        logger.info("Verify_command is a scalar (0-d array). Wrapping in list.")
        verify_cmd = np.array([verify_cmd.item()])
        
    if len(verify_cmd) == 0:
        logger.error("Verify_command.npy is empty!")
        return

    # Inspect first command
    sample = verify_cmd[0]
    if isinstance(sample, (bytes, np.bytes_)):
        logger.info(f"Sample Command: {sample.hex()}")
    else:
        logger.info(f"Sample Command (raw): {sample}")

    # Check for variance
    logger.info("Checking for data variance...")
    
    # If scalar or 1 element, variance is 0
    if len(verify_cmd) <= 1:
        logger.warning("Only 1 Verify Command found. Variance is 0.")
    else:
        unique_cmds = np.unique(verify_cmd, axis=0)
        logger.info(f"Unique Commands: {len(unique_cmds)}")
        if len(unique_cmds) == 1:
            logger.warning("⚠️ DATASET HAS CONSTANT PIN INPUT!")

    # Plot average trace
    avg_trace = np.mean(traces, axis=0)
    
    # Save plot
    if not os.path.exists("Analysis"):
        os.makedirs("Analysis")
        
    plt.figure(figsize=(12, 6))
    plt.plot(avg_trace)
    plt.title("Average Power Trace - Verify Command")
    plt.xlabel("Sample")
    plt.ylabel("Power")
    plt.savefig("Analysis/verify_spa_trace.png")
    logger.info("Saved SPA trace to Analysis/verify_spa_trace.png")

if __name__ == "__main__":
    run_pin_attack()
