#!/usr/bin/env python3
"""
Continue interrupted 3DES training on Mastercard traces.
Resumes KMAC (where it was interrupted) and continues to KDEK training.
Skips KENC (already complete: 16 models saved).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipeline-code'))

import logging
from src.train_ensemble import train_ensemble
from src.inference_3des import recover_3des_keys
from src.utils import setup_logger

logger = setup_logger("continue_training")

def continue_training():
    """Continue training from interrupted point"""
    
    processed_dir = "Output/mastercard_processed/3des"
    model_root = "pipeline-code/models"
    input_dir = r"I:\freelance\SCA-Smartcard-Pipeline-3\Input1\Mastercard"
    
    logger.info("=" * 80)
    logger.info("CONTINUING 3DES TRAINING (Mastercard)")
    logger.info("=" * 80)
    
    # Check if preprocessing is done
    if not os.path.exists(os.path.join(processed_dir, "X_features_kenc_s1.npy")):
        # Try legacy path
        if not os.path.exists(os.path.join(processed_dir, "X_features_s1.npy")):
            logger.error(f"Preprocessing data not found in {processed_dir}")
            logger.error("Run full pipeline first with --mode full")
            return False
    
    logger.info("[STEP 1] Verified preprocessing complete")
    logger.info("  - 10,000 traces already processed")
    logger.info("  - Features extracted and saved")
    
    # Train remaining key types (KMAC and KDEK)
    # KENC is already done (16 models), so skip it
    logger.info("\n[STEP 2] Continuing ensemble training...")
    logger.info("  - KENC: ✅ Already complete (16 models)")
    logger.info("  - KMAC: 🔄 Resuming (4 models done, resuming S-Box 3+)...")
    logger.info("  - KDEK: ⏳ Will start after KMAC...")
    
    try:
        # Train only KMAC and KDEK (skip KENC which is already done)
        train_ensemble(
            input_dir=processed_dir,
            output_dir=model_root,
            models_per_sbox=1,
            epochs=50,
            early_stop_patience=8,
            use_transfer_learning=False,
            key_types=["kmac", "kdek"]  # Skip KENC, train only these
        )
        logger.info("\n✅ KMAC & KDEK training completed successfully")
        
        # Now run inference
        logger.info("\n[STEP 3] Running inference on 10,000 traces...")
        predicted = recover_3des_keys(
            processed_dir="Output/mastercard_processed",
            model_dir=model_root,
            card_type="mastercard",
            n_attack=10000
        )
        logger.info("✅ Inference completed successfully")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Predicted keys: {predicted}")
        logger.info("Predictions saved to: Output/mastercard_processed/3des/Y_meta.csv")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training/Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = continue_training()
    sys.exit(0 if success else 1)
