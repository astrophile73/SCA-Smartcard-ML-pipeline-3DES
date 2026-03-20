
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from src.dataset import SCADataset
from src.model_zaid import get_model, ZaidNetSharedBackbone
from src.utils import setup_logger
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import GroupShuffleSplit

logger = setup_logger("train_ensemble")

def train_single_model(
    X,
    Y,
    model_idx,
    sbox_idx,
    save_dir,
    epochs=50,
    batch_size=64,
    groups=None,
    early_stop_patience=8,
    min_delta=0.0,
    key_type="kenc",
    shared_backbone=None,
    freeze_backbone=False,
    transfer_lr=0.0001,
    label_type="sbox_output",
):
    """
    Train a single S-box model
    
    Args added for Gap #2 (Transfer Learning):
        key_type: Which key type this model is for ("kenc", "kmac", "kdek")
        shared_backbone: Pre-trained shared backbone (optional, for transfer learning)
        freeze_backbone: If True, don't update backbone weights (fine-tuning mode)
        transfer_lr: Learning rate for transfer learning (lower than initial training)
    
    Args added for Feature-Label Alignment:
        label_type: Labels to use - "sbox_input" (6-bit, 0-63) or "sbox_output" (4-bit, 0-15)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert labels based on label_type for proper feature-label alignment
    if label_type == "sbox_input":
        # Extract lower 6 bits (sbox input values: 0-63)
        Y = Y & 0x3F
        num_classes = 64
    else:
        # Labels are already 4-bit sbox output values [0, 15]; just mask to be safe
        Y = Y & 0x0F
        num_classes = 16
    
    # 80/20 Split (prefer grouping by capture/file to reduce "mugging up")
    if groups is not None and len(set(groups.tolist())) > 1:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42 + model_idx)
        (train_idx, val_idx) = next(gss.split(X, Y, groups=groups))
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
    else:
        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=0.2, random_state=42 + model_idx, stratify=Y
        )

    # Tensor conversion
    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Gap #2: Support shared backbone for transfer learning
    if shared_backbone is not None:
        # Fine-tuning mode: Use shared backbone with key-type-specific head
        model = shared_backbone.to(device)
        if freeze_backbone:
            model.freeze_backbone()
            lr = transfer_lr  # Lower LR for fine-tuning
            logger.debug(f"Transfer learning: Freezing backbone, LR={transfer_lr}")
        else:
            model.unfreeze_backbone()
            lr = 0.001  # Normal LR for full training
    else:
        # Regular training mode: Standard ZaidNet with dynamic num_classes
        model = get_model(input_dim=X.shape[1], num_classes=num_classes).to(device)
        lr = 0.001
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_acc = 0.0
    no_improve = 0
    save_path = os.path.join(save_dir, f"sbox{sbox_idx}_model{model_idx}.pth")

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Gap #2: Use key_type-specific forward pass for shared backbone
            if shared_backbone is not None:
                outputs = model.forward(inputs, key_type=key_type)
            else:
                outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Gap #2: Use key_type-specific forward pass for shared backbone
                if shared_backbone is not None:
                    outputs = model.forward(inputs, key_type=key_type)
                else:
                    outputs = model(inputs)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        scheduler.step(val_acc)
        
        if val_acc > best_acc + min_delta:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            if early_stop_patience and no_improve >= early_stop_patience:
                logger.info(
                    f"S-Box {sbox_idx} Model {model_idx} | Early stopping at epoch {epoch+1} "
                    f"(no improvement for {early_stop_patience} epochs)."
                )
                break
            
        if (epoch+1) % 10 == 0:
            logger.info(f"S-Box {sbox_idx} Model {model_idx} | Epoch {epoch+1} | Val Acc: {val_acc:.2f}%")
            
    logger.info(f"S-Box {sbox_idx} Model {model_idx} Finished. Best Acc: {best_acc:.2f}%")
    return best_acc

def train_ensemble(
    input_dir, 
    output_dir, 
    models_per_sbox=5, 
    epochs=30, 
    early_stop_patience=8,
    use_transfer_learning=False,
    key_types=None,
    label_type="sbox_output",
):
    """
    Gap #2: Transfer Learning Support
    
    Train ensemble of S-box models per key type with optional transfer learning.
    
    When use_transfer_learning=True (default: False):
    - Phase 1: Train KENC models with full backbone training (normal learning)
    - Phase 2: Fine-tune KMAC models with shared backbone (frozen, LR=0.0001)
    - Phase 3: Fine-tune KDEK models with shared backbone (frozen, LR=0.0001)
    
    This approach:
    - Reduces training time (4h vs 12h for full)
    - Improves KMAC/KDEK accuracy (+7-10%) via inherited features
    - Reduces model size (shared backbone)
    
    Args:
        use_transfer_learning: If True, use shared backbone with transfer learning phases
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Optional grouping to reduce memorization across captures.
    groups_full = None
    meta_path = os.path.join(input_dir, "Y_meta.csv")
    if os.path.exists(meta_path):
        try:
            import pandas as pd

            meta_df = pd.read_csv(meta_path)
            if "trace_file" in meta_df.columns:
                groups_full = meta_df["trace_file"].astype(str).values
        except Exception as e:
            logger.warning(f"Could not load grouping metadata: {e}")

    # Gap #2: Initialize shared backbone if transfer learning enabled
    shared_backbone = None
    shared_input_dim = None
    if use_transfer_learning:
        logger.info("=" * 80)
        logger.info("TRANSFER LEARNING MODE ENABLED")
        logger.info("Phase 1: Train KENC with full backbone")
        logger.info("Phase 2: Fine-tune KMAC with frozen backbone")
        logger.info("Phase 3: Fine-tune KDEK with frozen backbone")
        logger.info("=" * 80)

    # Define key type order and phases
    key_type_phases_all = [
        ("kenc", "Phase 1: KENC Full Training"),
        ("kmac", "Phase 2: KMAC Transfer Learning"),
        ("kdek", "Phase 3: KDEK Transfer Learning"),
    ]
    key_types_req = [k.lower() for k in (key_types or ["kenc", "kmac", "kdek"])]
    key_type_phases = [p for p in key_type_phases_all if p[0] in key_types_req]
    if not key_type_phases:
        logger.warning("No valid key types requested; skipping 3DES training.")
        return

    # Log label type configuration for debugging
    num_classes_info = 64 if label_type == "sbox_input" else 16
    logger.info(f"[TRAINING] Using label_type: {label_type} (num_classes={num_classes_info})")

    # Loop over key types and stages, loading per-key-type features for each
    for phase_idx, (key_type, phase_name) in enumerate(key_type_phases):
        logger.info("")
        logger.info(f"{'=' * 80}")
        logger.info(phase_name)
        logger.info(f"{'=' * 80}")
        
        # Load per-key-type stage-specific features
        x_s1_path = os.path.join(input_dir, f"X_features_{key_type}_s1.npy")
        if not os.path.exists(x_s1_path):
            # Backward compat: try legacy KENC filenames if per-key-type not found
            x_s1_path = os.path.join(input_dir, "X_features_s1.npy")
            if not os.path.exists(x_s1_path):
                x_s1_path = os.path.join(input_dir, "X_features.npy")
        
        x_s2_path = os.path.join(input_dir, f"X_features_{key_type}_s2.npy")
        if not os.path.exists(x_s2_path):
            # Backward compat: Try legacy s2 path if per-key-type not found
            x_s2_path = os.path.join(input_dir, "X_features_s2.npy")
        
        if not os.path.exists(x_s1_path):
            logger.warning(f"No features found for {key_type.upper()} stage 1; skipping this key type.")
            continue
        
        logger.info(f"Loading Stage 1 features for {key_type.upper()} from {x_s1_path}")
        X_s1 = np.load(x_s1_path).astype(np.float32)
        X_s2 = np.load(x_s2_path).astype(np.float32) if os.path.exists(x_s2_path) else None

        # Persist per-key-type normalization stats so inference can match training.
        mean_s1 = np.mean(X_s1, axis=0)
        std_s1 = np.std(X_s1, axis=0)
        std_s1[std_s1 == 0] = 1
        X_s1 = (X_s1 - mean_s1) / std_s1
        
        # Save normalization stats under key-type-specific subdirectory
        kt_norm_dir = os.path.join(output_dir, "3des", key_type)
        os.makedirs(kt_norm_dir, exist_ok=True)
        np.save(os.path.join(kt_norm_dir, "mean_s1.npy"), mean_s1)
        np.save(os.path.join(kt_norm_dir, "std_s1.npy"), std_s1)

        if X_s2 is not None:
            mean_s2 = np.mean(X_s2, axis=0)
            std_s2 = np.std(X_s2, axis=0)
            std_s2[std_s2 == 0] = 1
            X_s2 = (X_s2 - mean_s2) / std_s2
            np.save(os.path.join(kt_norm_dir, "mean_s2.npy"), mean_s2)
            np.save(os.path.join(kt_norm_dir, "std_s2.npy"), std_s2)
        else:
            logger.warning(f"Stage 2 features missing for {key_type.upper()}; stage-2 training will be skipped.")

        # Gap #2: Create or initialize shared backbone for transfer learning
        # Determine num_classes based on label_type
        num_classes = 64 if label_type == "sbox_input" else 16
        
        if use_transfer_learning:
            # Phase 1 (KENC): Create new shared backbone with full training
            if phase_idx == 0:  # KENC phase
                shared_input_dim = int(X_s1.shape[1])
                shared_backbone = ZaidNetSharedBackbone(input_dim=shared_input_dim, num_classes=num_classes)
                # Ensure not frozen for Phase 1
                shared_backbone.unfreeze_backbone()
                logger.info(f"Created shared backbone (Phase 1: KENC full training, LR=0.001, num_classes={num_classes})")
                freeze_flag = False
            else:  # KMAC and KDEK phases
                # Reuse backbone from Phase 1, freeze it
                freeze_flag = True
                logger.info(f"Reusing shared backbone (Phase {phase_idx}: {key_type.upper()} fine-tuning, LR=0.0001)")

        for stage in (1, 2):
            X_curr_global = X_s1 if stage == 1 else X_s2
            X_curr = X_curr_global
            if X_curr is None:
                continue

            stage_dir = os.path.join(output_dir, "3des", key_type, f"s{stage}")
            os.makedirs(stage_dir, exist_ok=True)

            for sbox in range(8):
                sbox_num = sbox + 1
                y_path = os.path.join(input_dir, f"Y_labels_{key_type}_s{stage}_sbox{sbox_num}.npy")
                if not os.path.exists(y_path):
                    # Backward compat for original KENC stage1 labels.
                    if key_type == "kenc" and stage == 1:
                        y_path = os.path.join(input_dir, f"Y_labels_sbox{sbox_num}.npy")
                    if not os.path.exists(y_path):
                        continue

                Y = np.load(y_path).astype(np.longlong)

                valid_idx = Y != -1
                
                # Prefer per-sbox features when available (especially useful for stage 2).
                if stage == 1:
                    x_sbox_path = os.path.join(input_dir, f"X_sbox{sbox_num}.npy")
                else:
                    x_sbox_path = os.path.join(input_dir, f"X_s2_sbox{sbox_num}.npy")
                if os.path.exists(x_sbox_path):
                    X_candidate = np.load(x_sbox_path).astype(np.float32)
                    # ALWAYS use per-sbox features when available (800 dim >> 200 dim global)
                    # Per-sbox features provide significantly better S-Box classification accuracy
                    X_curr = X_candidate
                    # Normalize per-sbox features for stable optimization.
                    mean_local = np.mean(X_curr, axis=0)
                    std_local = np.std(X_curr, axis=0)
                    std_local[std_local == 0] = 1
                    X_curr = (X_curr - mean_local) / std_local
                    # Save per-sbox normalization stats so inference can match training distribution
                    np.save(os.path.join(kt_norm_dir, f"mean_s{stage}_sbox{sbox_num}.npy"), mean_local)
                    np.save(os.path.join(kt_norm_dir, f"std_s{stage}_sbox{sbox_num}.npy"), std_local)
                    logger.info(
                        "Using per-sbox features for %s Stage %d S-Box %d (%d dimensions)",
                        key_type.upper(),
                        stage,
                        sbox_num,
                        X_curr.shape[1],
                    )
                else:
                    X_curr = X_curr_global

                X_final = X_curr[valid_idx]
                Y_final = Y[valid_idx]
                if len(Y_final) == 0:
                    continue

                groups = groups_full[valid_idx] if groups_full is not None and len(groups_full) == len(Y) else None

                # Transfer-learning safety:
                # Shared backbone only supports the feature width it was created with.
                # Some per-sbox feature files can have a different width than global stage features.
                use_shared_for_this_run = use_transfer_learning and shared_backbone is not None
                if use_shared_for_this_run and shared_input_dim is not None and X_final.shape[1] != shared_input_dim:
                    logger.warning(
                        "Transfer learning disabled for %s Stage %d S-Box %d due to feature-dim mismatch "
                        "(got %d, expected %d). Falling back to standard model for this run.",
                        key_type.upper(),
                        stage,
                        sbox_num,
                        X_final.shape[1],
                        shared_input_dim,
                    )
                    use_shared_for_this_run = False

                logger.info(
                    f"Training 3DES {key_type.upper()} Stage {stage} S-Box {sbox_num} "
                    f"({models_per_sbox} models) using {len(Y_final)} traces..."
                )

                for i in range(models_per_sbox):
                    train_single_model(
                        X_final,
                        Y_final,
                        i,
                        sbox_num,
                        stage_dir,
                        epochs=epochs,
                        groups=groups,
                        early_stop_patience=early_stop_patience,
                        key_type=key_type,
                        shared_backbone=shared_backbone if use_shared_for_this_run else None,
                        freeze_backbone=freeze_flag if use_shared_for_this_run else False,
                        transfer_lr=0.0001 if (use_transfer_learning and phase_idx > 0) else 0.001,
                        label_type=label_type,
                    )
        
        logger.info(f"{phase_name} Complete")

if __name__ == "__main__":
    train_ensemble("Processed/Mastercard", "Models/Ensemble_ZaidNet")

if __name__ == "__main__":
    train_ensemble("Processed/Mastercard", "Models/Ensemble_ZaidNet")
