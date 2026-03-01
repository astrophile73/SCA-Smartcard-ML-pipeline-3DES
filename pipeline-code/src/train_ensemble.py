
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from src.dataset import SCADataset
from src.model_zaid import get_model
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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

    model = get_model(input_dim=X.shape[1], num_classes=16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_acc = 0.0
    no_improve = 0
    save_path = os.path.join(save_dir, f"sbox{sbox_idx}_model{model_idx}.pth")

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
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

def train_ensemble(input_dir, output_dir, models_per_sbox=5, epochs=30, early_stop_patience=8):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load stage-specific features (stage 1 is required; stage 2 optional but needed for full 16-byte recovery).
    x_s1_path = os.path.join(input_dir, "X_features_s1.npy")
    if not os.path.exists(x_s1_path):
        # Backward compat
        x_s1_path = os.path.join(input_dir, "X_features.npy")
    x_s2_path = os.path.join(input_dir, "X_features_s2.npy")

    logger.info(f"Loading Stage 1 features from {x_s1_path}")
    X_s1 = np.load(x_s1_path).astype(np.float32)
    X_s2 = np.load(x_s2_path).astype(np.float32) if os.path.exists(x_s2_path) else None

    # Persist normalization stats so inference can match training.
    mean_s1 = np.mean(X_s1, axis=0)
    std_s1 = np.std(X_s1, axis=0)
    std_s1[std_s1 == 0] = 1
    X_s1 = (X_s1 - mean_s1) / std_s1
    np.save(os.path.join(output_dir, "mean_s1.npy"), mean_s1)
    np.save(os.path.join(output_dir, "std_s1.npy"), std_s1)

    if X_s2 is not None:
        mean_s2 = np.mean(X_s2, axis=0)
        std_s2 = np.std(X_s2, axis=0)
        std_s2[std_s2 == 0] = 1
        X_s2 = (X_s2 - mean_s2) / std_s2
        np.save(os.path.join(output_dir, "mean_s2.npy"), mean_s2)
        np.save(os.path.join(output_dir, "std_s2.npy"), std_s2)
    else:
        logger.warning("Stage 2 features missing; stage-2 training will be skipped.")

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

    for key_type in ["kenc", "kmac", "kdek"]:
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
                    X_curr = np.load(x_sbox_path).astype(np.float32)
                    # Normalize per-sbox features for stable optimization.
                    mean_local = np.mean(X_curr, axis=0)
                    std_local = np.std(X_curr, axis=0)
                    std_local[std_local == 0] = 1
                    X_curr = (X_curr - mean_local) / std_local
                else:
                    X_curr = X_curr_global

                X_final = X_curr[valid_idx]
                Y_final = Y[valid_idx]
                if len(Y_final) == 0:
                    continue

                groups = groups_full[valid_idx] if groups_full is not None and len(groups_full) == len(Y) else None

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
                    )

if __name__ == "__main__":
    train_ensemble("Processed/Mastercard", "Models/Ensemble_ZaidNet")
