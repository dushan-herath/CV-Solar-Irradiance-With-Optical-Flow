"""
Exports denormalized model predictions, ground truth targets,
and error metrics (MSE, MAE, RMSE) for all horizons and targets.

Output: forecast_results.npz
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import IrradianceForecastDataset
from model import ImageEncoder, MultimodalForecaster


@torch.no_grad()
def evaluate(model, loader, device, mean_targets, std_targets):
    """Runs model inference and returns predictions, targets, and metrics."""
    model.eval()
    all_preds, all_targets = [], []

    for sky_seq, flow_seq, ts_seq, targets, *_ in tqdm(loader, desc="Evaluating", leave=False):
        sky_seq, flow_seq, ts_seq = sky_seq.to(device), flow_seq.to(device), ts_seq.to(device)
        preds = model(sky_seq, flow_seq, ts_seq)

        if preds.shape[1] != targets.shape[1]:
            preds = preds[:, :targets.shape[1], :]

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # --- Denormalize ---
    preds_denorm = preds * std_targets + mean_targets
    targets_denorm = targets * std_targets + mean_targets

    # --- Compute errors ---
    errors = preds_denorm - targets_denorm
    mse_per_horizon = np.mean(errors**2, axis=0)   # (horizon, target_dim)
    mae_per_horizon = np.mean(np.abs(errors), axis=0)
    rmse_per_horizon = np.sqrt(mse_per_horizon)

    return preds_denorm, targets_denorm, mse_per_horizon, mae_per_horizon, rmse_per_horizon


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CSV_PATH = "processed_dataset_cropped_full.csv"
    IMG_SEQ_LEN = 5
    TS_SEQ_LEN = 30
    MAX_HORIZON = 25
    TARGET_DIM = 1
    BATCH_SIZE = 12

    print(f"Exporting predictions & metrics on {DEVICE} using best_model.pth")

    # --- Load normalization stats ---
    if not os.path.exists("norm_stats.json"):
        raise FileNotFoundError("norm_stats.json not found.")
    full_norm_stats = json.load(open("norm_stats.json"))
    full_mean = pd.Series(full_norm_stats["mean"])
    full_std = pd.Series(full_norm_stats["std"])
    normalization_stats = {"mean": full_mean, "std": full_std}

    TARGET_NAMES = ["ghi"]
    mean_targets = np.array([full_mean[n] for n in TARGET_NAMES]).reshape(1, 1, TARGET_DIM)
    std_targets = np.array([full_std[n] for n in TARGET_NAMES]).reshape(1, 1, TARGET_DIM)

    # --- Dataset setup ---
    val_ds = IrradianceForecastDataset(
        csv_path=CSV_PATH,
        split="val",
        img_seq_len=IMG_SEQ_LEN,
        ts_seq_len=TS_SEQ_LEN,
        horizon=MAX_HORIZON,
        normalization_stats=normalization_stats
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)
    print(f"Dataset initialized (VAL): {len(val_ds)} samples, horizon={MAX_HORIZON}")

    # --- Model setup ---
    sky_encoder = ImageEncoder(model_name="convnextv2_tiny", pretrained=False, freeze=True)
    flow_encoder = ImageEncoder(model_name="convnextv2_tiny", pretrained=False, freeze=True)

    model = MultimodalForecaster(
        sky_encoder=sky_encoder,
        flow_encoder=flow_encoder,
        ts_feat_dim=len(full_mean),
        horizon=MAX_HORIZON,
        target_dim=TARGET_DIM,
        d_model=256,
        num_layers=2
    ).to(DEVICE)

    if not os.path.exists("best_model.pth"):
        raise FileNotFoundError("best_model.pth not found.")
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    print("Loaded best_model.pth")

    # --- Evaluate and collect results ---
    preds_denorm, targets_denorm, mse, mae, rmse = evaluate(
        model, val_loader, DEVICE, mean_targets, std_targets
    )

    # --- Save results ---
    save_path = "forecast_results.npz"
    np.savez_compressed(
        save_path,
        preds=preds_denorm,
        targets=targets_denorm,
        mse=mse,
        mae=mae,
        rmse=rmse,
        target_names=np.array(TARGET_NAMES)
    )

    print(f"\nSaved full results -> {save_path}")
    print(f"Shapes:")
    print(f"preds={preds_denorm.shape}")
    print(f"targets={targets_denorm.shape}")
    print(f"mse/mae/rmse={mse.shape} (horizon × target_dim)")
    print("Export complete!")
