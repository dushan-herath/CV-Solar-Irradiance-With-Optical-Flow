import os
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import IrradianceForecastDataset
from model import ImageEncoder, MultimodalForecaster


# ------------------ Training + Validation ------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, total=len(loader), desc="Training", leave=False)

    for sky_seq, flow_seq, ts_seq, targets, *_ in loop:
        sky_seq = sky_seq.to(device)
        flow_seq = flow_seq.to(device)
        ts_seq = ts_seq.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with torch.cuda.amp.autocast():
            preds = model(sky_seq, flow_seq, ts_seq)
            loss = criterion(preds, targets)

        # Backward with GradScaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)



def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        loop = tqdm(loader, total=len(loader), desc="Validation", leave=False)
        for sky_seq, flow_seq, ts_seq, targets, *_ in loop:
            sky_seq = sky_seq.to(device)
            flow_seq = flow_seq.to(device)
            ts_seq = ts_seq.to(device)
            targets = targets.to(device)

            preds = model(sky_seq, flow_seq, ts_seq)
            loss = criterion(preds, targets)
            total_loss += loss.item()

    return total_loss / len(loader)


# ------------------ Utilities ------------------

def plot_losses(train_losses, val_losses, save_path="training_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved loss plot to {save_path}")


def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)
    print(f"Checkpoint saved at epoch {state['epoch']+1} -> {filename}")


def load_checkpoint(filename, model, optimizer, device):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]
    train_losses = checkpoint["train_losses"]
    val_losses = checkpoint["val_losses"]
    print(f"Resumed from checkpoint: epoch {epoch+1} | best val loss = {best_val_loss:.5f}")
    return epoch, best_val_loss, train_losses, val_losses


# ------------------ Main ------------------

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()  

    # --- Config ---
    CSV_PATH = "processed_dataset_cropped_full.csv"
    BATCH_SIZE = 4
    NUM_EPOCHS = 25
    #LR = 1e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SEQ_LEN = 5
    TS_SEQ_LEN = 30
    HORIZON = 25
    TARGET_DIM = 1

    print(f"Training on {DEVICE}")

    # --- Dataset setup ---
    train_ds = IrradianceForecastDataset(
        csv_path=CSV_PATH,
        split="train",
        img_seq_len=IMG_SEQ_LEN,
        ts_seq_len=TS_SEQ_LEN,
        horizon=HORIZON,
    )
    val_ds = IrradianceForecastDataset(
        csv_path=CSV_PATH,
        split="val",
        img_seq_len=IMG_SEQ_LEN,
        ts_seq_len=TS_SEQ_LEN,
        horizon=HORIZON,
        normalization_stats=train_ds.normalization_stats,
    )

    # Save normalization stats
    json.dump(
        {
            "mean": train_ds.normalization_stats["mean"].to_dict(),
            "std": train_ds.normalization_stats["std"].to_dict(),
        },
        open("norm_stats.json", "w"),
        indent=4
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # --- Model setup ---
    sky_encoder = ImageEncoder(model_name="efficientnetv2_s", pretrained=False, freeze=False, unfreeze_last=2)
    flow_encoder = ImageEncoder(model_name="resnet18", pretrained=True, freeze=True, unfreeze_last=2)



    model = MultimodalForecaster(
        sky_encoder=sky_encoder,
        flow_encoder=flow_encoder,
        ts_feat_dim=len(train_ds.feature_cols),
        horizon=HORIZON,
        target_dim=TARGET_DIM,
        d_model=256,
        num_layers=2,
    ).to(DEVICE)


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"Model ready on {DEVICE} | "
        f"Total: {total_params/1e6:.2f}M | "
        f"Trainable: {trainable_params/1e6:.2f}M"
    )


    # --- Training setup ---
    criterion = nn.MSELoss()
    
    #optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # Layer-wise learning rates
    optimizer = torch.optim.AdamW([
        {"params": model.sky_encoder.parameters(), "lr": 1e-5},      # pretrained sky encoder
        {"params": model.flow_encoder.parameters(), "lr": 1e-5},     # pretrained flow encoder
        {"params": model.ts_encoder.parameters(), "lr": 1e-4},       # TS encoder
        {"params": model.fusion.parameters(), "lr": 1e-4},           # Fusion layers
        {"params": model.temporal.parameters(), "lr": 1e-4},         # Transformer
        {"params": model.head.parameters(), "lr": 1e-4},             # Output head
    ], weight_decay=1e-4)

    scaler = torch.amp.GradScaler(device=DEVICE.type if DEVICE.type == "cuda" else "cpu")


    # --- Resume from checkpoint if exists ---
    start_epoch = 0
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    if os.path.exists("checkpoint.pth"):
        start_epoch, best_val_loss, train_losses, val_losses = load_checkpoint("checkpoint.pth", model, optimizer, DEVICE)
    else:
        print("Starting new training session")

    # --- Training loop ---
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_loss = validate_one_epoch(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model updated!")

        # Save checkpoint each epoch
        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
        })

    plot_losses(train_losses, val_losses)
    print("\nTraining complete!")
    print(f"Best Validation Loss: {best_val_loss:.5f}")
