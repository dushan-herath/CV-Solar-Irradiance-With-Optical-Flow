import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
import os

class IrradianceForecastDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        val_ratio: float = 0.25,
        img_seq_len: int = 5,
        ts_seq_len: int = 30,
        horizon: int = 25,
        feature_cols=None,
        target_cols=None,
        transform=None,
        img_size: int = 224,
        time_col: str = "timestamp",
        normalization_stats: dict = None,
    ):
        full_df = pd.read_csv(csv_path)
        n = len(full_df)
        split_idx = int(n * (1 - val_ratio))

        # Split
        if split == "train":
            self.df = full_df.iloc[:split_idx].reset_index(drop=True)
        elif split == "val":
            self.df = full_df.iloc[split_idx:].reset_index(drop=True)
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.split = split
        self.img_seq_len = img_seq_len
        self.ts_seq_len = ts_seq_len
        self.horizon = horizon
        self.img_size = img_size
        self.time_col = time_col

        self.feature_cols = feature_cols or ["ghi", "dni", "dhi"]
        self.target_cols = target_cols or ["ghi"]

        # NEW: expected image columns
        self.sky_col = "image_path_sky"
        self.flow_col = "image_path_optical_flow"

        if self.sky_col not in self.df.columns:
            raise KeyError(f"CSV missing required column: {self.sky_col}")

        if self.flow_col not in self.df.columns:
            raise KeyError(f"CSV missing required column: {self.flow_col}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.max_lookback = max(img_seq_len, ts_seq_len)

        # Timestamps
        if self.time_col in self.df.columns:
            self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])

        # Normalization
        if split == "train":
            mean = self.df[self.feature_cols].mean()
            std = self.df[self.feature_cols].std()
            self.normalization_stats = {"mean": mean, "std": std}
        else:
            if normalization_stats is None:
                raise ValueError("Validation split requires normalization_stats")
            self.normalization_stats = normalization_stats
            mean = normalization_stats["mean"]
            std = normalization_stats["std"]

        self.df[self.feature_cols] = (self.df[self.feature_cols] - mean) / std

        # Summary
        print(f"\nDataset initialized ({split.upper()}):")
        print(f"Total samples available: {len(self)}")
        print(f"Image seq length: {self.img_seq_len}")
        print(f"Time-series seq length: {self.ts_seq_len}")
        print(f"Forecast horizon: {self.horizon}")
        print(f"Feature columns: {self.feature_cols}")
        print(f"Target columns: {self.target_cols}")
        print(f"Sky image column: {self.sky_col}")
        print(f"Flow image column: {self.flow_col}\n")

    def __len__(self):
        return len(self.df) - self.max_lookback - self.horizon

    def __getitem__(self, idx):
        img_window = self.df.iloc[idx + self.ts_seq_len - self.img_seq_len : idx + self.ts_seq_len]
        ts_window = self.df.iloc[idx : idx + self.ts_seq_len]
        target_window = self.df.iloc[idx + self.ts_seq_len : idx + self.ts_seq_len + self.horizon]

        # ===============================
        # SKY IMAGES
        # ===============================
        sky_seq = []
        for path in img_window[self.sky_col].values:
            img = Image.open(path).convert("RGB")
            sky_seq.append(self.transform(img))
        sky_seq = torch.stack(sky_seq)

        # ===============================
        # OPTICAL FLOW IMAGES
        # ===============================
        flow_seq = []
        for path in img_window[self.flow_col].values:
            img = Image.open(path).convert("RGB")
            flow_seq.append(self.transform(img))
        flow_seq = torch.stack(flow_seq)

        # ===============================
        # TIME SERIES
        # ===============================
        ts_seq = torch.tensor(ts_window[self.feature_cols].values, dtype=torch.float32)

        # ===============================
        # TARGETS
        # ===============================
        target_seq = torch.tensor(target_window[self.target_cols].values, dtype=torch.float32)

        # Timestamps
        if self.time_col in ts_window.columns:
            ts_times = [str(t) for t in ts_window[self.time_col].tolist()]
        else:
            ts_times = list(range(len(ts_window)))

        if self.time_col in target_window.columns:
            target_times = [str(t) for t in target_window[self.time_col].tolist()]
        else:
            target_times = list(range(len(ts_window), len(ts_window) + len(target_window)))

        sky_img_names = [os.path.basename(p) for p in img_window[self.sky_col].values]
        flow_img_names = [os.path.basename(p) for p in img_window[self.flow_col].values]

        return sky_seq, flow_seq, ts_seq, target_seq, ts_times, target_times, sky_img_names, flow_img_names

    # ================================================================
    # Visualization + DENORMALIZATION FIX
    # ================================================================
    @staticmethod
    def _denormalize(img_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return img_tensor * std + mean

    def show_sample(self, idx=None):
        if idx is None:
            idx = random.randint(0, len(self) - 1)

        (sky_seq, flow_seq, ts_seq, target_seq,
         ts_times, target_times, sky_names, flow_names) = self[idx]

        print(f"\nSample index: {idx}")
        print(f"Sky image seq: {sky_seq.shape}")
        print(f"Flow image seq: {flow_seq.shape}")
        print(f"TS seq: {ts_seq.shape}")
        print(f"Target seq: {target_seq.shape}")

        num_images = self.img_seq_len

        # ==================================================
        # IMAGE VISUALIZATION (2 rows) — FIXED
        # ==================================================
        fig, axes = plt.subplots(2, num_images, figsize=(3*num_images, 6))

        # Row 1: Sky images
        for i in range(num_images):
            img = self._denormalize(sky_seq[i]).permute(1, 2, 0).numpy().clip(0, 1)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Sky: {sky_names[i]}", fontsize=8)
            axes[0, i].axis("off")

        # Row 2: Optical-flow images
        for i in range(num_images):
            img = self._denormalize(flow_seq[i]).permute(1, 2, 0).numpy().clip(0, 1)
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"Flow: {flow_names[i]}", fontsize=8)
            axes[1, i].axis("off")

        plt.suptitle("Sky + Optical Flow Sequences", fontsize=12)
        plt.tight_layout()
        plt.show()

        # ==================================================
        # PRINT SAMPLE NUMERIC DATA
        # ==================================================
        print("\nFirst 5 time-series samples:")
        print(pd.DataFrame(ts_seq[:5].numpy(), columns=self.feature_cols))

        print("\nFirst 5 target values:")
        print(pd.DataFrame(target_seq[:5].numpy(), columns=self.target_cols))

        # ==================================================
        # TIME-SERIES VISUALIZATION
        # ==================================================
        plt.figure(figsize=(10, 4))

        past_vals = ts_seq[:, :len(self.target_cols)].numpy()
        future_vals = target_seq.numpy()

        for i, col in enumerate(self.target_cols):
            plt.plot(ts_times, past_vals[:, i], "--o", label=f"Past {col.upper()}")
            plt.plot(target_times, future_vals[:, i], "-x", label=f"Future {col.upper()}")

        plt.xlabel("Time")
        plt.ylabel("Irradiance (W/m²)")
        plt.title("Irradiance Forecast Visualization")
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
