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
        preload_to_gpu: bool = False,      # NEW
        device: torch.device = None,       # NEW
        half_precision: bool = True,       # NEW
    ):
        full_df = pd.read_csv(csv_path)
        n = len(full_df)
        split_idx = int(n * (1 - val_ratio))

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
        self.device = device
        self.preload_to_gpu = preload_to_gpu and device is not None and device.type == "cuda"
        self.half_precision = half_precision

        self.feature_cols = feature_cols or ["ghi", "dni", "dhi"]
        self.target_cols = target_cols or ["ghi"]

        self.sky_col = "image_path_sky"
        self.flow_col = "image_path_optical_flow"
        if self.sky_col not in self.df.columns or self.flow_col not in self.df.columns:
            raise KeyError(f"CSV missing required columns: {self.sky_col}, {self.flow_col}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.max_lookback = max(img_seq_len, ts_seq_len)

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

        # Preload images to GPU if requested
        if self.preload_to_gpu:
            print(f"Preloading {split} images to {device}...")
            dtype = torch.float16 if self.half_precision else torch.float32
            self.sky_images = [self.transform(Image.open(p).convert("RGB")).to(device=device, dtype=dtype)
                               for p in self.df[self.sky_col].values]
            self.flow_images = [self.transform(Image.open(p).convert("RGB")).to(device=device, dtype=dtype)
                                for p in self.df[self.flow_col].values]
            print(f"Preloading complete: {len(self.sky_images)} samples.")
        else:
            self.sky_images = None
            self.flow_images = None

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
        if self.preload_to_gpu and self.sky_images is not None:
            sky_seq = torch.stack(self.sky_images[idx + self.ts_seq_len - self.img_seq_len : idx + self.ts_seq_len])
        else:
            sky_seq = torch.stack([self.transform(Image.open(p).convert("RGB")) for p in img_window[self.sky_col].values])

        # ===============================
        # OPTICAL FLOW IMAGES
        # ===============================
        if self.preload_to_gpu and self.flow_images is not None:
            flow_seq = torch.stack(self.flow_images[idx + self.ts_seq_len - self.img_seq_len : idx + self.ts_seq_len])
        else:
            flow_seq = torch.stack([self.transform(Image.open(p).convert("RGB")) for p in img_window[self.flow_col].values])

        # ===============================
        # TIME SERIES
        # ===============================
        ts_seq = torch.tensor(ts_window[self.feature_cols].values, dtype=torch.float32)

        # ===============================
        # TARGETS
        # ===============================
        target_seq = torch.tensor(target_window[self.target_cols].values, dtype=torch.float32)

        # ===============================
        # TIMESTAMPS & IMAGE NAMES
        # ===============================
        ts_times = ts_window[self.time_col].astype(str).tolist() if self.time_col in ts_window.columns else list(range(len(ts_window)))
        target_times = target_window[self.time_col].astype(str).tolist() if self.time_col in target_window.columns else list(range(len(ts_window), len(ts_window)+len(target_window)))

        sky_img_names = [os.path.basename(p) for p in img_window[self.sky_col].values]
        flow_img_names = [os.path.basename(p) for p in img_window[self.flow_col].values]

        return sky_seq, flow_seq, ts_seq, target_seq, ts_times, target_times, sky_img_names, flow_img_names

    # ---------------------------
    # DENORMALIZE & SHOW SAMPLE
    # ---------------------------
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

        fig, axes = plt.subplots(2, num_images, figsize=(3*num_images, 6))
        for i in range(num_images):
            img = self._denormalize(sky_seq[i]).permute(1, 2, 0).cpu().numpy().clip(0, 1)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Sky: {sky_names[i]}", fontsize=8)
            axes[0, i].axis("off")

        for i in range(num_images):
            img = self._denormalize(flow_seq[i]).permute(1, 2, 0).cpu().numpy().clip(0, 1)
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"Flow: {flow_names[i]}", fontsize=8)
            axes[1, i].axis("off")

        plt.suptitle("Sky + Optical Flow Sequences", fontsize=12)
        plt.tight_layout()
        plt.show()

        # TS plot
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

        # Print numeric values
        print("\nFirst 5 time-series samples:")
        print(pd.DataFrame(ts_seq[:5].numpy(), columns=self.feature_cols))
        print("\nFirst 5 target values:")
        print(pd.DataFrame(target_seq[:5].numpy(), columns=self.target_cols))

if __name__ == "__main__":
    import torch

    # ----------------------
    # Hardcoded configuration
    # ----------------------
    CSV_PATH = "processed_dataset_cropped_full.csv"  # Replace with your actual CSV path
    SPLIT = "train"
    IMG_SIZE = 224
    PRELOAD = True         # Set to True to preload images to GPU
    HALF = True            # Set to True for float16 GPU tensors
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")

    # ----------------------
    # Initialize dataset
    # ----------------------
    ds = IrradianceForecastDataset(
        csv_path=CSV_PATH,
        split=SPLIT,
        img_size=IMG_SIZE,
        preload_to_gpu=PRELOAD,
        device=DEVICE,
        half_precision=HALF
    )

    # ----------------------
    # Show a random sample
    # ----------------------
    ds.show_sample()
