"""
Visualization tool for solar irradiance forecasting results.

Reads forecast_results.npz (exported from export_predictions_with_errors.py)
and provides:
  Error vs horizon plots (MSE, MAE, RMSE)
  Actual vs Predicted time series plots (per target & horizon)
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------
# Plot 1: Error vs Horizon
# -------------------------------
def plot_error_vs_horizon(errors, metric_name, target_names, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    horizons = np.arange(1, errors.shape[0] + 1)

    plt.figure(figsize=(8, 5))
    for i, name in enumerate(target_names):
        plt.plot(horizons, errors[:, i], marker='o', label=name.upper())
    plt.xlabel("Forecast Horizon")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} vs Forecast Horizon (Original Units)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{metric_name.lower()}_vs_horizon.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved -> {save_path}")


# -------------------------------
# Plot 2: Predicted vs Actual (per target, per horizon)
# -------------------------------
def plot_predictions_vs_time(preds, targets, target_names, horizon=1, num_samples=None, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    t_idx = horizon - 1
    num_samples = num_samples or preds.shape[0]
    num_samples = min(num_samples, preds.shape[0])

    for i, name in enumerate(target_names):
        plt.figure(figsize=(10, 5))
        plt.plot(targets[:num_samples, t_idx, i], label=f"Actual {name.upper()}", linewidth=2)
        plt.plot(preds[:num_samples, t_idx, i], linestyle="--", label=f"Predicted {name.upper()}", linewidth=2)
        plt.xlabel("Sample Index (Time)")
        plt.ylabel(f"{name.upper()} (W/m²)")
        plt.title(f"Actual vs Predicted {name.upper()} @ Horizon={horizon}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"pred_vs_actual_{name.lower()}_h{horizon}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved -> {save_path}")


# -------------------------------
# Main script
# -------------------------------
def main(file_path="forecast_results.npz"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Run export_predictions_with_errors.py first.")

    print(f"Loading results from {file_path}")
    data = np.load(file_path, allow_pickle=True)
    preds = data["preds"]
    targets = data["targets"]
    mse = data["mse"]
    mae = data["mae"]
    rmse = data["rmse"]
    target_names = data["target_names"]

    print("Data loaded successfully")
    print(f"preds shape: {preds.shape}")
    print(f"mse shape: {mse.shape}")

    # --- Error vs Horizon plots ---
    print("\nGenerating error vs horizon plots...")
    plot_error_vs_horizon(mse, "MSE", target_names)
    plot_error_vs_horizon(mae, "MAE", target_names)
    plot_error_vs_horizon(rmse, "RMSE", target_names)

    # --- Actual vs Predicted plots for selected horizons ---
    horizons_to_plot = [1, 5, 10, 25]  # customize as needed
    print("\nGenerating predicted vs actual plots...")
    for h in horizons_to_plot:
        plot_predictions_vs_time(preds, targets, target_names, horizon=h, num_samples=None)

    print("\nAll plots saved in the 'plots/' folder.")


if __name__ == "__main__":
    main()
