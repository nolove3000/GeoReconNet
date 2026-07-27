import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data_utils import (
    FORMAL_SCALER_PATH,
    FORMAL_SPLIT_PATH,
    PROJECT_ROOT,
    build_main_training_data,
    fit_or_load_formal_scaler,
    load_formal_split,
)
from common.plot_style import configure_plot_style
from high_resolution_clean.pytorch.model import HybridLossTorch
from high_resolution_clean.pytorch.train import CleanHighResolutionDataset, build_lr_scheduler, run_epoch
from .model import SIMPLE_DECONV_VERSION, SimpleDeconvBaseline


configure_plot_style()
DEFAULT_RUN_NAME = "simple_deconv_1024_v1"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the formal full-resolution simple-deconvolution baseline.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def save_history(history, output_directory, draw_plot):
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "metrics.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    if not draw_plot:
        return
    epochs = [record["epoch"] for record in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(epochs, [record["training_loss"] for record in history], label="Training")
    axes[0].plot(epochs, [record["validation_loss"] for record in history], label="Validation")
    axes[0].set(title="Loss history", xlabel="Epoch", ylabel="Hybrid loss")
    axes[1].plot(epochs, [record["training_damage_iou"] for record in history], label="Training")
    axes[1].plot(epochs, [record["validation_damage_iou"] for record in history], label="Validation")
    axes[1].set(title="Damage-class overlap", xlabel="Epoch", ylabel="Damage IoU")
    for axis in axes:
        axis.grid(False, which="both")
        axis.legend()
    figure.savefig(output_directory / "training_history.png", dpi=300)
    plt.close(figure)


def build_checkpoint(model, optimizer, scheduler, epoch, best_iou, arguments, split):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_validation_iou": best_iou,
        "architecture_version": SIMPLE_DECONV_VERSION,
        "input_dimension": 132,
        "latent_channels": arguments.latent_channels,
        "out_size": 1024,
        "seed": arguments.seed,
        "run_name": arguments.run_name,
        "split_path": str(FORMAL_SPLIT_PATH.relative_to(PROJECT_ROOT)),
        "scaler_path": str(FORMAL_SCALER_PATH.relative_to(PROJECT_ROOT)),
        "train_indices": split["train_indices"].tolist(),
        "validation_indices": split["validation_indices"].tolist(),
        "test_indices": split["test_indices"].tolist(),
        "loss": "damage_weighted_bce_plus_1.5_tversky",
        "optimizer": "AdamW(lr=2e-4, weight_decay=1e-5, eps=1e-7)",
    }


def main():
    arguments = parse_arguments()
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, masks, geometry_types = build_main_training_data(
        out_size=1024, return_damage_types=True
    )
    split = load_formal_split(FORMAL_SPLIT_PATH, verify_dataset=True)
    if not np.array_equal(geometry_types, split["geometry_types"]):
        raise ValueError("Simple-deconvolution data order does not match the frozen split")
    scaler = fit_or_load_formal_scaler(features, split["train_indices"])
    features = scaler.transform(features).astype(np.float32)
    loader_options = {
        "batch_size": arguments.batch_size,
        "num_workers": arguments.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        CleanHighResolutionDataset(features[split["train_indices"]], masks[split["train_indices"]]),
        shuffle=True,
        generator=torch.Generator().manual_seed(arguments.seed),
        **loader_options,
    )
    validation_loader = DataLoader(
        CleanHighResolutionDataset(features[split["validation_indices"]], masks[split["validation_indices"]]),
        shuffle=False,
        **loader_options,
    )
    model = SimpleDeconvBaseline(latent_channels=arguments.latent_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5, eps=1e-7)
    scheduler = build_lr_scheduler(optimizer)
    criterion = HybridLossTorch()
    initial_epoch = 0
    best_iou = -np.inf
    history = []
    if arguments.resume is not None:
        checkpoint = torch.load(arguments.resume, map_location=device, weights_only=False)
        if checkpoint.get("architecture_version") != SIMPLE_DECONV_VERSION:
            raise ValueError("Resume checkpoint does not match the formal simple-deconvolution baseline")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        initial_epoch = int(checkpoint["epoch"])
        best_iou = float(checkpoint["best_validation_iou"])

    model_directory = PROJECT_ROOT / "models/baselines/simple_deconv"
    training_directory = PROJECT_ROOT / "outputs/baselines/simple_deconv" / arguments.run_name / "training"
    model_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_directory / f"best_{arguments.run_name}.pt"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Simple deconv; device={device}; parameters={parameter_count:,}; output=1024x1024")
    for epoch in range(initial_epoch + 1, arguments.epochs + 1):
        start = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        training_loss, training_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss, validation_metrics = run_epoch(model, validation_loader, criterion, device)
        scheduler.step(validation_loss)
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            **{f"training_{key}": value for key, value in training_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "epoch_seconds": time.perf_counter() - start,
        }
        history.append(record)
        if validation_metrics["damage_iou"] > best_iou:
            best_iou = validation_metrics["damage_iou"]
            torch.save(
                build_checkpoint(model, optimizer, scheduler, epoch, best_iou, arguments, split),
                checkpoint_path,
            )
        save_history(history, training_directory, draw_plot=epoch % 10 == 0 or epoch == arguments.epochs)
        print(
            f"Epoch {epoch:03d}: loss={training_loss:.6f}; val_loss={validation_loss:.6f}; "
            f"IoU={training_metrics['damage_iou']:.6f}; val_IoU={validation_metrics['damage_iou']:.6f}; "
            f"best={best_iou:.6f}; s={record['epoch_seconds']:.2f}",
            flush=True,
        )
    print(f"Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
