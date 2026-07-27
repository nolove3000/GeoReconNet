import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data_utils import FORMAL_SCALER_PATH, FORMAL_SPLIT_PATH, PROJECT_ROOT, build_main_training_data, fit_or_load_formal_scaler, load_formal_split
from high_resolution_clean.pytorch.model import HybridLossTorch
from high_resolution_clean.pytorch.train import CleanHighResolutionDataset, build_lr_scheduler, run_epoch
from .model import ABLATION_VERSION, VARIANTS, ArchitectureAblationGenerator


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a controlled formal architecture ablation.")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main():
    args = parse_arguments()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, masks, geometry_types = build_main_training_data(out_size=1024, return_damage_types=True)
    split = load_formal_split(FORMAL_SPLIT_PATH, verify_dataset=True)
    if not np.array_equal(geometry_types, split["geometry_types"]):
        raise ValueError("Ablation data order does not match the frozen split")
    scaler = fit_or_load_formal_scaler(features, split["train_indices"])
    features = scaler.transform(features).astype(np.float32)
    options = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(CleanHighResolutionDataset(features[split["train_indices"]], masks[split["train_indices"]]), shuffle=True, generator=torch.Generator().manual_seed(args.seed), **options)
    validation_loader = DataLoader(CleanHighResolutionDataset(features[split["validation_indices"]], masks[split["validation_indices"]]), shuffle=False, **options)
    model = ArchitectureAblationGenerator(args.variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5, eps=1e-7)
    scheduler = build_lr_scheduler(optimizer)
    criterion = HybridLossTorch()
    initial_epoch, best_iou, history = 0, -np.inf, []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("architecture_version") != ABLATION_VERSION or checkpoint.get("variant") != args.variant:
            raise ValueError("Resume checkpoint does not match this ablation")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        initial_epoch, best_iou = int(checkpoint["epoch"]), float(checkpoint["best_validation_iou"])
    model_dir = PROJECT_ROOT / "models/ablations/architecture"
    output_dir = PROJECT_ROOT / "outputs/ablations/architecture" / args.variant / "training"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / f"best_{args.variant}_v1.pt"
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Ablation={args.variant}; device={device}; parameters={parameter_count:,}; output=1024x1024")
    for epoch in range(initial_epoch + 1, args.epochs + 1):
        start = time.perf_counter()
        lr = float(optimizer.param_groups[0]["lr"])
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_metrics = run_epoch(model, validation_loader, criterion, device)
        scheduler.step(val_loss)
        record = {"epoch": epoch, "learning_rate": lr, "next_learning_rate": float(optimizer.param_groups[0]["lr"]), "training_loss": train_loss, "validation_loss": val_loss, **{f"training_{k}": v for k, v in train_metrics.items()}, **{f"validation_{k}": v for k, v in val_metrics.items()}, "epoch_seconds": time.perf_counter() - start}
        history.append(record)
        if val_metrics["damage_iou"] > best_iou:
            best_iou = val_metrics["damage_iou"]
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch, "best_validation_iou": best_iou, "architecture_version": ABLATION_VERSION, "variant": args.variant, "input_dimension": 132, "out_size": 1024, "base_channels": model.base_channels, "parameter_count": parameter_count, "seed": args.seed, "split_path": str(FORMAL_SPLIT_PATH.relative_to(PROJECT_ROOT)), "scaler_path": str(FORMAL_SCALER_PATH.relative_to(PROJECT_ROOT)), "loss": "damage_weighted_bce_plus_1.5_tversky", "optimizer": "AdamW(lr=2e-4, weight_decay=1e-5, eps=1e-7)"}, checkpoint_path)
        (output_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Epoch {epoch:03d}: loss={train_loss:.6f}; val_loss={val_loss:.6f}; IoU={train_metrics['damage_iou']:.6f}; val_IoU={val_metrics['damage_iou']:.6f}; best={best_iou:.6f}; s={record['epoch_seconds']:.2f}", flush=True)
    print(f"Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
