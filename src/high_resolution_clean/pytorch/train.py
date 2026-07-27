import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common.data_utils import (
    FORMAL_SCALER_PATH,
    FORMAL_SPLIT_PATH,
    PROJECT_ROOT,
    build_main_training_data,
    fit_or_load_formal_scaler,
    load_formal_split,
)
from .model import (
    TORCH_ARCHITECTURE_VERSION,
    TORCH_INITIALIZATION_VERSION,
    DamageMetricAccumulator,
    HighResGeneratorTorch,
    HybridLossTorch,
)


BEST_HISTORY_LIMIT = 3
DEFAULT_RUN_NAME = "no_norm_dropout_000_v5"
BEST_HISTORY_PATTERN = re.compile(
    r"^best_epoch_(?P<epoch>\d+)_val_iou_(?P<iou>\d+(?:\.\d+)?)\.pt$"
)


class CleanHighResolutionDataset(Dataset):
    def __init__(self, features, masks):
        self.features = np.asarray(features, dtype=np.float32)
        self.masks = np.asarray(masks, dtype=np.uint8)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return torch.from_numpy(self.features[index]), torch.from_numpy(self.masks[index][None])


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train the clean PyTorch counterpart of the TensorFlow high-resolution model."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-size", type=int, default=1024, choices=[128, 256, 512, 1024])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--fresh-scaler", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Measure training speed without writing checkpoints or history artifacts.",
    )
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_lr_scheduler(optimizer):
    # Keep the scheduler hyperparameters aligned with the TensorFlow route.
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        threshold=1e-4,
        threshold_mode="abs",
        cooldown=0,
        min_lr=0.0,
    )


def run_epoch(model, data_loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = torch.zeros((), device=device)
    sample_count = 0
    metrics = DamageMetricAccumulator()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for features, masks in data_loader:
            features = features.to(device, non_blocking=True)
            masks = masks.to(device, dtype=torch.float32, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            solid_probability = model(features)
            loss = criterion(solid_probability, masks)
            if training:
                loss.backward()
                optimizer.step()
            batch_sample_count = len(features)
            loss_sum += loss.detach() * batch_sample_count
            sample_count += batch_sample_count
            metrics.update(solid_probability, masks)
    return float((loss_sum / max(sample_count, 1)).cpu()), metrics.compute()


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Requested resume checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_initialization = checkpoint.get("initialization_version")
    if checkpoint_initialization != TORCH_INITIALIZATION_VERSION:
        raise ValueError(
            "Checkpoint uses an incompatible or unrecorded initialization "
            f"({checkpoint_initialization!r}); start this Glorot-initialized route from scratch."
        )
    checkpoint_architecture = checkpoint.get("architecture_version")
    if checkpoint_architecture != TORCH_ARCHITECTURE_VERSION:
        raise ValueError(
            "Checkpoint uses an incompatible or unrecorded latent layout "
            f"({checkpoint_architecture!r}); start the NHWC-aligned route from scratch."
        )
    checkpoint_dropout = float(checkpoint.get("dropout", -1.0))
    if checkpoint_dropout != model.dropout:
        raise ValueError(
            f"Checkpoint dropout ({checkpoint_dropout}) does not match the requested model ({model.dropout})."
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is None:
            print("Checkpoint has no scheduler state; continuing with a fresh LR scheduler.")
        else:
            scheduler.load_state_dict(scheduler_state)
    return int(checkpoint["epoch"]), float(checkpoint["best_validation_iou"])


def build_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    arguments,
    validation_indices,
    test_indices,
    best_validation_iou,
):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "out_size": arguments.out_size,
        "input_dimension": 132,
        "scaler_path": str(FORMAL_SCALER_PATH.relative_to(PROJECT_ROOT)),
        "split_path": str(FORMAL_SPLIT_PATH.relative_to(PROJECT_ROOT)),
        "validation_indices": validation_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "seed": arguments.seed,
        "run_name": arguments.run_name,
        "initialization_version": TORCH_INITIALIZATION_VERSION,
        "architecture_version": TORCH_ARCHITECTURE_VERSION,
        "dropout": arguments.dropout,
        "best_validation_iou": best_validation_iou,
        "lr_scheduler": {
            "monitor": "validation_loss",
            "factor": 0.5,
            "non_improving_epoch_patience": 5,
            "threshold": 1e-4,
            "min_lr": 0.0,
        },
    }


def prune_best_history(directory, keep=BEST_HISTORY_LIMIT):
    """Keep only the highest validation-IoU historical best checkpoints."""
    directory = Path(directory)
    ranked = []
    for path in directory.glob("best_epoch_*_val_iou_*.pt"):
        match = BEST_HISTORY_PATTERN.match(path.name)
        if match is not None:
            ranked.append((float(match.group("iou")), int(match.group("epoch")), path))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    removed = []
    for _iou, _epoch, path in ranked[keep:]:
        path.unlink()
        removed.append(path)
    return removed


def main():
    arguments = parse_arguments()
    if arguments.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be a positive integer")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", arguments.run_name) is None:
        raise ValueError("--run-name may contain only letters, numbers, dots, underscores, and hyphens")
    if arguments.dropout != 0.0:
        raise ValueError("this architecture requires --dropout 0")
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)
        torch.backends.cudnn.benchmark = arguments.benchmark
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, masks, geometry_types = build_main_training_data(
        out_size=arguments.out_size, return_damage_types=True
    )
    stored_split = load_formal_split(FORMAL_SPLIT_PATH, verify_dataset=True)
    train_indices = stored_split["train_indices"]
    validation_indices = stored_split["validation_indices"]
    test_indices = stored_split["test_indices"]
    if not np.array_equal(geometry_types, stored_split["geometry_types"]):
        raise ValueError("Training data order does not match the frozen formal split")
    scaler = fit_or_load_formal_scaler(
        features, train_indices, force_refit=arguments.fresh_scaler
    )
    features = scaler.transform(features).astype(np.float32)

    loader_options = {
        "batch_size": arguments.batch_size,
        "num_workers": arguments.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        CleanHighResolutionDataset(features[train_indices], masks[train_indices]),
        shuffle=True,
        generator=torch.Generator().manual_seed(arguments.seed),
        **loader_options,
    )
    validation_loader = DataLoader(
        CleanHighResolutionDataset(features[validation_indices], masks[validation_indices]),
        shuffle=False,
        **loader_options,
    )

    model = HighResGeneratorTorch(out_size=arguments.out_size, dropout=arguments.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5, eps=1e-7)
    scheduler = None if arguments.benchmark else build_lr_scheduler(optimizer)
    criterion = HybridLossTorch()
    initial_epoch = 0
    best_validation_iou = -np.inf
    if arguments.resume is not None:
        initial_epoch, best_validation_iou = load_checkpoint(
            model, optimizer, scheduler, arguments.resume, device
        )

    model_directory = PROJECT_ROOT / "models/pytorch"
    training_directory = (
        PROJECT_ROOT / "outputs/high_resolution_clean/pytorch"
        / arguments.run_name / "training"
    )
    checkpoint_path = model_directory / (
        f"best_main_pytorch_{arguments.out_size}_{arguments.run_name}.pt"
    )
    run_history_directory = (
        model_directory / "history" / f"main_{arguments.out_size}" / arguments.run_name
    )
    periodic_directory = run_history_directory / "periodic"
    best_history_directory = run_history_directory / "best"
    history_path = training_directory / "metrics.json"
    if not arguments.benchmark:
        model_directory.mkdir(parents=True, exist_ok=True)
        periodic_directory.mkdir(parents=True, exist_ok=True)
        best_history_directory.mkdir(parents=True, exist_ok=True)
        training_directory.mkdir(parents=True, exist_ok=True)
        removed_best_paths = prune_best_history(best_history_directory)
        if removed_best_paths:
            print(f"Pruned {len(removed_best_paths)} historical best checkpoints; kept top {BEST_HISTORY_LIMIT}.")

    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        f"Framework=PyTorch; device={device}; resolution={arguments.out_size}x{arguments.out_size}; "
        f"trainable_parameters={parameter_count:,}; run={arguments.run_name}; "
        f"initialization={TORCH_INITIALIZATION_VERSION}; "
        f"architecture={TORCH_ARCHITECTURE_VERSION}; dropout={arguments.dropout}"
    )
    print(
        f"Stratified split: train={len(train_indices)}, validation={len(validation_indices)}, "
        f"test={len(test_indices)}; frozen split: {FORMAL_SPLIT_PATH}"
    )
    history = []
    epoch_seconds = []
    synchronize(device)
    total_start = time.perf_counter()
    for epoch in range(initial_epoch + 1, arguments.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        synchronize(device)
        training_start = time.perf_counter()
        training_loss, training_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        synchronize(device)
        training_seconds = time.perf_counter() - training_start

        validation_start = time.perf_counter()
        validation_loss, validation_metrics = run_epoch(model, validation_loader, criterion, device)
        synchronize(device)
        validation_seconds = time.perf_counter() - validation_start
        if scheduler is not None:
            scheduler.step(validation_loss)
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "next_learning_rate": next_learning_rate,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            **{f"training_{key}": value for key, value in training_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            "training_seconds": training_seconds,
            "validation_seconds": validation_seconds,
            "epoch_seconds": training_seconds + validation_seconds,
        }
        history.append(record)
        epoch_seconds.append(record["epoch_seconds"])
        print(
            f"Epoch {epoch:03d}: loss={training_loss:.6f}, val_loss={validation_loss:.6f}, "
            f"damage_iou={training_metrics['damage_iou']:.6f}, "
            f"val_damage_iou={validation_metrics['damage_iou']:.6f}, "
            f"lr={learning_rate:.8g}, next_lr={next_learning_rate:.8g}, "
            f"train_s={training_seconds:.3f}, val_s={validation_seconds:.3f}, "
            f"epoch_s={training_seconds + validation_seconds:.3f}"
        )

        is_new_best = validation_metrics["damage_iou"] > best_validation_iou
        if not arguments.benchmark and is_new_best:
            best_validation_iou = validation_metrics["damage_iou"]
            checkpoint = build_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                arguments,
                validation_indices,
                test_indices,
                best_validation_iou,
            )
            historical_best_path = best_history_directory / (
                f"best_epoch_{epoch:03d}_val_iou_{best_validation_iou:.9f}.pt"
            )
            torch.save(checkpoint, checkpoint_path)
            torch.save(checkpoint, historical_best_path)
            removed_best_paths = prune_best_history(best_history_directory)
            print(f"Saved new best: {historical_best_path}")
            if removed_best_paths:
                print(f"Removed from best top {BEST_HISTORY_LIMIT}: {removed_best_paths[0]}")
        if not arguments.benchmark and epoch % arguments.checkpoint_interval == 0:
            periodic_checkpoint = build_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                arguments,
                validation_indices,
                test_indices,
                best_validation_iou,
            )
            periodic_path = periodic_directory / f"epoch_{epoch:03d}.pt"
            torch.save(periodic_checkpoint, periodic_path)
            print(f"Saved periodic checkpoint: {periodic_path}")
        if not arguments.benchmark:
            history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    synchronize(device)
    total_seconds = time.perf_counter() - total_start
    print(
        f"Timing summary: measured_s={sum(epoch_seconds):.3f}, "
        f"mean_epoch_s={np.mean(epoch_seconds):.3f}, wall_s={total_seconds:.3f}, "
        f"benchmark={arguments.benchmark}"
    )
    if len(epoch_seconds) > 1:
        print(f"Warm mean (epochs 2+): mean_epoch_s={np.mean(epoch_seconds[1:]):.3f}")
    if not arguments.benchmark:
        print(f"Best checkpoint: {checkpoint_path}")
        print(f"Periodic checkpoints: {periodic_directory}")
        print(f"Historical best checkpoints (top {BEST_HISTORY_LIMIT}): {best_history_directory}")
        print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
