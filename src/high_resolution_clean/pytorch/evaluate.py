import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data_utils import (
    FORMAL_SPLIT_PATH,
    PROJECT_ROOT,
    build_main_training_data,
    load_formal_scaler,
    load_formal_split,
)
from common.plot_style import configure_plot_style
from .model import TORCH_ARCHITECTURE_VERSION, HighResGeneratorTorch, HybridLossTorch
from .train import CleanHighResolutionDataset


configure_plot_style()


METRIC_NAMES = ("damage_iou", "damage_dice", "damage_precision", "damage_recall")
GEOMETRY_LABELS = {
    "upward_crack_1mm": "1 mm upward crack",
    "upward_crack_5mm": "5 mm upward crack",
    "downward_crack_5mm": "5 mm downward crack",
    "circular_hole": "Circular hole/notch",
    "double_crack_1mm": "Double 1 mm cracks",
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate the formal clean PyTorch high-resolution model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            PROJECT_ROOT
            / "models/pytorch/best_main_pytorch_1024_no_norm_dropout_000_v5_continue150.pt"
        ),
    )
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--examples-per-class", type=int, default=3)
    parser.add_argument("--output-name")
    return parser.parse_args()


def safe_ratio(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(np.asarray(numerator, dtype=np.float64)),
        where=np.asarray(denominator) != 0,
    )


def metrics_from_counts(true_positive, false_positive, false_negative):
    return {
        "damage_iou": safe_ratio(
            true_positive, true_positive + false_positive + false_negative
        ),
        "damage_dice": safe_ratio(
            2 * true_positive, 2 * true_positive + false_positive + false_negative
        ),
        "damage_precision": safe_ratio(true_positive, true_positive + false_positive),
        "damage_recall": safe_ratio(true_positive, true_positive + false_negative),
    }


def bootstrap_mean_interval(values, resamples, seed):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1 or resamples == 0:
        return float(values.mean()), float(values.mean())
    generator = np.random.default_rng(seed)
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    chunk_size = max(1, min(resamples, 1000))
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        sampled = generator.integers(0, len(values), size=(stop - start, len(values)))
        bootstrap_means[start:stop] = values[sampled].mean(axis=1)
    return tuple(np.percentile(bootstrap_means, [2.5, 97.5]).tolist())


def summarize_records(records, bootstrap_resamples, bootstrap_seed):
    rows = []
    groups = [("overall", records)]
    geometry_order = list(GEOMETRY_LABELS)
    for geometry_type in geometry_order:
        groups.append(
            (geometry_type, [record for record in records if record["geometry_type"] == geometry_type])
        )
    for group_index, (group, group_records) in enumerate(groups):
        row = {"group": group, "label": GEOMETRY_LABELS.get(group, "Overall"), "n": len(group_records)}
        for metric_index, metric_name in enumerate(METRIC_NAMES):
            values = np.asarray([record[metric_name] for record in group_records])
            low, high = bootstrap_mean_interval(
                values,
                bootstrap_resamples,
                bootstrap_seed + group_index * 100 + metric_index,
            )
            row[f"{metric_name}_mean"] = float(values.mean())
            row[f"{metric_name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric_name}_ci95_low"] = low
            row[f"{metric_name}_ci95_high"] = high
        rows.append(row)

    class_rows = rows[1:]
    macro_row = {"group": "macro_average", "label": "Macro average", "n": len(records)}
    for metric_name in METRIC_NAMES:
        class_means = np.asarray([row[f"{metric_name}_mean"] for row in class_rows])
        macro_row[f"{metric_name}_mean"] = float(class_means.mean())
        macro_row[f"{metric_name}_std"] = float(class_means.std(ddof=1))
        low, high = bootstrap_mean_interval(
            class_means, bootstrap_resamples, bootstrap_seed + 10_000 + METRIC_NAMES.index(metric_name)
        )
        macro_row[f"{metric_name}_ci95_low"] = low
        macro_row[f"{metric_name}_ci95_high"] = high
    rows.append(macro_row)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_example(model, feature, truth, record, rank, device, output_directory):
    with torch.no_grad():
        probability = model(torch.from_numpy(feature[None]).to(device)).cpu().numpy()[0, 0]
    prediction = probability < 0.5
    damage_truth = truth == 0
    error = prediction != damage_truth
    figure, axes = plt.subplots(1, 4, figsize=(19, 4.5), constrained_layout=True)
    panels = [
        (damage_truth, "Ground-truth damage", "Blues", 0, 1),
        (1 - probability, "Predicted damage probability", "Blues", 0, 1),
        (prediction, "Thresholded prediction", "Blues", 0, 1),
        (error, "Pixelwise error", "Blues", 0, 1),
    ]
    for axis, (values, title, color_map, minimum, maximum) in zip(axes, panels):
        image = axis.imshow(
            values,
            extent=[0, 200, 0, 100],
            origin="lower",
            aspect="equal",
            cmap=color_map,
            vmin=minimum,
            vmax=maximum,
        )
        axis.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        f"{GEOMETRY_LABELS[record['geometry_type']]} | {rank} | "
        f"sample index={record['global_index']} | Damage IoU={record['damage_iou']:.4f}"
    )
    path = output_directory / record["geometry_type"] / (
        f"{rank}_sample_{record['global_index']}_iou_{record['damage_iou']:.4f}.png"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def save_ranked_examples(
    model, features, masks, records, examples_per_class, device, output_directory
):
    if examples_per_class <= 0:
        return []
    saved = []
    for geometry_type in GEOMETRY_LABELS:
        class_records = sorted(
            (record for record in records if record["geometry_type"] == geometry_type),
            key=lambda record: record["damage_iou"],
        )
        positions = np.linspace(
            0, len(class_records) - 1, min(examples_per_class, len(class_records))
        ).round().astype(int)
        labels = ["worst", "median", "best"] if len(positions) == 3 else [f"rank_{i + 1}" for i in range(len(positions))]
        for label, position in zip(labels, positions):
            record = class_records[int(position)]
            index = record["global_index"]
            saved.append(
                save_example(
                    model,
                    features[index],
                    masks[index],
                    record,
                    label,
                    device,
                    output_directory,
                )
            )
    return saved


def main():
    arguments = parse_arguments()
    if not arguments.model.is_file():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {arguments.model}")
    if not 0 < arguments.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1")
    if arguments.bootstrap_resamples < 0:
        raise ValueError("--bootstrap-resamples must be non-negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(arguments.model, map_location=device, weights_only=False)
    architecture = checkpoint.get("architecture_version")
    if architecture != TORCH_ARCHITECTURE_VERSION:
        raise ValueError(f"Checkpoint architecture is not supported: {architecture!r}")
    out_size = int(checkpoint.get("out_size", 1024))
    dropout = float(checkpoint.get("dropout", 0.0))
    model = HighResGeneratorTorch(out_size=out_size, dropout=dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    features, masks, geometry_types = build_main_training_data(
        out_size=out_size, return_damage_types=True
    )
    split = load_formal_split(FORMAL_SPLIT_PATH, verify_dataset=True)
    if not np.array_equal(geometry_types, split["geometry_types"]):
        raise ValueError("Evaluation data order does not match the frozen formal split")
    index_key = {
        "train": "train_indices",
        "validation": "validation_indices",
        "test": "test_indices",
    }[arguments.split]
    indices = np.asarray(split[index_key], dtype=np.int64)
    scaler = load_formal_scaler()
    features = scaler.transform(features).astype(np.float32)
    data_loader = DataLoader(
        CleanHighResolutionDataset(features[indices], masks[indices]),
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        pin_memory=device.type == "cuda",
    )

    criterion = HybridLossTorch()
    records = []
    loss_sum = 0.0
    offset = 0
    micro_counts = np.zeros(3, dtype=np.int64)
    with torch.no_grad():
        for batch_features, batch_masks in data_loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_masks = batch_masks.to(device, dtype=torch.float32, non_blocking=True)
            solid_probability = model(batch_features)
            loss_sum += float(criterion(solid_probability, batch_masks).cpu()) * len(batch_features)
            damage_prediction = solid_probability < arguments.threshold
            damage_truth = batch_masks < 0.5
            axes = tuple(range(1, damage_truth.ndim))
            true_positive = (damage_prediction & damage_truth).sum(dim=axes).cpu().numpy()
            false_positive = (damage_prediction & ~damage_truth).sum(dim=axes).cpu().numpy()
            false_negative = (~damage_prediction & damage_truth).sum(dim=axes).cpu().numpy()
            batch_metrics = metrics_from_counts(true_positive, false_positive, false_negative)
            micro_counts += np.asarray(
                [true_positive.sum(), false_positive.sum(), false_negative.sum()], dtype=np.int64
            )
            for local_index in range(len(batch_features)):
                global_index = int(indices[offset + local_index])
                record = {
                    "split_position": offset + local_index,
                    "global_index": global_index,
                    "sample_id": str(split["sample_ids"][global_index]),
                    "source_file": str(split["source_files"][global_index]),
                    "geometry_type": str(split["geometry_types"][global_index]),
                    "true_positive_pixels": int(true_positive[local_index]),
                    "false_positive_pixels": int(false_positive[local_index]),
                    "false_negative_pixels": int(false_negative[local_index]),
                }
                record.update(
                    {name: float(batch_metrics[name][local_index]) for name in METRIC_NAMES}
                )
                records.append(record)
            offset += len(batch_features)

    summary_rows = summarize_records(
        records, arguments.bootstrap_resamples, arguments.bootstrap_seed
    )
    micro_metrics = metrics_from_counts(*micro_counts)
    output_name = arguments.output_name or arguments.model.stem
    output_directory = (
        PROJECT_ROOT
        / "outputs/high_resolution_clean/pytorch"
        / output_name
        / "evaluation"
        / arguments.split
    )
    sample_path = output_directory / "per_sample_metrics.csv"
    summary_path = output_directory / "summary_metrics.csv"
    metadata_path = output_directory / "evaluation_metadata.json"
    write_csv(sample_path, records)
    write_csv(summary_path, summary_rows)
    output_directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": str(arguments.model.relative_to(PROJECT_ROOT)),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_iou": float(checkpoint["best_validation_iou"]),
        "architecture_version": architecture,
        "split": arguments.split,
        "split_path": str(FORMAL_SPLIT_PATH.relative_to(PROJECT_ROOT)),
        "sample_count": len(records),
        "threshold": arguments.threshold,
        "bootstrap_resamples": arguments.bootstrap_resamples,
        "bootstrap_seed": arguments.bootstrap_seed,
        "mean_loss": loss_sum / len(records),
        "micro_metrics": {name: float(value) for name, value in micro_metrics.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    example_paths = save_ranked_examples(
        model,
        features,
        masks,
        records,
        arguments.examples_per_class,
        device,
        output_directory / "ranked_examples",
    )

    print(
        f"model={arguments.model}; split={arguments.split}; samples={len(records)}; "
        f"loss={metadata['mean_loss']:.6f}"
    )
    for row in summary_rows:
        print(
            f"{row['group']}: n={row['n']}; IoU={row['damage_iou_mean']:.6f}; "
            f"Dice={row['damage_dice_mean']:.6f}; "
            f"Precision={row['damage_precision_mean']:.6f}; "
            f"Recall={row['damage_recall_mean']:.6f}"
        )
    print(
        "micro: "
        + "; ".join(f"{name}={float(value):.6f}" for name, value in micro_metrics.items())
    )
    print(f"Saved: {sample_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {metadata_path}")
    print(f"Saved {len(example_paths)} ranked example figures")


if __name__ == "__main__":
    main()
