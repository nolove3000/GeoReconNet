import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data_utils import FORMAL_SPLIT_PATH, PROJECT_ROOT, build_main_training_data, load_formal_scaler, load_formal_split
from high_resolution_clean.pytorch.evaluate import METRIC_NAMES, metrics_from_counts, summarize_records, write_csv
from high_resolution_clean.pytorch.train import CleanHighResolutionDataset
from .model import SIMPLE_DECONV_VERSION, SimpleDeconvBaseline


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate the formal simple-deconvolution baseline.")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/baselines/simple_deconv/best_simple_deconv_1024_v1.pt",
    )
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(arguments.model, map_location=device, weights_only=False)
    if checkpoint.get("architecture_version") != SIMPLE_DECONV_VERSION:
        raise ValueError("Checkpoint is not the formal simple-deconvolution baseline")
    model = SimpleDeconvBaseline(
        input_dimension=int(checkpoint["input_dimension"]),
        latent_channels=int(checkpoint["latent_channels"]),
        out_size=int(checkpoint["out_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    features, masks, geometry_types = build_main_training_data(out_size=1024, return_damage_types=True)
    split = load_formal_split(FORMAL_SPLIT_PATH, verify_dataset=True)
    if not np.array_equal(geometry_types, split["geometry_types"]):
        raise ValueError("Simple-deconvolution evaluation order does not match the frozen split")
    indices = np.asarray(split["validation_indices" if arguments.split == "validation" else "test_indices"])
    features = load_formal_scaler().transform(features).astype(np.float32)
    loader = DataLoader(
        CleanHighResolutionDataset(features[indices], masks[indices]),
        batch_size=arguments.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    records = []
    micro_counts = np.zeros(3, dtype=np.int64)
    offset = 0
    with torch.no_grad():
        for batch_features, batch_masks in loader:
            prediction = model(batch_features.to(device)) < 0.5
            truth = batch_masks.to(device) < 0.5
            axes = tuple(range(1, truth.ndim))
            tp = (prediction & truth).sum(dim=axes).cpu().numpy()
            fp = (prediction & ~truth).sum(dim=axes).cpu().numpy()
            fn = (~prediction & truth).sum(dim=axes).cpu().numpy()
            batch_metrics = metrics_from_counts(tp, fp, fn)
            micro_counts += np.asarray([tp.sum(), fp.sum(), fn.sum()], dtype=np.int64)
            for local_index in range(len(batch_features)):
                global_index = int(indices[offset + local_index])
                record = {
                    "split_position": offset + local_index,
                    "global_index": global_index,
                    "sample_id": str(split["sample_ids"][global_index]),
                    "source_file": str(split["source_files"][global_index]),
                    "geometry_type": str(split["geometry_types"][global_index]),
                    "true_positive_pixels": int(tp[local_index]),
                    "false_positive_pixels": int(fp[local_index]),
                    "false_negative_pixels": int(fn[local_index]),
                }
                record.update({name: float(batch_metrics[name][local_index]) for name in METRIC_NAMES})
                records.append(record)
            offset += len(batch_features)
    summary = summarize_records(records, arguments.bootstrap_resamples, arguments.bootstrap_seed)
    micro = metrics_from_counts(*micro_counts)
    output_directory = PROJECT_ROOT / "outputs/baselines/simple_deconv" / checkpoint["run_name"] / "evaluation" / arguments.split
    write_csv(output_directory / "per_sample_metrics.csv", records)
    write_csv(output_directory / "summary_metrics.csv", summary)
    metadata = {
        "model": str(arguments.model.relative_to(PROJECT_ROOT)),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_iou": float(checkpoint["best_validation_iou"]),
        "architecture_version": SIMPLE_DECONV_VERSION,
        "split": arguments.split,
        "sample_count": len(records),
        "micro_metrics": {key: float(value) for key, value in micro.items()},
    }
    (output_directory / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for row in summary:
        print(
            f"{row['group']}: n={row['n']}; IoU={row['damage_iou_mean']:.6f}; "
            f"Dice={row['damage_dice_mean']:.6f}; Precision={row['damage_precision_mean']:.6f}; "
            f"Recall={row['damage_recall_mean']:.6f}"
        )
    print("micro: " + "; ".join(f"{key}={float(value):.6f}" for key, value in micro.items()))


if __name__ == "__main__":
    main()
