import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.data_utils import FORMAL_SPLIT_PATH, PROJECT_ROOT, build_main_training_data, load_formal_scaler, load_formal_split
from high_resolution_clean.pytorch.evaluate import METRIC_NAMES, metrics_from_counts, summarize_records, write_csv
from high_resolution_clean.pytorch.train import CleanHighResolutionDataset
from .model import ABLATION_VERSION, ArchitectureAblationGenerator


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate a validation-selected architecture ablation.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model.resolve()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if checkpoint.get("architecture_version") != ABLATION_VERSION:
        raise ValueError("Checkpoint is not a formal architecture ablation")
    variant = checkpoint["variant"]
    model = ArchitectureAblationGenerator(variant).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    features, masks, geometry_types = build_main_training_data(out_size=1024, return_damage_types=True)
    split = load_formal_split(FORMAL_SPLIT_PATH, verify_dataset=True)
    if not np.array_equal(geometry_types, split["geometry_types"]):
        raise ValueError("Ablation evaluation order does not match the frozen split")
    indices = np.asarray(split["validation_indices" if args.split == "validation" else "test_indices"])
    features = load_formal_scaler().transform(features).astype(np.float32)
    loader = DataLoader(CleanHighResolutionDataset(features[indices], masks[indices]), batch_size=args.batch_size, shuffle=False, pin_memory=device.type == "cuda")
    records, micro_counts, offset = [], np.zeros(3, dtype=np.int64), 0
    with torch.no_grad():
        for batch_features, batch_masks in loader:
            prediction = model(batch_features.to(device)) < 0.5
            truth = batch_masks.to(device) < 0.5
            axes = tuple(range(1, truth.ndim))
            tp = (prediction & truth).sum(dim=axes).cpu().numpy()
            fp = (prediction & ~truth).sum(dim=axes).cpu().numpy()
            fn = (~prediction & truth).sum(dim=axes).cpu().numpy()
            metrics = metrics_from_counts(tp, fp, fn)
            micro_counts += np.asarray([tp.sum(), fp.sum(), fn.sum()], dtype=np.int64)
            for local_index in range(len(batch_features)):
                global_index = int(indices[offset + local_index])
                record = {"split_position": offset + local_index, "global_index": global_index, "sample_id": str(split["sample_ids"][global_index]), "source_file": str(split["source_files"][global_index]), "geometry_type": str(split["geometry_types"][global_index]), "true_positive_pixels": int(tp[local_index]), "false_positive_pixels": int(fp[local_index]), "false_negative_pixels": int(fn[local_index])}
                record.update({name: float(metrics[name][local_index]) for name in METRIC_NAMES})
                records.append(record)
            offset += len(batch_features)
    summary = summarize_records(records, args.bootstrap_resamples, args.bootstrap_seed)
    micro = metrics_from_counts(*micro_counts)
    output_dir = PROJECT_ROOT / "outputs/ablations/architecture" / variant / "evaluation" / args.split
    write_csv(output_dir / "per_sample_metrics.csv", records)
    write_csv(output_dir / "summary_metrics.csv", summary)
    (output_dir / "evaluation_metadata.json").write_text(json.dumps({"model": str(model_path.relative_to(PROJECT_ROOT)), "variant": variant, "checkpoint_epoch": int(checkpoint["epoch"]), "best_validation_iou": float(checkpoint["best_validation_iou"]), "parameter_count": int(checkpoint["parameter_count"]), "architecture_version": ABLATION_VERSION, "split": args.split, "sample_count": len(records), "micro_metrics": {k: float(v) for k, v in micro.items()}}, indent=2), encoding="utf-8")
    for row in summary:
        print(f"{row['group']}: n={row['n']}; IoU={row['damage_iou_mean']:.6f}; Dice={row['damage_dice_mean']:.6f}; Precision={row['damage_precision_mean']:.6f}; Recall={row['damage_recall_mean']:.6f}")
    print("micro: " + "; ".join(f"{key}={float(value):.6f}" for key, value in micro.items()))


if __name__ == "__main__":
    main()
