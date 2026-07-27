import argparse
import json
from pathlib import Path

import numpy as np

from common.data_utils import (
    FORMAL_SPLIT_PATH,
    PROJECT_ROOT,
    build_main_training_data,
    load_formal_scaler,
    load_formal_split,
)
from high_resolution_clean.pytorch.evaluate import (
    METRIC_NAMES,
    metrics_from_counts,
    summarize_records,
    write_csv,
)


OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs/baselines/nearest_neighbor"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate the formal 1-NN modal-feature retrieval baseline."
    )
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--out-size", type=int, default=1024, choices=[128, 256, 512, 1024])
    parser.add_argument("--metric-batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output-name", default="formal_split_v1")
    return parser.parse_args()


def find_nearest_neighbors(training_features, query_features):
    """Return training-row positions and Euclidean distances for each query."""
    training = np.asarray(training_features, dtype=np.float64)
    queries = np.asarray(query_features, dtype=np.float64)
    if training.ndim != 2 or queries.ndim != 2 or training.shape[1] != queries.shape[1]:
        raise ValueError(
            f"Expected compatible 2D feature matrices, got {training.shape} and {queries.shape}"
        )
    squared_distances = (
        np.sum(queries * queries, axis=1, keepdims=True)
        + np.sum(training * training, axis=1)[None, :]
        - 2.0 * queries @ training.T
    )
    np.maximum(squared_distances, 0.0, out=squared_distances)
    neighbor_positions = np.argmin(squared_distances, axis=1)
    distances = np.sqrt(squared_distances[np.arange(len(queries)), neighbor_positions])
    return neighbor_positions.astype(np.int64), distances


def evaluate_retrieved_masks(masks, query_indices, neighbor_indices, split, batch_size=16):
    records = []
    micro_counts = np.zeros(3, dtype=np.int64)
    for start in range(0, len(query_indices), batch_size):
        stop = min(start + batch_size, len(query_indices))
        query_batch = masks[query_indices[start:stop]]
        neighbor_batch = masks[neighbor_indices[start:stop]]
        damage_truth = query_batch == 0
        damage_prediction = neighbor_batch == 0
        axes = tuple(range(1, damage_truth.ndim))
        true_positive = (damage_prediction & damage_truth).sum(axis=axes)
        false_positive = (damage_prediction & ~damage_truth).sum(axis=axes)
        false_negative = (~damage_prediction & damage_truth).sum(axis=axes)
        batch_metrics = metrics_from_counts(true_positive, false_positive, false_negative)
        micro_counts += np.asarray(
            [true_positive.sum(), false_positive.sum(), false_negative.sum()], dtype=np.int64
        )
        for local_index in range(stop - start):
            query_index = int(query_indices[start + local_index])
            neighbor_index = int(neighbor_indices[start + local_index])
            record = {
                "split_position": start + local_index,
                "global_index": query_index,
                "sample_id": str(split["sample_ids"][query_index]),
                "source_file": str(split["source_files"][query_index]),
                "geometry_type": str(split["geometry_types"][query_index]),
                "neighbor_global_index": neighbor_index,
                "neighbor_sample_id": str(split["sample_ids"][neighbor_index]),
                "neighbor_source_file": str(split["source_files"][neighbor_index]),
                "neighbor_geometry_type": str(split["geometry_types"][neighbor_index]),
                "same_geometry_type": bool(
                    split["geometry_types"][query_index]
                    == split["geometry_types"][neighbor_index]
                ),
                "true_positive_pixels": int(true_positive[local_index]),
                "false_positive_pixels": int(false_positive[local_index]),
                "false_negative_pixels": int(false_negative[local_index]),
            }
            record.update(
                {name: float(batch_metrics[name][local_index]) for name in METRIC_NAMES}
            )
            records.append(record)
    return records, metrics_from_counts(*micro_counts)


def main():
    arguments = parse_arguments()
    if arguments.metric_batch_size <= 0:
        raise ValueError("--metric-batch-size must be positive")
    if arguments.bootstrap_resamples < 0:
        raise ValueError("--bootstrap-resamples must be non-negative")

    features, masks, geometry_types = build_main_training_data(
        out_size=arguments.out_size, return_damage_types=True
    )
    split = load_formal_split(FORMAL_SPLIT_PATH, verify_dataset=True)
    if not np.array_equal(geometry_types, split["geometry_types"]):
        raise ValueError("Baseline data order does not match the frozen formal split")
    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    query_indices = np.asarray(
        split["validation_indices" if arguments.split == "validation" else "test_indices"],
        dtype=np.int64,
    )
    scaler = load_formal_scaler()
    standardized_features = scaler.transform(features).astype(np.float32)
    neighbor_positions, neighbor_distances = find_nearest_neighbors(
        standardized_features[train_indices], standardized_features[query_indices]
    )
    neighbor_indices = train_indices[neighbor_positions]
    if np.intersect1d(neighbor_indices, query_indices).size:
        raise RuntimeError("A query sample leaked into the 1-NN training candidate set")

    records, micro_metrics = evaluate_retrieved_masks(
        masks,
        query_indices,
        neighbor_indices,
        split,
        batch_size=arguments.metric_batch_size,
    )
    for record, distance in zip(records, neighbor_distances):
        record["feature_distance"] = float(distance)
    summary_rows = summarize_records(
        records, arguments.bootstrap_resamples, arguments.bootstrap_seed
    )
    same_type_rate = float(np.mean([record["same_geometry_type"] for record in records]))

    output_directory = OUTPUT_DIRECTORY / arguments.output_name / arguments.split
    per_sample_path = output_directory / "per_sample_metrics.csv"
    summary_path = output_directory / "summary_metrics.csv"
    metadata_path = output_directory / "evaluation_metadata.json"
    write_csv(per_sample_path, records)
    write_csv(summary_path, summary_rows)
    output_directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "method": "1-nearest-neighbor Euclidean retrieval",
        "candidate_set": "frozen training split only",
        "feature_dimension": int(standardized_features.shape[1]),
        "preprocessing": "formal training-only scaler",
        "split": arguments.split,
        "split_path": str(FORMAL_SPLIT_PATH.relative_to(PROJECT_ROOT)),
        "training_candidate_count": len(train_indices),
        "query_count": len(query_indices),
        "out_size": arguments.out_size,
        "bootstrap_resamples": arguments.bootstrap_resamples,
        "bootstrap_seed": arguments.bootstrap_seed,
        "same_geometry_type_retrieval_rate": same_type_rate,
        "mean_feature_distance": float(neighbor_distances.mean()),
        "micro_metrics": {name: float(value) for name, value in micro_metrics.items()},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

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
    print(f"same_geometry_type_retrieval_rate={same_type_rate:.6f}")
    print(f"Saved: {per_sample_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {metadata_path}")


if __name__ == "__main__":
    main()
