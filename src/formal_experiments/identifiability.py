"""Empirical modal ambiguity and local perturbation-sensitivity analysis."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from baselines.nearest_neighbor.evaluate import find_nearest_neighbors
from common.data_utils import (
    FORMAL_SPLIT_PATH,
    PROJECT_ROOT,
    build_fusion_raw_data,
    build_main_training_data,
    load_formal_scaler,
    load_formal_split,
)
from common.noise import add_measurement_noise
from common.plot_style import configure_plot_style
from high_resolution_clean.pytorch.model import HighResGeneratorTorch
from .data import build_selected_features


configure_plot_style()

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "models/pytorch/best_main_pytorch_1024_no_norm_dropout_000_v5_continue150.pt"
)
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs/formal_experiments/identifiability_v1"


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--mask-size", type=int, default=256, choices=(128, 256, 512))
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--distance-quantile", type=float, default=0.25)
    parser.add_argument("--perturbations", type=int, default=50)
    parser.add_argument("--mode-noise", type=float, default=0.01)
    parser.add_argument("--frequency-noise", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--inference-batch-size", type=int, default=5)
    return parser.parse_args()


def damage_iou(left, right):
    left_damage = np.asarray(left) == 0
    right_damage = np.asarray(right) == 0
    intersection = np.logical_and(left_damage, right_damage).sum()
    union = np.logical_or(left_damage, right_damage).sum()
    return float(intersection / max(union, 1))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_ambiguity_examples(records, example_count, distance_quantile):
    cutoff = float(np.quantile([row["feature_distance"] for row in records], distance_quantile))
    close = [row for row in records if row["feature_distance"] <= cutoff]
    selected = sorted(close, key=lambda row: (row["geometry_iou"], row["feature_distance"]))[:example_count]
    return selected, cutoff


def save_ambiguity_figure(selected, masks, split, output_path):
    figure, axes = plt.subplots(len(selected), 3, figsize=(12, 3.2 * len(selected)), squeeze=False, constrained_layout=True)
    for row_index, record in enumerate(selected):
        query = masks[record["query_global_index"]]
        neighbor = masks[record["neighbor_global_index"]]
        panels = (
            (query == 0, "Test geometry", "Blues", 0, 1),
            (neighbor == 0, "Nearest training geometry", "Blues", 0, 1),
            (query != neighbor, "Geometric disagreement", "Blues", 0, 1),
        )
        for axis, (values, title, color_map, minimum, maximum) in zip(axes[row_index], panels):
            axis.imshow(values, origin="lower", extent=(0, 200, 0, 100), aspect="equal", cmap=color_map, vmin=minimum, vmax=maximum, interpolation="nearest")
            axis.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
            axis.grid(False, which="both")
        axes[row_index, 0].set_ylabel(
            f"{split['geometry_types'][record['query_global_index']]}\n"
            f"d={record['feature_distance']:.3f}, IoU={record['geometry_iou']:.3f}\ny (mm)"
        )
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def pool_probability(probability, output_size):
    probability = np.asarray(probability)
    factor = probability.shape[-1] // output_size
    return probability.reshape(output_size, factor, output_size, factor).mean(axis=(1, 3))


def perturbation_statistics(model, scaler, modal, frequencies, source_index, arguments, device):
    features = []
    for repetition in range(arguments.perturbations):
        generator = np.random.default_rng(arguments.seed + repetition * 1_000_003 + int(source_index))
        noisy_modal, noisy_frequency = add_measurement_noise(
            modal[source_index], frequencies[source_index], arguments.mode_noise,
            arguments.frequency_noise, generator,
        )
        features.append(build_selected_features(noisy_modal[None], noisy_frequency[None])[0])
    features = scaler.transform(np.asarray(features)).astype(np.float32)
    pooled = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), arguments.inference_batch_size):
            solid = model(torch.from_numpy(features[start:start + arguments.inference_batch_size]).to(device)).cpu().numpy()[:, 0]
            pooled.extend(pool_probability(1 - item, arguments.mask_size) for item in solid)
        clean_feature = scaler.transform(build_selected_features(modal[source_index:source_index + 1], frequencies[source_index:source_index + 1])).astype(np.float32)
        clean_damage = 1 - model(torch.from_numpy(clean_feature).to(device)).cpu().numpy()[0, 0]
    pooled = np.asarray(pooled)
    return pool_probability(clean_damage, arguments.mask_size), pooled.mean(axis=0), pooled.std(axis=0, ddof=1)


def save_sensitivity_figure(selected, masks, modal, frequencies, model, scaler, arguments, device, output_path):
    figure, axes = plt.subplots(len(selected), 4, figsize=(16, 3.2 * len(selected)), squeeze=False, constrained_layout=True)
    maxima = []
    statistics = []
    for record in selected:
        statistics.append(perturbation_statistics(model, scaler, modal, frequencies, record["query_global_index"], arguments, device))
        maxima.append(float(statistics[-1][2].max()))
    shared_std_maximum = max(max(maxima), 0.01)
    for row_index, (record, (clean_damage, mean_damage, standard_deviation)) in enumerate(zip(selected, statistics)):
        panels = (
            (masks[record["query_global_index"]] == 0, "Ground-truth damage", 0, 1),
            (clean_damage, "Clean-input prediction", 0, 1),
            (mean_damage, "Perturbed prediction mean", 0, 1),
            (standard_deviation, "Perturbation standard deviation", 0, shared_std_maximum),
        )
        for axis, (values, title, minimum, maximum) in zip(axes[row_index], panels):
            image = axis.imshow(values, origin="lower", extent=(0, 200, 0, 100), aspect="equal", cmap="Blues", vmin=minimum, vmax=maximum, interpolation="nearest")
            axis.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
            axis.grid(False, which="both")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axes[row_index, 0].set_ylabel(f"test index {record['query_global_index']}\ny (mm)")
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return shared_std_maximum


def main():
    arguments = parse_arguments()
    if not 0 < arguments.distance_quantile <= 1:
        raise ValueError("--distance-quantile must be in (0, 1]")
    if arguments.examples <= 0 or arguments.perturbations < 2:
        raise ValueError("Examples must be positive and perturbations must be at least two")
    if not arguments.model.is_file():
        raise FileNotFoundError(arguments.model)

    features, masks, geometry_types = build_main_training_data(arguments.mask_size, return_damage_types=True)
    modal, frequencies, _ = build_fusion_raw_data(arguments.mask_size)
    split = load_formal_split(FORMAL_SPLIT_PATH, verify_dataset=True)
    if not np.array_equal(geometry_types, split["geometry_types"]):
        raise RuntimeError("Dataset order does not match the frozen split")
    train_indices = np.asarray(split["train_indices"], dtype=np.int64)
    test_indices = np.asarray(split["test_indices"], dtype=np.int64)
    scaler = load_formal_scaler()
    standardized = scaler.transform(features).astype(np.float32)
    positions, distances = find_nearest_neighbors(standardized[train_indices], standardized[test_indices])
    neighbors = train_indices[positions]

    records = []
    for query_index, neighbor_index, distance in zip(test_indices, neighbors, distances):
        records.append({
            "query_global_index": int(query_index),
            "query_sample_id": str(split["sample_ids"][query_index]),
            "query_geometry_type": str(split["geometry_types"][query_index]),
            "neighbor_global_index": int(neighbor_index),
            "neighbor_sample_id": str(split["sample_ids"][neighbor_index]),
            "neighbor_geometry_type": str(split["geometry_types"][neighbor_index]),
            "same_geometry_type": bool(split["geometry_types"][query_index] == split["geometry_types"][neighbor_index]),
            "feature_distance": float(distance),
            "geometry_iou": damage_iou(masks[query_index], masks[neighbor_index]),
        })
    selected, cutoff = select_ambiguity_examples(records, arguments.examples, arguments.distance_quantile)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIRECTORY / "nearest_training_ambiguity.csv", records)
    write_csv(OUTPUT_DIRECTORY / "selected_ambiguity_examples.csv", selected)
    save_ambiguity_figure(selected, masks, split, OUTPUT_DIRECTORY / "modal_ambiguity_examples.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(arguments.model, map_location=device, weights_only=False)
    model = HighResGeneratorTorch(
        input_dimension=int(checkpoint.get("input_dimension", 132)), out_size=1024,
        base_channels=int(checkpoint.get("base_channels", 256)), dropout=float(checkpoint.get("dropout", 0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    std_maximum = save_sensitivity_figure(
        selected, masks, modal, frequencies, model, scaler, arguments, device,
        OUTPUT_DIRECTORY / "local_perturbation_sensitivity.png",
    )
    distances_array = np.asarray([row["feature_distance"] for row in records])
    ious_array = np.asarray([row["geometry_iou"] for row in records])
    metadata = {
        "split": "formal test queries versus formal training candidates",
        "test_queries": len(test_indices),
        "training_candidates": len(train_indices),
        "feature_dimension": standardized.shape[1],
        "distance_quantile": arguments.distance_quantile,
        "close_distance_cutoff": cutoff,
        "nearest_distance_mean": float(distances_array.mean()),
        "nearest_distance_median": float(np.median(distances_array)),
        "nearest_geometry_iou_mean": float(ious_array.mean()),
        "nearest_geometry_iou_median": float(np.median(ious_array)),
        "nearest_geometry_iou_below_0_5_rate": float(np.mean(ious_array < 0.5)),
        "selected_examples": selected,
        "perturbations": arguments.perturbations,
        "mode_noise": arguments.mode_noise,
        "frequency_noise": arguments.frequency_noise,
        "sensitivity_map_shared_std_maximum": std_maximum,
        "interpretation": "Empirical finite-dataset ambiguity and local input-perturbation sensitivity; not calibrated predictive uncertainty or proof of global mathematical non-uniqueness.",
    }
    (OUTPUT_DIRECTORY / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Saved: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
