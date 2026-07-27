import argparse
import csv
from pathlib import Path

import numpy as np

from common.data_utils import (
    FORMAL_SPLIT_MANIFEST_PATH,
    FORMAL_SPLIT_PATH,
    build_main_sample_metadata,
    load_formal_split,
    stratified_main_split,
    validate_split_indices,
)


SPLIT_VERSION = "formal_split_v1"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create the immutable five-stratum formal data split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--split-path", type=Path, default=FORMAL_SPLIT_PATH)
    parser.add_argument("--manifest-path", type=Path, default=FORMAL_SPLIT_MANIFEST_PATH)
    parser.add_argument("--force", action="store_true", help="Replace an existing formal split intentionally.")
    return parser.parse_args()


def split_labels(sample_count, train_indices, validation_indices, test_indices):
    labels = np.full(sample_count, "", dtype="<U10")
    labels[train_indices] = "train"
    labels[validation_indices] = "validation"
    labels[test_indices] = "test"
    if np.any(labels == ""):
        raise ValueError("Some samples were not assigned to a split")
    return labels


def save_manifest(path, metadata, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_index", "sample_id", "split", "source_file", "geometry_type", "damage_type",
        "parameter_count", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for index, sample_id in enumerate(metadata["sample_ids"]):
            count = int(metadata["parameter_counts"][index])
            row = {
                "sample_index": index,
                "sample_id": sample_id,
                "split": labels[index],
                "source_file": metadata["source_files"][index],
                "geometry_type": metadata["geometry_types"][index],
                "damage_type": metadata["damage_types"][index],
                "parameter_count": count,
            }
            for parameter_index in range(8):
                value = metadata["geometry_parameters"][index, parameter_index]
                row[f"p{parameter_index + 1}"] = "" if parameter_index >= count else f"{float(value):g}"
            writer.writerow(row)


def print_summary(stored, split_path, manifest_path):
    geometry_types = stored["geometry_types"]
    print(f"Split version: {str(stored['split_version'].item())}")
    print(f"Dataset fingerprint: {str(stored['dataset_fingerprint'].item())}")
    print(f"Split file: {split_path}")
    print(f"Manifest: {manifest_path}")
    for split_name, key in [
        ("train", "train_indices"),
        ("validation", "validation_indices"),
        ("test", "test_indices"),
    ]:
        indices = stored[key]
        values, counts = np.unique(geometry_types[indices], return_counts=True)
        composition = ", ".join(f"{value}={count}" for value, count in zip(values, counts))
        print(f"{split_name}: {len(indices)} ({composition})")


def main():
    arguments = parse_arguments()
    if arguments.split_path.exists() and not arguments.force:
        stored = load_formal_split(arguments.split_path, verify_dataset=True)
        print("Existing formal split verified; no files were replaced.")
        print_summary(stored, arguments.split_path, arguments.manifest_path)
        return

    metadata = build_main_sample_metadata()
    train_indices, validation_indices, test_indices = stratified_main_split(
        metadata["geometry_types"],
        validation_size=arguments.validation_size,
        test_size=arguments.test_size,
        seed=arguments.seed,
    )
    validate_split_indices(train_indices, validation_indices, test_indices, len(metadata["sample_ids"]))
    labels = split_labels(len(metadata["sample_ids"]), train_indices, validation_indices, test_indices)

    arguments.split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.split_path,
        split_version=SPLIT_VERSION,
        dataset_fingerprint=metadata["dataset_fingerprint"],
        canonical_source_order=np.asarray(["666.xlsx", "444.xlsx", "777.xlsx", "板2.xlsx", "555.xlsx"]),
        train_indices=np.asarray(train_indices, dtype=np.int64),
        validation_indices=np.asarray(validation_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64),
        split_assignments=labels,
        sample_ids=metadata["sample_ids"],
        source_files=metadata["source_files"],
        damage_types=metadata["damage_types"],
        geometry_types=metadata["geometry_types"],
        geometry_parameters=metadata["geometry_parameters"],
        parameter_counts=metadata["parameter_counts"],
        seed=arguments.seed,
        validation_size=arguments.validation_size,
        test_size=arguments.test_size,
    )
    save_manifest(arguments.manifest_path, metadata, labels)
    stored = load_formal_split(arguments.split_path, verify_dataset=True)
    print_summary(stored, arguments.split_path, arguments.manifest_path)


if __name__ == "__main__":
    main()
