import argparse

import numpy as np

from common.data_utils import (
    FORMAL_PREPROCESSING_VERSION,
    FORMAL_SCALER_PATH,
    FORMAL_SPLIT_PATH,
    build_modal_feature_matrix,
    fit_or_load_formal_scaler,
    iter_main_source_data,
    load_formal_split,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create or verify the shared formal modal scaler.")
    parser.add_argument("--force", action="store_true", help="Intentionally refit the formal scaler.")
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    sources = list(iter_main_source_data())
    modal = np.concatenate([source["modal"] for source in sources]).astype(np.float32)
    frequencies = np.concatenate([source["frequencies"] for source in sources]).astype(np.float32)
    features = build_modal_feature_matrix(modal, frequencies)
    stored_split = load_formal_split(FORMAL_SPLIT_PATH, verify_dataset=True)
    scaler = fit_or_load_formal_scaler(
        features,
        stored_split["train_indices"],
        force_refit=arguments.force,
    )
    standardized = scaler.transform(features).astype(np.float32)
    train = standardized[stored_split["train_indices"]]

    print(f"Preprocessing version: {FORMAL_PREPROCESSING_VERSION}")
    print(f"Feature shape: {features.shape}")
    print("Feature order: per mode [21 L2/sign-aligned components, 1 natural frequency]")
    print(f"Scaler fit samples: {len(train)} training samples only")
    print(f"Maximum absolute training-feature mean after scaling: {np.max(np.abs(train.mean(axis=0))):.3e}")
    print(f"Maximum absolute deviation from unit training-feature std: {np.max(np.abs(train.std(axis=0) - 1)):.3e}")
    print(f"Scaler artifact: {FORMAL_SCALER_PATH}")


if __name__ == "__main__":
    main()
