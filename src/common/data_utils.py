import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "excel"
BASE_POSITIONS = [16, 20, 13, 16, 20, 16]
FORMAL_SPLIT_PATH = PROJECT_ROOT / "models" / "splits" / "formal_split_v1.npz"
FORMAL_SPLIT_MANIFEST_PATH = PROJECT_ROOT / "models" / "splits" / "formal_split_v1_manifest.csv"
FORMAL_SCALER_PATH = PROJECT_ROOT / "models" / "scalers" / "formal_modal_l2_sign_v1.pkl"
FORMAL_PREPROCESSING_VERSION = "modal_l2_maxabs_sign_zscore_v1"


# This order is the canonical global sample order for every formal experiment.
# Do not reorder these entries without creating a new split version.
MAIN_DATA_SOURCES = [
    ("666.xlsx", 4, 4, 5, "upward_crack", "upward_crack_1mm"),
    ("444.xlsx", 4, 4, 5, "upward_crack", "upward_crack_5mm"),
    ("777.xlsx", 4, 4, 5, "circular_hole", "circular_hole"),
    ("板2.xlsx", 8, 8, 9, "double_crack", "double_crack_1mm"),
    ("555.xlsx", 4, 4, 5, "downward_crack", "downward_crack_5mm"),
]
def load_grouped_excel(file_path, parameter_columns, frequency_column, modal_start_column):
    dataframe = pd.read_excel(file_path)
    parameters_list = []
    frequencies_list = []
    modal_list = []

    for row_start in range(0, len(dataframe), 6):
        group = dataframe.iloc[row_start : row_start + 6]
        if len(group) != 6:
            continue

        modal = group.iloc[:, modal_start_column:].to_numpy(dtype=np.float32)
        if modal.shape != (6, 21):
            continue

        parameters_list.append(group.iloc[0, :parameter_columns].to_numpy(dtype=np.float32))
        frequencies_list.append(group.iloc[:, frequency_column].to_numpy(dtype=np.float32))
        modal_list.append(modal)

    if not modal_list:
        raise ValueError(f"No valid six-mode samples found in {file_path}")

    return (
        np.asarray(parameters_list, dtype=np.float32),
        np.asarray(frequencies_list, dtype=np.float32),
        np.asarray(modal_list, dtype=np.float32),
    )


def normalize_modal_by_reference(modal_data):
    """Legacy preprocessing retained only for reproducing historical experiments."""
    normalized = np.zeros_like(modal_data, dtype=np.float32)
    for sample_index, sample in enumerate(modal_data):
        for mode_index, reference_index in enumerate(BASE_POSITIONS):
            reference_value = sample[mode_index, reference_index]
            if reference_value == 0:
                reference_value = 1e-10
            normalized[sample_index, mode_index] = sample[mode_index] / reference_value
    return normalized


def normalize_modal_l2_sign(modal_data, epsilon=1e-10):
    """L2-normalize each observed mode and make its max-absolute component positive."""
    modal = np.asarray(modal_data, dtype=np.float32)
    if modal.ndim not in (2, 3) or modal.shape[-1] != 21:
        raise ValueError(f"Expected modal shape (modes, 21) or (samples, modes, 21), got {modal.shape}")
    if not np.all(np.isfinite(modal)):
        raise ValueError("Modal data contain non-finite values")

    norms = np.linalg.norm(modal, axis=-1, keepdims=True)
    normalized = modal / np.maximum(norms, np.float32(epsilon))
    maximum_indices = np.argmax(np.abs(normalized), axis=-1)
    dominant = np.take_along_axis(normalized, maximum_indices[..., np.newaxis], axis=-1)
    signs = np.where(dominant < 0, -1.0, 1.0).astype(np.float32)
    return (normalized * signs).astype(np.float32)


def build_modal_feature(modal, frequencies):
    """Create one feature vector ordered as 21 modal components then frequency per mode."""
    modal = np.asarray(modal, dtype=np.float32)
    frequencies = np.asarray(frequencies, dtype=np.float32)
    if modal.ndim != 2 or modal.shape[1] != 21:
        raise ValueError(f"Expected one modal sample with shape (modes, 21), got {modal.shape}")
    if frequencies.shape != (modal.shape[0],):
        raise ValueError(f"Expected {modal.shape[0]} frequencies, got {frequencies.shape}")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("Frequency data contain non-finite values")
    normalized_modal = normalize_modal_l2_sign(modal)
    return np.concatenate([normalized_modal, frequencies[:, np.newaxis]], axis=1).reshape(-1).astype(np.float32)


def build_modal_feature_matrix(modal_data, frequency_data):
    """Vectorized formal feature construction for a batch of raw modal samples."""
    modal = np.asarray(modal_data, dtype=np.float32)
    frequencies = np.asarray(frequency_data, dtype=np.float32)
    if modal.ndim != 3 or modal.shape[2] != 21:
        raise ValueError(f"Expected modal batch shape (samples, modes, 21), got {modal.shape}")
    if frequencies.shape != modal.shape[:2]:
        raise ValueError(f"Expected frequency shape {modal.shape[:2]}, got {frequencies.shape}")
    normalized_modal = normalize_modal_l2_sign(modal)
    features = np.concatenate([normalized_modal, frequencies[..., np.newaxis]], axis=2)
    return features.reshape(len(modal), -1).astype(np.float32)


def _index_fingerprint(indices):
    return hashlib.sha256(np.ascontiguousarray(indices, dtype=np.int64).tobytes()).hexdigest()


def fit_or_load_formal_scaler(
    clean_features,
    train_indices,
    split_path=FORMAL_SPLIT_PATH,
    scaler_path=FORMAL_SCALER_PATH,
    force_refit=False,
):
    """Fit the shared z-score scaler on training samples only, or verify and reuse it."""
    clean_features = np.asarray(clean_features, dtype=np.float32)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if clean_features.ndim != 2 or clean_features.shape[1] != 132:
        raise ValueError(f"Expected formal clean features with shape (samples, 132), got {clean_features.shape}")
    if not np.all(np.isfinite(clean_features)):
        raise ValueError("Formal feature matrix contains non-finite values")
    stored_split = load_formal_split(split_path, verify_dataset=True)
    if not np.array_equal(train_indices, stored_split["train_indices"]):
        raise ValueError("Scaler training indices do not match the frozen formal training split")

    expected = {
        "preprocessing_version": FORMAL_PREPROCESSING_VERSION,
        "dataset_fingerprint": str(stored_split["dataset_fingerprint"].item()),
        "train_indices_fingerprint": _index_fingerprint(train_indices),
        "feature_dimension": 132,
        "feature_order": "per_mode:[21_l2_sign_aligned_components,frequency]",
        "fit_sample_count": len(train_indices),
    }
    scaler_path = Path(scaler_path)
    if scaler_path.is_file() and not force_refit:
        bundle = joblib.load(scaler_path)
        if not isinstance(bundle, dict) or "scaler" not in bundle:
            raise ValueError(f"Formal scaler artifact has an unsupported format: {scaler_path}")
        for name, value in expected.items():
            if bundle.get(name) != value:
                raise ValueError(f"Formal scaler metadata mismatch for `{name}`")
        return bundle["scaler"]

    scaler = StandardScaler().fit(clean_features[train_indices])
    bundle = {**expected, "scaler": scaler}
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, scaler_path)
    return scaler


def load_formal_scaler(
    split_path=FORMAL_SPLIT_PATH,
    scaler_path=FORMAL_SCALER_PATH,
):
    """Load and verify the shared formal scaler without fitting it."""
    stored_split = load_formal_split(split_path, verify_dataset=True)
    scaler_path = Path(scaler_path)
    if not scaler_path.is_file():
        raise FileNotFoundError(
            "Formal scaler not found: "
            f"{scaler_path}. Run `PYTHONPATH=src python -m infrastructure.create_formal_preprocessing` first."
        )
    bundle = joblib.load(scaler_path)
    expected = {
        "preprocessing_version": FORMAL_PREPROCESSING_VERSION,
        "dataset_fingerprint": str(stored_split["dataset_fingerprint"].item()),
        "train_indices_fingerprint": _index_fingerprint(stored_split["train_indices"]),
        "feature_dimension": 132,
        "feature_order": "per_mode:[21_l2_sign_aligned_components,frequency]",
        "fit_sample_count": len(stored_split["train_indices"]),
    }
    if not isinstance(bundle, dict) or "scaler" not in bundle:
        raise ValueError(f"Formal scaler artifact has an unsupported format: {scaler_path}")
    for name, value in expected.items():
        if bundle.get(name) != value:
            raise ValueError(f"Formal scaler metadata mismatch for `{name}`")
    return bundle["scaler"]


def create_grid(out_size, plate_length_x=200.0, plate_length_y=100.0):
    x_coordinates = np.arange(0, plate_length_x, plate_length_x / out_size, dtype=np.float32)[:out_size]
    y_coordinates = np.arange(0, plate_length_y, plate_length_y / out_size, dtype=np.float32)[:out_size]
    return np.meshgrid(x_coordinates, y_coordinates)


def generate_upward_crack_mask(parameters, out_size):
    x1, y1, length, width = [float(value) for value in parameters]
    grid_x, grid_y = create_grid(out_size)
    mask = np.ones_like(grid_x, dtype=np.uint8)
    crack = (grid_x >= x1) & (grid_x <= x1 + width) & (grid_y >= y1) & (grid_y <= y1 + length)
    mask[crack] = 0.0
    return mask


def generate_downward_crack_mask(parameters, out_size):
    x1, y1, length, width = [float(value) for value in parameters]
    grid_x, grid_y = create_grid(out_size)
    mask = np.ones_like(grid_x, dtype=np.uint8)
    crack = (grid_x >= x1) & (grid_x <= x1 + width) & (grid_y <= y1) & (grid_y >= y1 - length)
    mask[crack] = 0.0
    return mask


def generate_hole_mask(parameters, out_size):
    x1, y1, _length, radius = [float(value) for value in parameters]
    grid_x, grid_y = create_grid(out_size)
    mask = np.ones_like(grid_x, dtype=np.uint8)
    hole = (grid_x - x1) ** 2 + (grid_y - y1) ** 2 <= radius**2
    mask[hole] = 0.0
    return mask


def generate_double_crack_mask(parameters, out_size):
    x1, y1, length1, width1, x2, y2, length2, width2 = [float(value) for value in parameters]
    grid_x, grid_y = create_grid(out_size)
    mask = np.ones_like(grid_x, dtype=np.uint8)
    crack1 = (grid_x >= x1) & (grid_x <= x1 + width1) & (grid_y >= y1) & (grid_y <= y1 + length1)
    crack2 = (grid_x >= x2) & (grid_x <= x2 + width2) & (grid_y >= y2) & (grid_y <= y2 + length2)
    mask[crack1 | crack2] = 0.0
    return mask


MASK_FUNCTIONS = {
    "upward_crack": generate_upward_crack_mask,
    "downward_crack": generate_downward_crack_mask,
    "circular_hole": generate_hole_mask,
    "double_crack": generate_double_crack_mask,
}


def iter_main_source_data():
    """Yield every formal data source in the canonical global sample order."""
    for file_name, parameter_columns, frequency_column, modal_start_column, damage_type, geometry_type in MAIN_DATA_SOURCES:
        file_path = DATA_DIR / file_name
        parameters, frequencies, modal = load_grouped_excel(
            file_path, parameter_columns, frequency_column, modal_start_column
        )
        yield {
            "file_name": file_name,
            "file_path": file_path,
            "parameter_columns": parameter_columns,
            "frequency_column": frequency_column,
            "modal_start_column": modal_start_column,
            "damage_type": damage_type,
            "geometry_type": geometry_type,
            "mask_function": MASK_FUNCTIONS[damage_type],
            "parameters": parameters,
            "frequencies": frequencies,
            "modal": modal,
        }


def build_main_sample_metadata():
    """Build traceable sample metadata and a fingerprint for the formal split."""
    sample_ids = []
    source_files = []
    damage_types = []
    geometry_types = []
    parameter_rows = []
    parameter_counts = []
    fingerprint = hashlib.sha256()

    for source in iter_main_source_data():
        parameters = source["parameters"]
        frequencies = np.ascontiguousarray(source["frequencies"], dtype=np.float32)
        modal = np.ascontiguousarray(source["modal"], dtype=np.float32)
        fingerprint.update(source["file_name"].encode("utf-8"))
        fingerprint.update(source["geometry_type"].encode("utf-8"))
        fingerprint.update(np.ascontiguousarray(parameters, dtype=np.float32).tobytes())
        fingerprint.update(frequencies.tobytes())
        fingerprint.update(modal.tobytes())

        for local_index, values in enumerate(parameters):
            value_text = "_".join(f"{float(value):g}" for value in values)
            sample_ids.append(f"{Path(source['file_name']).stem}:{local_index:04d}:{value_text}")
            source_files.append(source["file_name"])
            damage_types.append(source["damage_type"])
            geometry_types.append(source["geometry_type"])
            padded = np.full(8, np.nan, dtype=np.float32)
            padded[: len(values)] = values
            parameter_rows.append(padded)
            parameter_counts.append(len(values))

    return {
        "sample_ids": np.asarray(sample_ids),
        "source_files": np.asarray(source_files),
        "damage_types": np.asarray(damage_types),
        "geometry_types": np.asarray(geometry_types),
        "geometry_parameters": np.asarray(parameter_rows, dtype=np.float32),
        "parameter_counts": np.asarray(parameter_counts, dtype=np.int8),
        "dataset_fingerprint": fingerprint.hexdigest(),
    }


def validate_split_indices(train_indices, validation_indices, test_indices, sample_count):
    arrays = [np.asarray(values, dtype=np.int64) for values in (train_indices, validation_indices, test_indices)]
    combined = np.concatenate(arrays)
    if len(combined) != sample_count:
        raise ValueError(f"Split contains {len(combined)} indices for {sample_count} samples")
    if len(np.unique(combined)) != sample_count:
        raise ValueError("Train, validation, and test indices are not mutually exclusive")
    if not np.array_equal(np.sort(combined), np.arange(sample_count)):
        raise ValueError("Split indices do not cover the canonical dataset exactly")


def load_formal_split(split_path=FORMAL_SPLIT_PATH, verify_dataset=True):
    """Load the immutable formal split and optionally verify it against the Excel data."""
    split_path = Path(split_path)
    if not split_path.is_file():
        raise FileNotFoundError(
            "Formal split not found: "
            f"{split_path}. Run `PYTHONPATH=src python -m infrastructure.create_formal_split` first."
        )
    stored = np.load(split_path)
    required = {
        "train_indices", "validation_indices", "test_indices", "sample_ids",
        "source_files", "damage_types", "geometry_types", "geometry_parameters",
        "parameter_counts", "dataset_fingerprint", "seed", "validation_size", "test_size",
    }
    missing = required - set(stored.files)
    if missing:
        raise ValueError(f"Formal split is missing fields: {sorted(missing)}")
    validate_split_indices(
        stored["train_indices"], stored["validation_indices"], stored["test_indices"], len(stored["sample_ids"])
    )
    if verify_dataset:
        current = build_main_sample_metadata()
        stored_fingerprint = str(stored["dataset_fingerprint"].item())
        if stored_fingerprint != current["dataset_fingerprint"]:
            raise ValueError(
                "Formal split fingerprint does not match the current Excel data or canonical source order"
            )
        for name in ["sample_ids", "source_files", "damage_types", "geometry_types", "parameter_counts"]:
            if not np.array_equal(stored[name], current[name]):
                raise ValueError(f"Formal split field `{name}` does not match the current dataset")
        if not np.allclose(stored["geometry_parameters"], current["geometry_parameters"], equal_nan=True):
            raise ValueError("Formal split geometry parameters do not match the current dataset")
    return stored


def build_main_training_data(out_size=1024, return_damage_types=False):
    feature_batches = []
    mask_batches = []
    damage_type_batches = []

    for source in iter_main_source_data():
        parameters = source["parameters"]
        frequencies = source["frequencies"]
        modal = source["modal"]
        features = build_modal_feature_matrix(modal, frequencies)
        # Keep masks compact in host memory. They are cast to float32 by the
        # TensorFlow input pipeline immediately before each batch is trained.
        masks = np.asarray(
            [source["mask_function"](sample_parameters, out_size) for sample_parameters in parameters],
            dtype=np.uint8,
        )
        feature_batches.append(features.astype(np.float32))
        mask_batches.append(masks)
        damage_type_batches.append(np.repeat(source["geometry_type"], len(masks)))

    result = np.concatenate(feature_batches), np.concatenate(mask_batches)
    if return_damage_types:
        return *result, np.concatenate(damage_type_batches)
    return result


def stratified_main_split(damage_types, validation_size=0.1, test_size=0.1, seed=42):
    """Return deterministic train/validation/test indices stratified by damage type."""
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be positive and sum to less than 1")

    indices = np.arange(len(damage_types))
    train_validation_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=damage_types,
    )
    relative_validation_size = validation_size / (1 - test_size)
    train_indices, validation_indices = train_test_split(
        train_validation_indices,
        test_size=relative_validation_size,
        random_state=seed,
        stratify=damage_types[train_validation_indices],
    )
    return train_indices, validation_indices, test_indices


def build_fusion_raw_data(out_size=512):
    frequency_batches = []
    modal_batches = []
    mask_batches = []

    for source in iter_main_source_data():
        parameters = source["parameters"]
        frequency_batches.append(source["frequencies"])
        modal_batches.append(source["modal"])
        mask_batches.append(
            np.asarray([source["mask_function"](sample_parameters, out_size) for sample_parameters in parameters])
        )

    return (
        np.concatenate(modal_batches).astype(np.float32),
        np.concatenate(frequency_batches).astype(np.float32),
        np.concatenate(mask_batches).astype(np.float32),
    )
