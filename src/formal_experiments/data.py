import hashlib
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from common.data_utils import PROJECT_ROOT, build_fusion_raw_data, build_modal_feature_matrix
from common.noise import add_measurement_noise


FORMAL_SENSOR_INDICES = {
    0: (),
    1: (6,),
    2: (5, 6),
    3: (4, 5, 6),
    4: (3, 4, 5, 6),
    5: (2, 3, 4, 5, 6),
    6: (1, 2, 3, 4, 5, 6),
    7: tuple(range(7)),
}


def select_raw_inputs(modal, frequencies, num_modes=6, num_sensors=7):
    if num_modes not in (1, 2, 3, 4, 5, 6) or num_sensors not in FORMAL_SENSOR_INDICES:
        raise ValueError("Formal configurations use 1-6 modes and 0-7 sensors")
    modal = np.asarray(modal)[:, :num_modes]
    frequencies = np.asarray(frequencies)[:, :num_modes]
    if num_sensors < 7:
        # Raw spreadsheets are axis-major: X(b1..b7), Y(b1..b7), Z(b1..b7).
        # Preserve that ordering when retaining a sensor subset.
        components = np.asarray(
            [[7 * axis + sensor for sensor in FORMAL_SENSOR_INDICES[num_sensors]] for axis in range(3)],
            dtype=np.int64,
        ).reshape(-1)
        modal = modal[:, :, components]
    return modal.astype(np.float32), frequencies.astype(np.float32)


def build_selected_features(modal, frequencies):
    # build_modal_feature_matrix accepts any observed component count only in the
    # formal 21-component route, so reproduce its exact operations for subsets.
    if modal.shape[-1] == 0:
        return np.asarray(frequencies, dtype=np.float32).reshape(len(modal), -1)
    norms = np.linalg.norm(modal, axis=-1, keepdims=True)
    normalized = modal / np.maximum(norms, np.float32(1e-10))
    dominant = np.take_along_axis(normalized, np.argmax(np.abs(normalized), axis=-1)[..., None], axis=-1)
    normalized *= np.where(dominant < 0, -1.0, 1.0).astype(np.float32)
    return np.concatenate([normalized, frequencies[..., None]], axis=-1).reshape(len(modal), -1).astype(np.float32)


def scaler_path(num_modes, num_sensors):
    version = artifact_version(num_sensors)
    return PROJECT_ROOT / "models/scalers/sensitivity" / f"modes_{num_modes}_sensors_{num_sensors}_{version}.pkl"


def artifact_version(num_sensors):
    if num_sensors == 7:
        return "v1"
    if num_sensors == 0:
        return "v2"
    return "v3"


def fit_or_load_scaler(features, train_indices, num_modes, num_sensors):
    path = scaler_path(num_modes, num_sensors)
    fingerprint = hashlib.sha256(np.ascontiguousarray(train_indices, dtype=np.int64).tobytes()).hexdigest()
    metadata = {"num_modes": num_modes, "num_sensors": num_sensors, "feature_dimension": features.shape[1], "train_indices_sha256": fingerprint, "sensor_indices_zero_based": list(FORMAL_SENSOR_INDICES[num_sensors])}
    if num_sensors < 7:
        metadata["component_order"] = "axis_major_xyz"
    if 0 < num_sensors < 7:
        metadata["sensor_subset_protocol"] = "free_edge_tail_b8_minus_n_through_b7_v1"
    if path.is_file():
        bundle = joblib.load(path)
        if any(bundle.get(key) != value for key, value in metadata.items()):
            raise ValueError(f"Sensitivity scaler metadata mismatch: {path}")
        return bundle["scaler"]
    path.parent.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler().fit(features[train_indices])
    joblib.dump({**metadata, "scaler": scaler}, path)
    return scaler


class FixedFeatureDataset(Dataset):
    def __init__(self, features, masks):
        self.features = np.asarray(features, np.float32); self.masks = np.asarray(masks, np.uint8)
    def __len__(self): return len(self.features)
    def __getitem__(self, index): return torch.from_numpy(self.features[index]), torch.from_numpy(self.masks[index][None])


class NoiseAugmentedDataset(Dataset):
    def __init__(self, modal, frequencies, masks, scaler, global_indices, seed=42, max_mode_noise=.05, max_frequency_noise=.01, randomize_levels=True):
        self.modal=modal; self.frequencies=frequencies; self.masks=masks; self.scaler=scaler
        self.global_indices=np.asarray(global_indices); self.seed=seed; self.epoch=0
        self.max_mode_noise=max_mode_noise; self.max_frequency_noise=max_frequency_noise
        self.randomize_levels=randomize_levels
    def set_epoch(self, epoch): self.epoch=epoch
    def __len__(self): return len(self.modal)
    def __getitem__(self, index):
        rng=np.random.default_rng(self.seed + self.epoch * 100_003 + int(self.global_indices[index]))
        mode_level=rng.uniform(0.0,self.max_mode_noise) if self.randomize_levels else self.max_mode_noise
        frequency_level=rng.uniform(0.0,self.max_frequency_noise) if self.randomize_levels else self.max_frequency_noise
        modal,frequency=add_measurement_noise(self.modal[index],self.frequencies[index],mode_level,frequency_level,rng)
        feature=build_selected_features(modal[None],frequency[None])
        feature=self.scaler.transform(feature)[0].astype(np.float32)
        return torch.from_numpy(feature),torch.from_numpy(self.masks[index][None].astype(np.uint8))


def load_raw_1024(): return build_fusion_raw_data(out_size=1024)
