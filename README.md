# GeoReconNet

[中文说明](README.zh-CN.md)

GeoReconNet reconstructs high-resolution plate-geometry damage masks from incomplete vibration modal information. This public research package accompanies:

> Kaikai Qian, Fangqing Gao, Longbo Liu, Lei Huang, and Yihan Mao, “Structural Geometry Reconstruction from Incomplete Vibration Modal Information via Deep Learning,” 2026. <https://doi.org/10.1016/j.rineng.2026.112319>

The repository contains the data and code required to train and evaluate the plate-specific GeoReconNet model reported in the paper. It also contains the frozen split, noise-augmentation route, direct baselines, and controlled architecture ablations used in the reported comparisons.

## Included content

- `data/excel/`: five source datasets used by the plate study.
- `models/splits/`: the immutable 2,435/305/305 train/validation/test split and its manifest.
- `src/high_resolution_clean/pytorch/`: the main 1024 × 1024 GeoReconNet model, training entry point, loss, metrics, checkpointing, and evaluation.
- `src/formal_experiments/`: modal-input sensitivity, noise-augmented training, robustness evaluation, and empirical identifiability analysis.
- `src/baselines/`: 1-nearest-neighbor and simple transposed-convolution baselines.
- `src/ablations/`: controlled architecture ablations.
- `src/infrastructure/`: split and preprocessing verification utilities.
- `tests/`: model, data protocol, baseline, and ablation checks.

Generated checkpoints, scaler files, plots, and other run outputs are intentionally excluded from version control.

## Environment

Python 3.10 or newer is recommended. For GPU training, install the CUDA-enabled PyTorch build appropriate for the local NVIDIA driver from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install the remaining dependencies.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=src
```

Confirm the environment:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

## Train the main model

The paper configuration uses six modes, seven triaxial sensor locations, 132 input features, 1024 × 1024 output masks, AdamW, and validation Damage IoU for checkpoint selection.

```bash
python -m high_resolution_clean.pytorch.train \
  --epochs 200 \
  --batch-size 8 \
  --out-size 1024 \
  --dropout 0 \
  --run-name georeconnet_main
```

The best checkpoint is written to:

```text
models/pytorch/best_main_pytorch_1024_georeconnet_main.pt
```

Evaluate the frozen test split:

```bash
python -m high_resolution_clean.pytorch.evaluate \
  --model models/pytorch/best_main_pytorch_1024_georeconnet_main.pt \
  --split test
```

## Train the noise-augmented model

```bash
python -m formal_experiments.train \
  --kind noise_augmented \
  --num-modes 6 \
  --num-sensors 7 \
  --epochs 200 \
  --batch-size 8
```

Input-sensitivity models use the same entry point with `--kind sensitivity` and the desired `--num-modes` and `--num-sensors` values.

## Baselines and ablations

```bash
python -m baselines.simple_deconv.train --epochs 200 --batch-size 8
python -m baselines.nearest_neighbor.evaluate --split test
python -m ablations.architecture.train --variant single_projection --epochs 200 --batch-size 8
python -m ablations.architecture.train --variant light_decoder --epochs 200 --batch-size 8
```

All neural routes use the same frozen split, training-only preprocessing statistics, damage-aware hybrid loss, and validation-only checkpoint selection. Test data are used only for final evaluation.

## Data protocol

The canonical source order and dataset fingerprint are defined in `src/common/data_utils.py`. The stored split is verified against the five Excel files before training. Modal shapes are L2-normalized mode by mode, sign-aligned using the maximum-absolute component, concatenated with natural frequencies, and standardized using statistics fitted only on the training subset.

Damage pixels—holes and cracks—are treated as the positive class for IoU, Dice, precision, and recall. The training objective combines damage-weighted binary cross-entropy with Tversky loss:

```text
L = L_wBCE + 1.5 L_Tversky
alpha = 0.3, beta = 0.7, damage-pixel weight = 25
```

## Citation and license

Please cite the paper when using this code or data. Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff).

The repository is licensed for academic research, education, and personal non-commercial use. See [`LICENSE`](LICENSE).
