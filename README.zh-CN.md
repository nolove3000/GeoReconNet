# GeoReconNet

[English](README.md)

GeoReconNet 使用不完整振动模态信息重建板结构的高分辨率损伤几何掩码。本公开研究仓库对应论文：

> Kaikai Qian, Fangqing Gao, Longbo Liu, Lei Huang, and Yihan Mao, “Structural Geometry Reconstruction from Incomplete Vibration Modal Information via Deep Learning,” 2026. <https://doi.org/10.1016/j.rineng.2026.112319>

仓库仅提供复现论文板结构 GeoReconNet 模型训练与评价所需的数据和代码，并包含论文对比所用的冻结数据划分、噪声增强训练、直接基线和受控架构消融。

## 仓库内容

- `data/excel/`：板结构研究使用的五个源数据集。
- `models/splits/`：固定的 2,435/305/305 训练、验证和测试划分及其清单。
- `src/high_resolution_clean/pytorch/`：1024 × 1024 GeoReconNet 主模型、训练入口、损失函数、指标、检查点和评价程序。
- `src/formal_experiments/`：模态输入敏感性、噪声增强训练、鲁棒性评价和经验可辨识性分析。
- `src/baselines/`：1-近邻与简单转置卷积基线。
- `src/ablations/`：受控架构消融。
- `src/infrastructure/`：数据划分与预处理核验工具。
- `tests/`：模型、数据协议、基线和消融测试。

训练生成的检查点、标准化器、图像和其他输出不纳入版本控制。

## 环境配置

建议使用 Python 3.10 或更高版本。GPU 训练时，请先通过 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装与本机 NVIDIA 驱动匹配的 CUDA 版 PyTorch，再安装其余依赖。

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

Linux 或 macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=src
```

检查环境：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

## 训练主模型

论文配置使用 6 阶模态、7 个三向传感器位置、132 维输入和 1024 × 1024 输出掩码，采用 AdamW 训练，并仅根据验证集损伤 IoU 选择检查点。

```bash
python -m high_resolution_clean.pytorch.train \
  --epochs 200 \
  --batch-size 8 \
  --out-size 1024 \
  --dropout 0 \
  --run-name georeconnet_main
```

最佳检查点保存为：

```text
models/pytorch/best_main_pytorch_1024_georeconnet_main.pt
```

在冻结测试集上评价：

```bash
python -m high_resolution_clean.pytorch.evaluate \
  --model models/pytorch/best_main_pytorch_1024_georeconnet_main.pt \
  --split test
```

## 训练噪声增强模型

```bash
python -m formal_experiments.train \
  --kind noise_augmented \
  --num-modes 6 \
  --num-sensors 7 \
  --epochs 200 \
  --batch-size 8
```

输入敏感性模型使用相同入口，将 `--kind` 设置为 `sensitivity`，并通过 `--num-modes` 和 `--num-sensors` 指定输入配置。

## 基线与消融

```bash
python -m baselines.simple_deconv.train --epochs 200 --batch-size 8
python -m baselines.nearest_neighbor.evaluate --split test
python -m ablations.architecture.train --variant single_projection --epochs 200 --batch-size 8
python -m ablations.architecture.train --variant light_decoder --epochs 200 --batch-size 8
```

所有神经网络路线使用同一冻结划分、仅由训练集拟合的预处理统计量、损伤感知混合损失和仅基于验证集的检查点选择。测试集只用于最终评价。

## 数据协议

`src/common/data_utils.py`定义规范数据源顺序和数据指纹。训练前会根据五个 Excel 文件验证固定划分。每阶模态振型先进行 L2 归一化，再根据最大绝对值分量统一符号，随后与固有频率拼接；标准化统计量只由训练子集拟合。

损伤像素，即孔洞和裂缝，在 IoU、Dice、精确率和召回率中作为正类。训练目标由损伤加权二元交叉熵和 Tversky 损失组成：

```text
L = L_wBCE + 1.5 L_Tversky
alpha = 0.3, beta = 0.7, damage-pixel weight = 25
```

## 引用与许可

使用本仓库的代码或数据时，请引用上述论文。机器可读的引用信息见 [`CITATION.cff`](CITATION.cff)。

本仓库仅许可用于学术研究、教育及个人非商业用途，详见 [`LICENSE`](LICENSE)。
