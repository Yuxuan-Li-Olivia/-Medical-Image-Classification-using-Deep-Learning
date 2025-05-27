# COVID-19 Medical Image Classification

## 项目简介

使用 ResNet 对 COVID-19 胸部 CT 图像进行分类，支持数据加载、模型训练、评估和可视化。

---

## 项目结构

```
COVID-19 Medical Image Classification/
    covid19_vision/
        __init__.py
        data_utils.py       # 数据加载
        model_utils.py      # 模型构建
        train_utils.py      # 训练与评估
        visualize.py        # 可视化
        feature.py          # resnet18特征图可视化
    covid19_app.py          # 主脚本
    visualization_pics/     #可视化结果图片展示
    covid19_rawdata/
        COVID-19+Aug+CGAN/  # 数据集
            train+Aug+CGAN/
            val+Aug+CGAN/
            test/
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision matplotlib seaborn scikit-learn
```

### 2. 数据集问题

我们已经从 [Kaggle](https://www.kaggle.com/datasets/mloey1/covid19-chest-ct-image-augmentation-gan-dataset) 下载数据集，解压到 `./COVID-19/COVID-19+Aug+CGAN` 目录，您无需进行额外操作。

### 3. 运行主脚本

```bash
python covid19_app.py --data_dir ./COVID-19/COVID-19+Aug+CGAN --image_size 256 --batch_size 8 --num_epochs 15 --lr 0.01 --device cuda
```

#### 参数说明

- `--data_dir`：数据集路径，默认为 `./COVID-19/COVID-19+Aug+CGAN`。
- `--image_size`：图像大小，默认为 256。
- `--batch_size`：批量大小，默认为 8。
- `--num_epochs`：训练轮数，默认为 15。
- `--lr`：学习率，默认为 0.01。
- `--device`：训练设备（`cuda` 或 `cpu`），默认为 `cuda`（如果可用）。

---

## 模块说明

### 1. 数据加载

```python
from covid19_vision import load_data

train_loader, val_loader, test_loader = load_data(data_dir="./COVID-19/COVID-19+Aug+CGAN", image_size=256, batch_size=8)
```

### 2. 模型构建

```python
from covid19_vision import build_resnet

model = build_resnet()
```

### 3. 训练与评估

```python
from covid19_vision import train_model

train_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, num_epochs=15, lr=0.01, device="cuda")
```

### 4. 可视化

```python
from covid19_vision import plot_training_curves, plot_confusion_matrix

plot_training_curves(train_losses, train_accs, val_accs)
plot_confusion_matrix(model, test_loader, device="cuda")
```

---

## 注意事项

1. 确保数据集路径正确。
2. 如需 GPU 支持，请安装 CUDA 和 cuDNN。
3. 更多问题，请参考报告。

---

## 许可证

MIT License.
