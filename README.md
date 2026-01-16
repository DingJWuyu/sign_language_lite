# 轻量化手语识别项目 (Sign Language Lite)

这是一个基于 Uni-Sign 项目的轻量化中文手语翻译系统，专为有限算力设备（如游戏本）设计。

## 📋 目录

- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [详细步骤](#详细步骤)
- [项目结构](#项目结构)
- [常见问题](#常见问题)

---

## 💻 系统要求

### 最低配置
- **操作系统**: Windows 10/11
- **显卡**: NVIDIA GTX 1650 或更高 (至少 4GB 显存)
- **内存**: 16GB RAM
- **存储**: 10GB 可用空间
- **Python**: 3.9 或 3.10

### 推荐配置
- **显卡**: NVIDIA RTX 3060/4060/5060 或更高 (8GB+ 显存)
- **内存**: 32GB RAM

---

## 🚀 快速开始

### 第一步：安装 Anaconda（如果还没有）

1. 访问 https://www.anaconda.com/download
2. 下载 Windows 版本
3. 安装时勾选 "Add to PATH"

### 第二步：创建虚拟环境

打开 **Anaconda Prompt** 或 **PowerShell**，执行：

```powershell
# 创建名为 sign_lite 的虚拟环境
conda create -n sign_lite python=3.10 -y

# 激活环境
conda activate sign_lite
```

### 第三步：安装依赖

```powershell
# 进入项目目录
cd E:\Uni-Sign\sign_language_lite

# 安装 PyTorch (GPU版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements_lite.txt
```

### 第四步：下载预训练模型

```powershell
# 方法1: 使用 huggingface-cli（推荐）
pip install huggingface_hub
huggingface-cli download google/mt5-small --local-dir ./pretrained_weight/mt5-small

# 方法2: 手动下载
# 访问 https://huggingface.co/google/mt5-small
# 下载所有文件到 pretrained_weight/mt5-small 文件夹
```

### 第五步：准备数据

```powershell
# 创建演示数据（用于测试代码是否正常）
python data_sampling.py --mode demo
```

### 第六步：开始训练

```powershell
python train.py
```

---

## 📖 详细步骤

### 1. 环境配置详解

#### 1.1 检查 CUDA 是否可用

```python
# 在 Python 中运行
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

#### 1.2 如果 CUDA 不可用

1. 确认已安装 NVIDIA 显卡驱动
2. 重新安装 PyTorch:
   ```powershell
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### 2. 数据准备详解

#### 2.1 使用演示数据（测试用）

```powershell
python data_sampling.py --mode demo
```

这会创建 10 个模拟样本，用于测试代码是否正常运行。

#### 2.2 使用真实数据

如果你想使用真实的 CSL-Daily 数据集：

1. **下载数据集**
   - 访问 https://ustc-slr.github.io/datasets/2021_csl_daily/
   - 下载视频文件和标签文件
   - 或从 Hugging Face 下载: https://huggingface.co/datasets/ZechengLi19/CSL-News

2. **下载姿态数据**
   - 从 https://huggingface.co/ZechengLi19/Uni-Sign 下载 pose_format 数据
   
3. **放置数据**
   ```
   E:\Uni-Sign\dataset\CSL_Daily\
   ├── labels.train          # 训练标签
   ├── labels.dev            # 验证标签
   ├── labels.test           # 测试标签
   ├── sentence-crop\        # 视频文件
   │   ├── S005870_P0006_T00.mp4
   │   └── ...
   └── pose_format\          # 姿态文件
       ├── S005870_P0006_T00.pkl
       └── ...
   ```

4. **采样数据**
   ```powershell
   python data_sampling.py --mode original
   ```

### 3. 训练详解

#### 3.1 训练参数说明

编辑 `config_lite.py` 可以调整训练参数：

```python
class TrainConfig:
    batch_size = 2          # 批量大小，显存不足时减小
    gradient_accumulation = 4  # 梯度累积，模拟更大批量
    epochs = 30             # 训练轮数
    learning_rate = 5e-4    # 学习率
    max_length = 128        # 最大帧数，显存不足时减小
    use_amp = True          # 混合精度，建议开启
```

#### 3.2 显存不足解决方案

如果遇到 "CUDA out of memory" 错误：

1. 减小 `batch_size` 到 1
2. 减小 `max_length` 到 64
3. 确保 `use_amp = True`

#### 3.3 训练输出

训练过程中会保存：
- `checkpoints/best_model.pth` - 最佳模型
- `checkpoints/checkpoint_epoch_X.pth` - 定期检查点
- `checkpoints/final_model.pth` - 最终模型

### 4. 推理和测试

```powershell
# 使用训练好的模型进行推理
python inference.py --model_path checkpoints/best_model.pth
```

### 5. 实时演示（需要摄像头）

```powershell
# 需要先安装 rtmlib
cd ../demo/rtmlib-main
pip install -e .
cd ../../sign_language_lite

# 运行实时演示
python realtime_demo.py
```

---

## 📁 项目结构

```
sign_language_lite/
├── config_lite.py          # 配置文件
├── models_lite.py          # 模型定义
├── datasets_lite.py        # 数据集处理
├── train.py                # 训练脚本
├── inference.py            # 推理脚本
├── realtime_demo.py        # 实时演示
├── data_sampling.py        # 数据采样工具
├── requirements_lite.txt   # 依赖列表
├── README.md               # 本文件
├── data/
│   └── CSL_Daily_lite/     # 轻量化数据集
│       ├── labels.train
│       ├── labels.dev
│       ├── labels.test
│       └── pose_format/    # 姿态文件
├── pretrained_weight/
│   └── mt5-small/          # mT5-small 预训练模型
└── checkpoints/            # 训练保存的模型
```

---

## ❓ 常见问题

### Q1: 安装 PyTorch 时报错

**解决方案**:
```powershell
# 先卸载
pip uninstall torch torchvision torchaudio

# 重新安装（选择合适的CUDA版本）
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Q2: 找不到 mt5-small 模型

**解决方案**:
```powershell
# 安装 huggingface_hub
pip install huggingface_hub

# 下载模型
huggingface-cli download google/mt5-small --local-dir ./pretrained_weight/mt5-small
```

或者直接在代码中让 transformers 自动下载（需要网络）：
```python
# 修改 config_lite.py
mt5_path = "google/mt5-small"  # 会自动从网络下载
```

### Q3: 训练时显存不足

**解决方案**:
1. 编辑 `config_lite.py`:
   ```python
   batch_size = 1
   max_length = 64
   use_amp = True
   ```

2. 关闭其他占用显存的程序

### Q4: 数据集加载失败

**解决方案**:
1. 确保数据文件存在
2. 运行演示数据生成:
   ```powershell
   python data_sampling.py --mode demo
   ```

### Q5: transformers 或 einops 导入失败

**解决方案**:
```powershell
pip install transformers einops
```

---

## 📚 参考资源

- [Uni-Sign 原项目](https://github.com/ZechengLi19/Uni-Sign)
- [CSL-Daily 数据集](https://ustc-slr.github.io/datasets/2021_csl_daily/)
- [CSL-News 数据集](https://huggingface.co/datasets/ZechengLi19/CSL-News)
- [mT5 模型](https://huggingface.co/google/mt5-small)
- [PyTorch 官网](https://pytorch.org/)

---

## 📝 更新日志

- 2026-01-16: 初始版本

## 📧 联系方式

如有问题，请参考原项目或提交 Issue。
