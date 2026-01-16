"""轻量化配置文件"""
import os

# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型路径 - 使用绝对路径
mt5_path = os.path.join(BASE_DIR, "pretrained_weight", "mt5-small")

# 数据路径 - 使用绝对路径
train_label_path = os.path.join(BASE_DIR, "data", "CSL_Daily_lite", "labels.train")
dev_label_path = os.path.join(BASE_DIR, "data", "CSL_Daily_lite", "labels.dev")
test_label_path = os.path.join(BASE_DIR, "data", "CSL_Daily_lite", "labels.test")

pose_dir = os.path.join(BASE_DIR, "data", "CSL_Daily_lite", "pose_format")
rgb_dir = os.path.join(BASE_DIR, "data", "CSL_Daily_lite", "sentence-crop")

# 训练配置
class TrainConfig:
    # 基础配置
    batch_size = 2          # 小批量
    gradient_accumulation = 4  # 梯度累积模拟大批量
    epochs = 30
    learning_rate = 1e-4    # 降低学习率避免NaN
    weight_decay = 0.01
    
    # 序列配置
    max_length = 128        # 最大帧数 (原256)
    max_text_length = 50    # 最大文本长度
    
    # 优化配置
    use_amp = True          # 混合精度训练
    gradient_checkpointing = True  # 梯度检查点
    
    # 设备配置
    device = 'cuda'
    num_workers = 0         # Windows下建议设为0


class InferenceConfig:
    batch_size = 1
    max_length = 128
    num_beams = 3
    device = 'cuda'