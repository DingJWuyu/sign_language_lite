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
    batch_size = 4          # 根据显存决定
    gradient_accumulation = 16  # 4 * 16 = 64 有效batch
    epochs = 50             # 增加训练轮数
    learning_rate = 2e-4    # 提高学习率以适应解冻的Encoder
    weight_decay = 0.01
    
    # 学习率调度
    warmup_epochs = 5       # 学习率预热轮数
    lr_scheduler = 'cosine_warmup'  # 预热+余弦退火
    min_lr = 1e-6           # 最小学习率
    
    # 标签平滑
    label_smoothing = 0.1   # 标签平滑，防止过拟合
    
    # CTC 配置 (Phase 2 新增)
    use_ctc = True          # 启用 CTC Loss 辅助监督
    ctc_weight = 0.5        # CTC Loss 权重: total = trans + ctc_weight * ctc
    
    # 序列配置
    max_length = 128        # 最大帧数 (原256)
    max_text_length = 50    # 最大文本长度
    
    # 优化配置
    use_amp = False         # 禁用混合精度训练，避免NaN问题
    gradient_checkpointing = True  # 梯度检查点
    
    # TensorBoard 配置
    use_tensorboard = True
    log_dir = 'runs'        # TensorBoard日志目录
    log_interval = 10       # 每N步记录一次
    
    # 验证配置
    val_interval = 1        # 每N轮验证一次（改为每轮验证）
    save_interval = 5       # 每N轮保存检查点
    
    # 断点续训
    resume_from = None      # 检查点路径，为None则从头训练
    auto_resume = True      # 自动从最新检查点恢复
    
    # 设备配置
    device = 'cuda'
    num_workers = 0         # Windows下建议设为0


class InferenceConfig:
    batch_size = 1
    max_length = 128
    num_beams = 4           # 增加beam数量
    device = 'cuda'
    
    # 生成参数
    max_new_tokens = 50
    length_penalty = 1.0
    no_repeat_ngram_size = 2  # 避免重复