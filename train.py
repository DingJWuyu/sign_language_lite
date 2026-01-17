"""
训练脚本 - 支持断点续训和TensorBoard可视化
用于训练轻量化手语翻译模型 (Phase 2: 支持 CTC Loss)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import argparse
import glob
import math
import json

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_lite import TrainConfig, mt5_path, train_label_path, dev_label_path
from models_lite import SignLanguageLite
from datasets_lite import SignLanguageDataset, SignLanguageDatasetSimple
from tensorboard_logger import TensorBoardLogger, get_experiment_name

# 检查 CUDA 可用性
if torch.cuda.is_available():
    from torch.amp import GradScaler, autocast
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("警告: CUDA 不可用，将使用 CPU 训练（会很慢）")
    from contextlib import nullcontext as autocast
    class GradScaler:
        def __init__(self, device='cuda'): pass
        def scale(self, loss): return loss
        def step(self, optimizer): optimizer.step()
        def update(self): pass


class WarmupCosineScheduler:
    """带预热的余弦退火学习率调度器"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch
        
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if epoch < self.warmup_epochs:
                # 线性预热
                lr = base_lr * (epoch + 1) / self.warmup_epochs
            else:
                # 余弦退火
                progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                lr = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            
            param_group['lr'] = lr
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        return {'last_epoch': self.last_epoch}
    
    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']


# ============ Gloss 词表构建 (Phase 2 新增) ============
class GlossVocab:
    """Gloss 词表管理器"""
    def __init__(self):
        self.word2idx = {'<blank>': 0, '<unk>': 1}  # 0: CTC blank, 1: unknown
        self.idx2word = {0: '<blank>', 1: '<unk>'}
        self.next_idx = 2
        
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.next_idx
            self.idx2word[self.next_idx] = word
            self.next_idx += 1
        return self.word2idx[word]
    
    def encode(self, gloss_str):
        """将 gloss 字符串编码为索引列表"""
        if not gloss_str:
            return []
        words = gloss_str.split()
        return [self.word2idx.get(w, 1) for w in words]  # 未知词用 <unk>
    
    def __len__(self):
        return self.next_idx
    
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'word2idx': self.word2idx}, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path):
        vocab = cls()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            vocab.word2idx = data['word2idx']
            vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
            vocab.next_idx = max(vocab.idx2word.keys()) + 1
        return vocab


def build_gloss_vocab(dataset):
    """从数据集构建 Gloss 词表"""
    vocab = GlossVocab()
    for i in range(len(dataset)):
        try:
            _, _, _, gloss = dataset[i]
            if gloss:
                for word in gloss.split():
                    vocab.add_word(word)
        except:
            continue
    print(f"Gloss 词表大小: {len(vocab)}")
    return vocab


def find_latest_checkpoint(checkpoint_dir):
    """查找最新的检查点文件"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # 查找所有检查点文件
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))
    
    if not checkpoints:
        # 尝试查找 latest_checkpoint.pth
        latest = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        if os.path.exists(latest):
            return latest
        return None
    
    # 按epoch排序
    def get_epoch(path):
        filename = os.path.basename(path)
        try:
            return int(filename.replace('checkpoint_epoch_', '').replace('.pth', '').replace('_interrupted', ''))
        except:
            return 0
    
    checkpoints.sort(key=get_epoch)
    return checkpoints[-1]


def load_checkpoint(checkpoint_path, model, optimizer, scheduler=None):
    """加载检查点"""
    print(f"\n加载检查点: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        # strict=False 允许加载不完全匹配的模型（处理新增层或结构变更）
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing_keys:
            print(f"  [警告] 缺失键 (将使用随机初始化): {missing_keys[:5]} ...")
        if unexpected_keys:
            print(f"  [警告] 未预期键 (将被忽略): {unexpected_keys[:5]} ...")
    else:
        # 旧格式的检查点，直接是模型权重
        model.load_state_dict(checkpoint, strict=False)
        return 0, float('inf')
    
    # 加载优化器状态
    # 如果模型参数发生了巨大变化（如解冻层），旧的优化器状态可能导致错误
    if 'optimizer_state_dict' in checkpoint and optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            print(f"  [警告] 无法加载优化器状态 (可能是参数量变更): {e}")
            print("  已重置优化器状态，从头开始优化。")
    
    # 加载调度器状态
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            pass
    
    start_epoch = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))
    
    print(f"  从 epoch {start_epoch} 继续训练")
    print(f"  之前最佳验证损失: {best_loss:.4f}")
    
    return start_epoch, best_loss


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, best_loss, is_best=False):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_loss': best_loss,
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    # 同时保存为latest
    save_dir = os.path.dirname(checkpoint_path)
    latest_path = os.path.join(save_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(model.state_dict(), best_path)
        print(f"  最佳模型已保存: {best_path}")


def evaluate(model, data_loader, config):
    """评估模型，返回损失和生成的样本"""
    model.eval()
    total_loss = 0
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for src_input, tgt_input in data_loader:
            # 移动到设备
            for key in ['body', 'left', 'right', 'face', 'attention_mask']:
                if key in src_input and torch.is_tensor(src_input[key]):
                    src_input[key] = src_input[key].to(config.device)
            
            # 计算损失
            loss = model(src_input, tgt_input)
            total_loss += loss.item()
            
            # 生成预测（仅取前几个batch）
            if len(predictions) < 50:
                preds = model.generate(src_input, max_new_tokens=50)
                predictions.extend(preds)
                ground_truths.extend(tgt_input['gt_sentence'])
    
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    
    return avg_loss, predictions, ground_truths


def compute_metrics(predictions, ground_truths):
    """计算评估指标"""
    if not predictions or not ground_truths:
        return {}
    
    exact_match = 0
    char_correct = 0
    char_total = 0
    
    for pred, gt in zip(predictions, ground_truths):
        pred = pred.strip()
        gt = gt.strip()
        
        # 完全匹配
        if pred == gt:
            exact_match += 1
        
        # 字符级准确率
        for p_char, g_char in zip(pred, gt):
            if p_char == g_char:
                char_correct += 1
            char_total += 1
        
        # 补齐长度差异
        char_total += abs(len(pred) - len(gt))
    
    metrics = {
        'exact_match': exact_match / len(predictions) if predictions else 0,
        'char_accuracy': char_correct / char_total if char_total > 0 else 0,
    }
    
    return metrics


def train(args):
    """主训练函数"""
    config = TrainConfig()
    
    # 命令行参数覆盖配置
    if args.resume:
        config.resume_from = args.resume
    if args.lr:
        config.learning_rate = args.lr
    if args.epochs:
        config.epochs = args.epochs
    if args.no_tensorboard:
        config.use_tensorboard = False
    if args.no_resume:
        config.auto_resume = False
    
    # 如果没有GPU，使用CPU
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.use_amp = False
    
    print(f"\n{'='*60}")
    print("训练配置:")
    print(f"  设备: {config.device}")
    print(f"  批量大小: {config.batch_size}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  预热轮数: {config.warmup_epochs}")
    print(f"  训练轮数: {config.epochs}")
    print(f"  标签平滑: {config.label_smoothing}")
    print(f"  TensorBoard: {config.use_tensorboard}")
    print(f"  自动恢复: {config.auto_resume}")
    print(f"{'='*60}\n")
    
    # 检查数据文件
    if not os.path.exists(train_label_path):
        print(f"错误: 找不到训练数据文件 {train_label_path}")
        print("\n请先运行: python data_sampling.py --mode demo")
        return
    
    if not os.path.exists(mt5_path):
        print(f"错误: 找不到mT5模型 {mt5_path}")
        return
    
    # 创建数据集
    print("加载训练数据...")
    try:
        train_dataset = SignLanguageDataset(train_label_path, config, phase='train')
    except Exception as e:
        print(f"加载训练数据失败: {e}")
        train_dataset = SignLanguageDatasetSimple(train_label_path, config, phase='train')
    
    if len(train_dataset) == 0:
        print("错误: 训练数据集为空")
        return
    
    # 构建 Gloss 词表 (Phase 2 新增)
    print("构建 Gloss 词表...")
    gloss_vocab = build_gloss_vocab(train_dataset)
    gloss_vocab_size = len(gloss_vocab)
    
    # 保存词表
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    gloss_vocab.save(os.path.join(save_dir, 'gloss_vocab.json'))
    
    print("加载验证数据...")
    try:
        dev_dataset = SignLanguageDataset(dev_label_path, config, phase='dev')
    except:
        dev_dataset = SignLanguageDatasetSimple(dev_label_path, config, phase='dev')
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dev_dataset.collate_fn
    ) if len(dev_dataset) > 0 else None
    
    # 创建模型
    print("\n加载模型...")
    
    class Args:
        pass
    
    model_args = Args()
    model_args.mt5_path = mt5_path
    model_args.max_length = config.max_length
    model_args.label_smoothing = config.label_smoothing
    # Phase 2: CTC 相关配置
    model_args.gloss_vocab_size = gloss_vocab_size
    model_args.use_ctc = getattr(config, 'use_ctc', True)
    model_args.ctc_weight = getattr(config, 'ctc_weight', 0.5)
    
    try:
        model = SignLanguageLite(model_args)
        model = model.to(config.device)
        print(f"模型加载成功!")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器（带预热）
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.epochs,
        min_lr=config.min_lr
    )
    
    # 创建保存目录
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    # 断点续训
    start_epoch = 0
    best_loss = float('inf')
    
    if config.resume_from:
        # 从指定的检查点恢复
        if os.path.exists(config.resume_from):
            start_epoch, best_loss = load_checkpoint(
                config.resume_from, model, optimizer, scheduler
            )
    elif config.auto_resume:
        # 自动查找最新的检查点
        latest_checkpoint = find_latest_checkpoint(save_dir)
        if latest_checkpoint:
            start_epoch, best_loss = load_checkpoint(
                latest_checkpoint, model, optimizer, scheduler
            )
    
    # TensorBoard
    tb_logger = None
    if config.use_tensorboard:
        experiment_name = get_experiment_name(config)
        tb_logger = TensorBoardLogger(config.log_dir, experiment_name)
    
    # 混合精度配置
    amp_dtype = torch.float16
    if config.use_amp and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            print("AMP: 启用 BFloat16 加速 (比 FP16 更稳定)")
            amp_dtype = torch.bfloat16
        else:
            print("AMP: 启用 FP16 加速")

    scaler = GradScaler('cuda') if config.use_amp and amp_dtype == torch.float16 else None
    
    # 训练循环
    print(f"\n开始训练...")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(dev_dataset) if dev_dataset else 0}")
    print(f"每轮迭代次数: {len(train_loader)}")
    if start_epoch > 0:
        print(f"从 epoch {start_epoch + 1} 继续训练")
    print()
    
    global_step = start_epoch * len(train_loader)
    
    try:
        for epoch in range(start_epoch, config.epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')
            
            for step, (src_input, tgt_input) in enumerate(pbar):
                try:
                    # 移动到设备 (Phase 2: 增加 face)
                    for key in ['body', 'left', 'right', 'face', 'attention_mask']:
                        if key in src_input and torch.is_tensor(src_input[key]):
                            src_input[key] = src_input[key].to(config.device)
                    
                    # 准备 Gloss 标签 (Phase 2 新增)
                    gloss_labels = None
                    gloss_lengths = None
                    if model_args.use_ctc and 'gt_gloss' in tgt_input:
                        gloss_list = tgt_input['gt_gloss']
                        encoded_glosses = [gloss_vocab.encode(g) for g in gloss_list]
                        
                        # 过滤空 gloss
                        valid_glosses = [g for g in encoded_glosses if len(g) > 0]
                        
                        if len(valid_glosses) == len(encoded_glosses):
                            # 填充到相同长度
                            max_gloss_len = max(len(g) for g in encoded_glosses)
                            padded_glosses = []
                            lengths = []
                            for g in encoded_glosses:
                                lengths.append(len(g))
                                padded = g + [0] * (max_gloss_len - len(g))  # 0 是 blank
                                padded_glosses.append(padded)
                            
                            gloss_labels = torch.tensor(padded_glosses, dtype=torch.long, device=config.device)
                            gloss_lengths = torch.tensor(lengths, dtype=torch.long, device=config.device)
                    
                    # 前向传播 (Phase 2: 传入 gloss)
                    with autocast(device_type=config.device, enabled=config.use_amp, dtype=amp_dtype):
                        loss = model(src_input, tgt_input, gloss_labels, gloss_lengths)
                    
                    # 检查NaN
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"\n警告: 检测到NaN/Inf loss，跳过此batch")
                        optimizer.zero_grad()
                        continue
                    
                    loss = loss / config.gradient_accumulation
                    
                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    if (step + 1) % config.gradient_accumulation == 0:
                        if scaler:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                        optimizer.zero_grad()
                    
                    batch_loss = loss.item() * config.gradient_accumulation
                    total_loss += batch_loss
                    num_batches += 1
                    
                    pbar.set_postfix({
                        'loss': f'{batch_loss:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })
                    
                    # TensorBoard 记录（每N步）
                    if tb_logger and global_step % config.log_interval == 0:
                        tb_logger.log_loss(batch_loss, global_step, prefix='train')
                        tb_logger.log_learning_rate(scheduler.get_last_lr()[0], global_step)
                    
                    global_step += 1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n显存不足! 尝试减小 batch_size 或 max_length")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    raise
            
            # 更新学习率
            scheduler.step(epoch)
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}, LR: {current_lr:.2e}')
            
            # 验证
            val_loss = None
            val_metrics = None
            
            if dev_loader and (epoch + 1) % config.val_interval == 0:
                val_loss, predictions, ground_truths = evaluate(model, dev_loader, config)
                val_metrics = compute_metrics(predictions, ground_truths)
                
                print(f'Validation Loss: {val_loss:.4f}')
                if val_metrics:
                    print(f'  完全匹配率: {val_metrics["exact_match"]:.2%}')
                    print(f'  字符准确率: {val_metrics["char_accuracy"]:.2%}')
                
                # TensorBoard 记录预测样本
                if tb_logger and predictions:
                    tb_logger.log_prediction_samples(predictions, ground_truths, epoch)
                
                # 保存最佳模型
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_checkpoint(
                        os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                        model, optimizer, scheduler, epoch + 1, best_loss, is_best=True
                    )
            
            # TensorBoard epoch 摘要
            if tb_logger:
                tb_logger.log_epoch_summary(
                    epoch + 1, avg_loss, val_loss, val_metrics, current_lr
                )
                tb_logger.flush()
            
            # 定期保存检查点
            if (epoch + 1) % config.save_interval == 0:
                save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                save_checkpoint(save_path, model, optimizer, scheduler, epoch + 1, best_loss)
                print(f'检查点已保存: {save_path}')
    
    except KeyboardInterrupt:
        print("\n\n训练被中断!")
        # 保存中断时的检查点
        interrupt_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}_interrupted.pth')
        save_checkpoint(interrupt_path, model, optimizer, scheduler, epoch + 1, best_loss)
        print(f'中断检查点已保存: {interrupt_path}')
        print(f'下次训练将自动从此处继续')
    
    finally:
        # 保存最终模型
        final_save_path = os.path.join(save_dir, 'final_model.pth')
        torch.save(model.state_dict(), final_save_path)
        print(f'\n最终模型保存到: {final_save_path}')
        
        if tb_logger:
            tb_logger.close()
            print(f'TensorBoard 日志已保存')


def main():
    parser = argparse.ArgumentParser(description='手语翻译模型训练')
    parser.add_argument('--resume', type=str, default=None,
                        help='从指定检查点恢复训练')
    parser.add_argument('--lr', type=float, default=None,
                        help='覆盖学习率')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖训练轮数')
    parser.add_argument('--no-tensorboard', action='store_true',
                        help='禁用TensorBoard')
    parser.add_argument('--no-resume', action='store_true',
                        help='禁用自动恢复，从头训练')
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
