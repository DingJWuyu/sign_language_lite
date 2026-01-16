"""
训练脚本
用于训练轻量化手语翻译模型
"""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_lite import TrainConfig, mt5_path, train_label_path, dev_label_path
from models_lite import SignLanguageLite
from datasets_lite import SignLanguageDataset, SignLanguageDatasetSimple

# 检查 CUDA 可用性
if torch.cuda.is_available():
    from torch.cuda.amp import GradScaler, autocast
    print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("警告: CUDA 不可用，将使用 CPU 训练（会很慢）")
    # 创建dummy的autocast和scaler
    from contextlib import nullcontext as autocast
    class GradScaler:
        def scale(self, loss): return loss
        def step(self, optimizer): optimizer.step()
        def update(self): pass


def train():
    """主训练函数"""
    config = TrainConfig()
    
    # 如果没有GPU，使用CPU
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.use_amp = False
    
    print(f"\n{'='*50}")
    print("训练配置:")
    print(f"  设备: {config.device}")
    print(f"  批量大小: {config.batch_size}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  训练轮数: {config.epochs}")
    print(f"  最大序列长度: {config.max_length}")
    print(f"  混合精度: {config.use_amp}")
    print(f"{'='*50}\n")
    
    # 检查数据文件是否存在
    if not os.path.exists(train_label_path):
        print(f"错误: 找不到训练数据文件 {train_label_path}")
        print("\n请先运行以下命令创建演示数据:")
        print("  python data_sampling.py --mode demo")
        return
    
    # 检查mt5模型是否存在
    if not os.path.exists(mt5_path):
        print(f"错误: 找不到mT5模型 {mt5_path}")
        print("\n请下载 mT5-small 模型:")
        print("  方法1: 使用 huggingface-cli download google/mt5-small")
        print("  方法2: 从 https://huggingface.co/google/mt5-small 手动下载")
        return
    
    # 创建数据集
    print("加载训练数据...")
    try:
        train_dataset = SignLanguageDataset(train_label_path, config, phase='train')
    except Exception as e:
        print(f"加载训练数据失败: {e}")
        print("尝试使用简化版数据集...")
        train_dataset = SignLanguageDatasetSimple(train_label_path, config, phase='train')
    
    if len(train_dataset) == 0:
        print("错误: 训练数据集为空")
        return
    
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
        num_workers=0,  # Windows 下建议设为0
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
    
    args = Args()
    args.mt5_path = mt5_path
    args.max_length = config.max_length
    
    try:
        model = SignLanguageLite(args)
        model = model.to(config.device)
        print(f"模型加载成功!")
        
        # 统计参数量
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
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    
    # 混合精度训练
    scaler = GradScaler() if config.use_amp and config.device == 'cuda' else None
    
    # 创建保存目录
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练循环
    best_loss = float('inf')
    
    print(f"\n开始训练...")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(dev_dataset) if dev_dataset else 0}")
    print(f"每轮迭代次数: {len(train_loader)}")
    print()
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')
        
        for step, (src_input, tgt_input) in enumerate(pbar):
            try:
                # 移动到设备
                for key in ['body', 'left', 'right', 'attention_mask']:
                    if key in src_input and torch.is_tensor(src_input[key]):
                        src_input[key] = src_input[key].to(config.device)
                
                # 前向传播
                if config.use_amp and scaler is not None:
                    with autocast():
                        loss = model(src_input, tgt_input)
                        loss = loss / config.gradient_accumulation
                    
                    scaler.scale(loss).backward()
                    
                    if (step + 1) % config.gradient_accumulation == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss = model(src_input, tgt_input)
                    loss = loss / config.gradient_accumulation
                    loss.backward()
                    
                    if (step + 1) % config.gradient_accumulation == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                
                total_loss += loss.item() * config.gradient_accumulation
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n显存不足! 尝试减小 batch_size 或 max_length")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                raise
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1} - Average Loss: {avg_loss:.4f}')
        
        # 验证
        if dev_loader and (epoch + 1) % 5 == 0:
            val_loss = evaluate(model, dev_loader, config)
            print(f'Validation Loss: {val_loss:.4f}')
            
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(model.state_dict(), save_path)
                print(f'模型已保存到: {save_path}')
        
        # 每10轮保存一次检查点
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f'检查点已保存: {save_path}')
    
    # 保存最终模型
    final_save_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_save_path)
    print(f'\n训练完成! 最终模型保存到: {final_save_path}')


def evaluate(model, data_loader, config):
    """评估模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src_input, tgt_input in tqdm(data_loader, desc='验证中'):
            # 移动到设备
            for key in ['body', 'left', 'right', 'attention_mask']:
                if key in src_input and torch.is_tensor(src_input[key]):
                    src_input[key] = src_input[key].to(config.device)
            
            if config.use_amp and config.device == 'cuda':
                with autocast():
                    loss = model(src_input, tgt_input)
            else:
                loss = model(src_input, tgt_input)
            
            total_loss += loss.item()
    
    return total_loss / len(data_loader) if len(data_loader) > 0 else 0


if __name__ == '__main__':
    train()