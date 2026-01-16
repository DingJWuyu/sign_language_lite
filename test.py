"""
测试脚本 - 诊断训练问题
"""
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

from config_lite import TrainConfig, mt5_path, train_label_path
from datasets_lite import SignLanguageDataset, SignLanguageDatasetSimple
from torch.utils.data import DataLoader


def test_data():
    """测试数据加载"""
    print("\n" + "=" * 50)
    print("1. 测试数据加载")
    print("=" * 50)
    
    config = TrainConfig()
    
    try:
        dataset = SignLanguageDataset(train_label_path, config, phase='train')
    except Exception as e:
        print(f"使用标准数据集失败: {e}")
        dataset = SignLanguageDatasetSimple(train_label_path, config, phase='train')
    
    print(f"数据集大小: {len(dataset)}")
    
    if len(dataset) == 0:
        print("错误: 数据集为空!")
        return None
    
    # 获取一个样本
    name, pose, text, gloss = dataset[0]
    print(f"\n样本名称: {name}")
    print(f"文本: {text}")
    print(f"Gloss: {gloss}")
    
    print(f"\n姿态数据形状:")
    for key, value in pose.items():
        print(f"  {key}: {value.shape}, dtype={value.dtype}")
        print(f"    min={value.min().item():.4f}, max={value.max().item():.4f}")
        print(f"    有NaN: {torch.isnan(value).any().item()}")
        print(f"    有Inf: {torch.isinf(value).any().item()}")
    
    # 测试 collate_fn
    print("\n测试批次整理...")
    loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    src_input, tgt_input = next(iter(loader))
    
    print(f"\nbatch 数据:")
    for key, value in src_input.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape}, dtype={value.dtype}")
            print(f"    有NaN: {torch.isnan(value).any().item()}")
    
    print(f"\n目标文本: {tgt_input['gt_sentence']}")
    
    return src_input, tgt_input


def test_model_forward():
    """测试模型前向传播"""
    print("\n" + "=" * 50)
    print("2. 测试模型前向传播")
    print("=" * 50)
    
    from models_lite import SignLanguageLite
    
    config = TrainConfig()
    
    class Args:
        pass
    args = Args()
    args.mt5_path = mt5_path
    args.max_length = config.max_length
    
    print(f"\n加载模型: {mt5_path}")
    model = SignLanguageLite(args)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # 创建测试输入
    B, T = 2, 30  # batch_size, 时间步
    
    test_input = {
        'body': torch.randn(B, T, 9, 3).to(device) * 0.5,  # 9个身体关键点
        'left': torch.randn(B, T, 21, 3).to(device) * 0.5,  # 21个左手关键点
        'right': torch.randn(B, T, 21, 3).to(device) * 0.5,  # 21个右手关键点
        'attention_mask': torch.ones(B, T).to(device),
    }
    
    print("\n测试输入:")
    for key, value in test_input.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape}, device={value.device}")
    
    # 测试 encode_pose
    print("\n测试姿态编码...")
    with torch.no_grad():
        pose_embeds = model.encode_pose(test_input)
        print(f"姿态嵌入: {pose_embeds.shape}")
        print(f"  min={pose_embeds.min().item():.4f}, max={pose_embeds.max().item():.4f}")
        print(f"  有NaN: {torch.isnan(pose_embeds).any().item()}")
        print(f"  有Inf: {torch.isinf(pose_embeds).any().item()}")
    
    # 测试完整前向传播
    print("\n测试完整前向传播...")
    test_target = {
        'gt_sentence': ['你好', '谢谢'],
        'gt_gloss': ['你 好', '谢 谢']
    }
    
    with torch.no_grad():
        try:
            loss = model(test_input, test_target)
            print(f"Loss: {loss.item():.4f}")
            print(f"Loss 是 NaN: {torch.isnan(loss).item()}")
            print(f"Loss 是 Inf: {torch.isinf(loss).item()}")
        except Exception as e:
            print(f"前向传播失败: {e}")
            import traceback
            traceback.print_exc()
    
    return model


def test_with_real_data():
    """使用真实数据测试"""
    print("\n" + "=" * 50)
    print("3. 使用真实数据测试")
    print("=" * 50)
    
    from models_lite import SignLanguageLite
    
    config = TrainConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载数据
    try:
        dataset = SignLanguageDataset(train_label_path, config, phase='train')
    except:
        dataset = SignLanguageDatasetSimple(train_label_path, config, phase='train')
    
    if len(dataset) == 0:
        print("数据集为空，跳过测试")
        return
    
    loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    src_input, tgt_input = next(iter(loader))
    
    # 移动到设备
    for key in ['body', 'left', 'right', 'attention_mask']:
        if key in src_input and torch.is_tensor(src_input[key]):
            src_input[key] = src_input[key].to(device)
    
    print("\n真实数据统计:")
    for key in ['body', 'left', 'right']:
        data = src_input[key]
        print(f"  {key}:")
        print(f"    shape: {data.shape}")
        print(f"    min: {data.min().item():.4f}, max: {data.max().item():.4f}")
        print(f"    mean: {data.mean().item():.4f}, std: {data.std().item():.4f}")
        print(f"    NaN count: {torch.isnan(data).sum().item()}")
        print(f"    Zero count: {(data == 0).sum().item()} / {data.numel()}")
    
    # 加载模型
    class Args:
        pass
    args = Args()
    args.mt5_path = mt5_path
    args.max_length = config.max_length
    
    model = SignLanguageLite(args)
    model = model.to(device)
    model.eval()
    
    # 测试
    print("\n测试前向传播...")
    with torch.no_grad():
        try:
            # 先测试编码
            pose_embeds = model.encode_pose(src_input)
            print(f"姿态嵌入 - shape: {pose_embeds.shape}")
            print(f"           min: {pose_embeds.min().item():.4f}")
            print(f"           max: {pose_embeds.max().item():.4f}")
            print(f"           NaN: {torch.isnan(pose_embeds).any().item()}")
            
            # 测试完整前向
            loss = model(src_input, tgt_input)
            print(f"\nLoss: {loss.item():.4f}")
            print(f"Loss 是 NaN: {torch.isnan(loss).item()}")
            
            if torch.isnan(loss):
                print("\n诊断: Loss 是 NaN!")
                print("可能原因: 禁用混合精度训练试试")
                
        except Exception as e:
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    print("=" * 60)
    print("手语识别模型诊断测试")
    print("=" * 60)
    
    # 测试数据
    test_data()
    
    # 测试模型
    test_model_forward()
    
    # 使用真实数据测试
    test_with_real_data()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)