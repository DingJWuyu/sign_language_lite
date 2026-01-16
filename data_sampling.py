"""
数据采样工具
从原始 CSL-Daily 数据集中采样出轻量化数据集
"""
import os
import random
import gzip
import pickle
import shutil
from collections import defaultdict
from tqdm import tqdm


def load_dataset_file(filename):
    """加载gzip压缩的pickle文件"""
    with gzip.open(filename, "rb") as f:
        return pickle.load(f)


def save_dataset_file(data, filename):
    """保存为gzip压缩的pickle文件"""
    with gzip.open(filename, "wb") as f:
        pickle.dump(data, f)


def sample_dataset(input_path, output_path, sample_ratio=0.1, max_samples=2500, seed=42):
    """
    从原始数据集中采样，保持类别平衡
    
    Args:
        input_path: 原始标签文件路径 (gzip pickle格式)
        output_path: 采样后标签文件保存路径
        sample_ratio: 采样比例 (0-1)
        max_samples: 最大样本数
        seed: 随机种子
    
    Returns:
        sampled_data: 采样后的数据字典
    """
    random.seed(seed)
    
    print(f"正在加载数据: {input_path}")
    
    try:
        data = load_dataset_file(input_path)
    except Exception as e:
        print(f"加载失败: {e}")
        return {}
    
    print(f"原始样本数: {len(data)}")
    
    # 按句子长度分层采样（保证短句和长句都有覆盖）
    length_bins = defaultdict(list)
    
    for key, sample in data.items():
        text = sample.get('text', '')
        text_len = len(text)
        # 分成6个长度区间: 0-10, 10-20, 20-30, 30-40, 40-50, 50+
        bin_idx = min(text_len // 10, 5)
        length_bins[bin_idx].append(key)
    
    print(f"长度分布:")
    for bin_idx in sorted(length_bins.keys()):
        print(f"  区间 {bin_idx*10}-{(bin_idx+1)*10}: {len(length_bins[bin_idx])} 个样本")
    
    # 计算每个区间应采样的数量
    target_samples = min(int(len(data) * sample_ratio), max_samples)
    samples_per_bin = target_samples // len(length_bins)
    
    print(f"目标采样数: {target_samples}")
    print(f"每区间采样数: {samples_per_bin}")
    
    # 从每个区间采样
    sampled_keys = []
    for bin_idx, keys in length_bins.items():
        n_samples = min(len(keys), samples_per_bin)
        if n_samples > 0:
            sampled_keys.extend(random.sample(keys, n_samples))
    
    # 如果采样数不足，从剩余样本中随机补充
    remaining = set(data.keys()) - set(sampled_keys)
    if len(sampled_keys) < target_samples and remaining:
        extra_needed = min(target_samples - len(sampled_keys), len(remaining))
        sampled_keys.extend(random.sample(list(remaining), extra_needed))
    
    # 确保不超过最大样本数
    if len(sampled_keys) > max_samples:
        sampled_keys = random.sample(sampled_keys, max_samples)
    
    # 构建采样后的数据集
    sampled_data = {k: data[k] for k in sampled_keys}
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_dataset_file(sampled_data, output_path)
    
    print(f"采样完成!")
    print(f"  原始样本数: {len(data)}")
    print(f"  采样后样本数: {len(sampled_data)}")
    print(f"  保存到: {output_path}")
    
    return sampled_data


def split_dataset(data, train_ratio=0.8, dev_ratio=0.1, seed=42):
    """
    划分训练/验证/测试集
    
    Args:
        data: 数据字典
        train_ratio: 训练集比例
        dev_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        train_data, dev_data, test_data: 三个数据字典
    """
    random.seed(seed)
    
    keys = list(data.keys())
    random.shuffle(keys)
    
    n_train = int(len(keys) * train_ratio)
    n_dev = int(len(keys) * dev_ratio)
    
    train_keys = keys[:n_train]
    dev_keys = keys[n_train:n_train + n_dev]
    test_keys = keys[n_train + n_dev:]
    
    return (
        {k: data[k] for k in train_keys},
        {k: data[k] for k in dev_keys},
        {k: data[k] for k in test_keys}
    )


def copy_pose_files(sampled_data, src_pose_dir, dst_pose_dir):
    """
    复制采样数据对应的姿态文件
    
    Args:
        sampled_data: 采样后的数据字典
        src_pose_dir: 源姿态文件目录
        dst_pose_dir: 目标姿态文件目录
    """
    os.makedirs(dst_pose_dir, exist_ok=True)
    
    copied = 0
    not_found = 0
    
    for key, sample in tqdm(sampled_data.items(), desc="复制姿态文件"):
        video_path = sample.get('video_path', f"{key}.mp4")
        pose_filename = video_path.replace('.mp4', '.pkl')
        
        src_path = os.path.join(src_pose_dir, pose_filename)
        dst_path = os.path.join(dst_pose_dir, pose_filename)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied += 1
        else:
            not_found += 1
    
    print(f"复制完成: {copied} 个文件")
    if not_found > 0:
        print(f"警告: {not_found} 个文件未找到")


def copy_video_files(sampled_data, src_video_dir, dst_video_dir):
    """
    复制采样数据对应的视频文件
    
    Args:
        sampled_data: 采样后的数据字典
        src_video_dir: 源视频文件目录
        dst_video_dir: 目标视频文件目录
    """
    os.makedirs(dst_video_dir, exist_ok=True)
    
    copied = 0
    not_found = 0
    
    for key, sample in tqdm(sampled_data.items(), desc="复制视频文件"):
        video_path = sample.get('video_path', f"{key}.mp4")
        
        src_path = os.path.join(src_video_dir, video_path)
        dst_path = os.path.join(dst_video_dir, video_path)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied += 1
        else:
            not_found += 1
    
    print(f"复制完成: {copied} 个文件")
    if not_found > 0:
        print(f"警告: {not_found} 个文件未找到")


def create_lite_dataset_from_original():
    """
    从原始 CSL-Daily 数据集创建轻量化版本
    
    使用说明:
    1. 首先下载原始 CSL-Daily 数据集
    2. 将数据放在 ../dataset/CSL_Daily/ 目录下
    3. 运行此函数
    """
    # 路径配置 - 根据实际情况修改
    original_data_dir = "../dataset/CSL_Daily"  # 原始数据目录
    lite_data_dir = "./data/CSL_Daily_lite"      # 轻量化数据目录
    
    # 原始标签文件路径
    original_labels = {
        'train': os.path.join(original_data_dir, "labels.train"),
        'dev': os.path.join(original_data_dir, "labels.dev"),
        'test': os.path.join(original_data_dir, "labels.test"),
    }
    
    # 检查原始数据是否存在
    for phase, path in original_labels.items():
        if not os.path.exists(path):
            print(f"错误: 找不到 {path}")
            print("请先下载 CSL-Daily 数据集并放置在正确位置")
            return
    
    # 采样参数
    sample_config = {
        'train': {'max_samples': 2000, 'sample_ratio': 0.1},
        'dev': {'max_samples': 300, 'sample_ratio': 0.15},
        'test': {'max_samples': 300, 'sample_ratio': 0.15},
    }
    
    # 为每个划分采样
    all_sampled = {}
    for phase, config in sample_config.items():
        output_path = os.path.join(lite_data_dir, f"labels.{phase}")
        sampled = sample_dataset(
            original_labels[phase],
            output_path,
            **config
        )
        all_sampled[phase] = sampled
    
    # 复制对应的姿态文件
    src_pose_dir = os.path.join(original_data_dir, "pose_format")
    dst_pose_dir = os.path.join(lite_data_dir, "pose_format")
    
    if os.path.exists(src_pose_dir):
        print("\n开始复制姿态文件...")
        for phase, data in all_sampled.items():
            copy_pose_files(data, src_pose_dir, dst_pose_dir)
    else:
        print(f"警告: 姿态文件目录不存在 {src_pose_dir}")
    
    # 复制对应的视频文件 (可选)
    src_video_dir = os.path.join(original_data_dir, "sentence-crop")
    dst_video_dir = os.path.join(lite_data_dir, "sentence-crop")
    
    if os.path.exists(src_video_dir):
        print("\n开始复制视频文件...")
        for phase, data in all_sampled.items():
            copy_video_files(data, src_video_dir, dst_video_dir)
    else:
        print(f"警告: 视频文件目录不存在 {src_video_dir}")
    
    print("\n数据集创建完成!")
    print(f"保存位置: {lite_data_dir}")


def create_demo_dataset():
    """
    创建一个演示用的小型数据集（用于测试代码是否正常运行）
    不需要原始数据，直接生成模拟数据
    """
    import numpy as np
    
    lite_data_dir = "./data/CSL_Daily_lite"
    pose_dir = os.path.join(lite_data_dir, "pose_format")
    
    os.makedirs(pose_dir, exist_ok=True)
    
    # 生成15个演示样本 (增加样本数)
    demo_texts = [
        "你好",
        "谢谢你",
        "再见",
        "我很高兴认识你",
        "今天天气很好",
        "请帮帮我",
        "对不起",
        "没关系",
        "早上好",
        "晚安",
        "吃饭了吗",
        "我爱你",
        "辛苦了",
        "不客气",
        "欢迎光临"
    ]
    
    demo_data = {}
    
    for i, text in enumerate(demo_texts):
        sample_name = f"demo_{i:04d}"
        
        # 创建样本数据
        demo_data[sample_name] = {
            'name': sample_name,
            'video_path': f"{sample_name}.mp4",
            'text': text,
            'gloss': list(text)  # 简单地用字符作为gloss
        }
        
        # 创建模拟的姿态数据
        # 每个样本60帧，133个关键点
        num_frames = 60
        num_keypoints = 133
        
        # 生成更合理的姿态数据（归一化到0-1范围）
        # 模拟真实的人体姿态分布
        base_x = 0.5  # 人体中心x
        base_y = 0.4  # 人体中心y
        
        keypoints = np.zeros((num_frames, 1, num_keypoints, 2), dtype=np.float32)
        
        for f in range(num_frames):
            # 添加一些时序变化模拟动作
            t = f / num_frames
            offset = np.sin(t * np.pi * 2) * 0.05
            
            # 身体关键点 (0-22): 集中在身体中心
            keypoints[f, 0, :23, 0] = base_x + np.random.uniform(-0.15, 0.15, 23) + offset
            keypoints[f, 0, :23, 1] = base_y + np.random.uniform(-0.2, 0.3, 23)
            
            # 面部关键点 (23-90): 在头部区域
            keypoints[f, 0, 23:91, 0] = base_x + np.random.uniform(-0.08, 0.08, 68)
            keypoints[f, 0, 23:91, 1] = base_y - 0.15 + np.random.uniform(-0.05, 0.05, 68)
            
            # 左手关键点 (91-111): 在左侧
            keypoints[f, 0, 91:112, 0] = base_x - 0.2 + np.random.uniform(-0.05, 0.05, 21) + offset
            keypoints[f, 0, 91:112, 1] = base_y + 0.1 + np.random.uniform(-0.05, 0.05, 21)
            
            # 右手关键点 (112-132): 在右侧
            keypoints[f, 0, 112:133, 0] = base_x + 0.2 + np.random.uniform(-0.05, 0.05, 21) - offset
            keypoints[f, 0, 112:133, 1] = base_y + 0.1 + np.random.uniform(-0.05, 0.05, 21)
        
        # 确保在0-1范围内
        keypoints = np.clip(keypoints, 0.0, 1.0)
        
        # 置信度分数
        scores = np.random.uniform(0.7, 1.0, (num_frames, 1, num_keypoints)).astype(np.float32)
        
        pose_data = {
            'keypoints': keypoints,
            'scores': scores,
        }
        
        # 保存姿态文件
        pose_path = os.path.join(pose_dir, f"{sample_name}.pkl")
        with open(pose_path, 'wb') as f:
            pickle.dump(pose_data, f)
    
    # 划分数据集
    keys = list(demo_data.keys())
    train_keys = keys[:10]
    dev_keys = keys[10:13]
    test_keys = keys[13:]
    
    # 保存标签文件
    for phase, phase_keys in [('train', train_keys), ('dev', dev_keys), ('test', test_keys)]:
        phase_data = {k: demo_data[k] for k in phase_keys}
        output_path = os.path.join(lite_data_dir, f"labels.{phase}")
        save_dataset_file(phase_data, output_path)
        print(f"保存 {phase}: {len(phase_data)} 个样本 -> {output_path}")
    
    print("\n演示数据集创建完成!")
    print(f"保存位置: {lite_data_dir}")
    print("注意: 这只是模拟数据，仅用于测试代码运行")


def analyze_dataset(label_path):
    """
    分析数据集统计信息
    
    Args:
        label_path: 标签文件路径
    """
    print(f"\n分析数据集: {label_path}")
    
    try:
        data = load_dataset_file(label_path)
    except Exception as e:
        print(f"加载失败: {e}")
        return
    
    print(f"样本总数: {len(data)}")
    
    # 文本长度统计
    text_lengths = []
    for key, sample in data.items():
        text = sample.get('text', '')
        text_lengths.append(len(text))
    
    import statistics
    print(f"文本长度统计:")
    print(f"  最小: {min(text_lengths)}")
    print(f"  最大: {max(text_lengths)}")
    print(f"  平均: {statistics.mean(text_lengths):.1f}")
    print(f"  中位数: {statistics.median(text_lengths)}")
    
    # 显示一些样本
    print(f"\n样本示例:")
    for i, (key, sample) in enumerate(data.items()):
        if i >= 5:
            break
        print(f"  {key}: {sample.get('text', '')[:30]}...")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='数据采样工具')
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['original', 'demo', 'analyze'],
                        help='运行模式: original=从原始数据采样, demo=创建演示数据, analyze=分析数据')
    parser.add_argument('--label_path', type=str, default=None,
                        help='用于分析的标签文件路径')
    
    args = parser.parse_args()
    
    if args.mode == 'original':
        print("从原始数据集创建轻量化版本...")
        create_lite_dataset_from_original()
    elif args.mode == 'demo':
        print("创建演示数据集...")
        create_demo_dataset()
    elif args.mode == 'analyze':
        if args.label_path:
            analyze_dataset(args.label_path)
        else:
            # 分析默认路径
            analyze_dataset('./data/CSL_Daily_lite/labels.train')
