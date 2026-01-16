"""
推理脚本
用于加载训练好的模型进行手语翻译推理
"""
import torch
import os
import argparse
from tqdm import tqdm

from config_lite import InferenceConfig, mt5_path, test_label_path, pose_dir
from models_lite import SignLanguageLite
from datasets_lite import SignLanguageDataset
from torch.utils.data import DataLoader


def load_model(model_path, config):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型权重文件路径
        config: 配置对象
    
    Returns:
        model: 加载好的模型
    """
    class Args:
        pass
    
    args = Args()
    args.mt5_path = mt5_path
    args.max_length = config.max_length
    
    model = SignLanguageLite(args)
    
    if os.path.exists(model_path):
        print(f"加载模型权重: {model_path}")
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        print(f"警告: 模型文件不存在 {model_path}")
        print("将使用未训练的模型")
    
    model = model.to(config.device)
    model.eval()
    
    return model


def inference_single(model, pose_data, config):
    """
    对单个样本进行推理
    
    Args:
        model: 模型
        pose_data: 姿态数据字典
        config: 配置对象
    
    Returns:
        text: 翻译结果
    """
    # 准备输入
    src_input = {}
    
    for key in ['body', 'left', 'right']:
        tensor = pose_data[key]
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)  # 添加batch维度
        src_input[key] = tensor.to(config.device)
    
    # 创建attention mask
    T = src_input['body'].shape[1]
    src_input['attention_mask'] = torch.ones(1, T).to(config.device)
    
    # 推理
    with torch.no_grad():
        result = model.generate(src_input, max_new_tokens=50)
    
    return result[0] if result else ""


def inference_batch(model, dataloader, config, output_file=None):
    """
    批量推理
    
    Args:
        model: 模型
        dataloader: 数据加载器
        config: 配置对象
        output_file: 输出文件路径 (可选)
    
    Returns:
        results: 推理结果列表
    """
    results = []
    
    model.eval()
    
    with torch.no_grad():
        for src_input, tgt_input in tqdm(dataloader, desc="推理中"):
            # 移动到设备
            for key in ['body', 'left', 'right', 'attention_mask']:
                if key in src_input:
                    src_input[key] = src_input[key].to(config.device)
            
            # 生成翻译
            predictions = model.generate(src_input)
            
            # 收集结果
            names = src_input.get('names', ['unknown'] * len(predictions))
            gt_sentences = tgt_input.get('gt_sentence', [''] * len(predictions))
            
            for name, pred, gt in zip(names, predictions, gt_sentences):
                results.append({
                    'name': name,
                    'prediction': pred,
                    'ground_truth': gt
                })
    
    # 保存结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(f"名称: {r['name']}\n")
                f.write(f"预测: {r['prediction']}\n")
                f.write(f"真实: {r['ground_truth']}\n")
                f.write("-" * 50 + "\n")
        print(f"结果已保存到: {output_file}")
    
    return results


def evaluate(results):
    """
    评估翻译结果
    
    Args:
        results: 推理结果列表
    
    Returns:
        metrics: 评估指标
    """
    # 简单的准确率评估
    exact_match = 0
    total = len(results)
    
    for r in results:
        pred = r['prediction'].strip()
        gt = r['ground_truth'].strip()
        
        if pred == gt:
            exact_match += 1
    
    accuracy = exact_match / total if total > 0 else 0
    
    print(f"\n评估结果:")
    print(f"  样本数: {total}")
    print(f"  完全匹配: {exact_match}")
    print(f"  准确率: {accuracy:.2%}")
    
    # 显示一些预测结果
    print(f"\n预测示例:")
    for i, r in enumerate(results[:5]):
        print(f"  [{i+1}] 预测: {r['prediction']}")
        print(f"      真实: {r['ground_truth']}")
    
    return {
        'accuracy': accuracy,
        'exact_match': exact_match,
        'total': total
    }


def main():
    parser = argparse.ArgumentParser(description='手语翻译推理')
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='模型权重文件路径')
    parser.add_argument('--label_path', type=str, default=None,
                        help='标签文件路径 (默认使用测试集)')
    parser.add_argument('--output', type=str, default='inference_results.txt',
                        help='输出结果文件路径')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批量大小')
    
    args = parser.parse_args()
    
    # 配置
    config = InferenceConfig()
    config.batch_size = args.batch_size
    
    # 加载模型
    model = load_model(args.model_path, config)
    
    # 加载数据
    label_path = args.label_path if args.label_path else test_label_path
    
    print(f"加载测试数据: {label_path}")
    
    try:
        dataset = SignLanguageDataset(label_path, config, phase='test')
        
        if len(dataset) == 0:
            print("错误: 数据集为空")
            return
        
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )
        
        # 推理
        results = inference_batch(model, dataloader, config, args.output)
        
        # 评估
        if results:
            evaluate(results)
            
    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
