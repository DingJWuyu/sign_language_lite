"""
推理脚本 - 带有全面的评估指标
用于加载训练好的模型进行手语翻译推理
"""
import torch
import os
import argparse
import sys
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    args.label_smoothing = 0  # 推理时不需要标签平滑
    
    # 自动推断 Gloss 词表大小 (Fix for shape mismatch)
    gloss_vocab_path = os.path.join(os.path.dirname(model_path), 'gloss_vocab.json')
    if not os.path.exists(gloss_vocab_path):
        # 尝试默认路径
        gloss_vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'gloss_vocab.json')
        
    if os.path.exists(gloss_vocab_path):
        try:
            import json
            with open(gloss_vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
                # 如果是 word2idx 结构
                if 'word2idx' in vocab_data:
                    args.gloss_vocab_size = len(vocab_data['word2idx'])
                else:
                    args.gloss_vocab_size = len(vocab_data)
                print(f"已加载 Gloss 词表，大小: {args.gloss_vocab_size}")
        except Exception as e:
            print(f"无法加载 Gloss 词表: {e}，使用默认大小 2000")
            args.gloss_vocab_size = 2000
    else:
        print("未找到 Gloss 词表文件，使用默认大小 2000")
        args.gloss_vocab_size = 2000

    model = SignLanguageLite(args)
    
    if os.path.exists(model_path):
        print(f"加载模型权重: {model_path}")
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print(f"警告: 模型文件不存在 {model_path}")
        print("将使用未训练的模型")
    
    model = model.to(config.device)
    model.eval()
    
    return model


def compute_bleu(reference, hypothesis, max_n=4):
    """
    计算BLEU分数（简化版，基于字符级别）
    
    Args:
        reference: 参考文本
        hypothesis: 生成文本
        max_n: 最大n-gram
    
    Returns:
        bleu_score: BLEU分数
    """
    import math
    
    ref_chars = list(reference.strip())
    hyp_chars = list(hypothesis.strip())
    
    if len(hyp_chars) == 0:
        return 0.0
    
    # 计算各阶n-gram精确率
    precisions = []
    for n in range(1, min(max_n + 1, len(hyp_chars) + 1)):
        # 参考文本的n-gram计数
        ref_ngrams = Counter()
        for i in range(len(ref_chars) - n + 1):
            ngram = tuple(ref_chars[i:i+n])
            ref_ngrams[ngram] += 1
        
        # 假设文本的n-gram计数
        hyp_ngrams = Counter()
        for i in range(len(hyp_chars) - n + 1):
            ngram = tuple(hyp_chars[i:i+n])
            hyp_ngrams[ngram] += 1
        
        # 计算clipped计数
        clipped_count = 0
        total_count = 0
        for ngram, count in hyp_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))
            total_count += count
        
        if total_count > 0:
            precisions.append(clipped_count / total_count)
        else:
            precisions.append(0)
    
    if not precisions or all(p == 0 for p in precisions):
        return 0.0
    
    # 几何平均
    log_precision = sum(math.log(p) if p > 0 else -float('inf') for p in precisions) / len(precisions)
    
    # 简短惩罚
    bp = 1.0
    if len(hyp_chars) < len(ref_chars):
        bp = math.exp(1 - len(ref_chars) / len(hyp_chars))
    
    bleu = bp * math.exp(log_precision) if log_precision > -float('inf') else 0.0
    
    return bleu


def compute_edit_distance(s1, s2):
    """计算编辑距离"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    return dp[m][n]


def compute_wer(reference, hypothesis):
    """
    计算词错误率 (WER) - 对中文使用字符级别
    
    Args:
        reference: 参考文本
        hypothesis: 生成文本
    
    Returns:
        wer: 词错误率 (0-1, 越低越好)
    """
    ref_chars = list(reference.strip())
    hyp_chars = list(hypothesis.strip())
    
    if len(ref_chars) == 0:
        return 1.0 if len(hyp_chars) > 0 else 0.0
    
    edit_dist = compute_edit_distance(ref_chars, hyp_chars)
    wer = edit_dist / len(ref_chars)
    
    return min(wer, 1.0)  # 限制在0-1范围内


def compute_accuracy_metrics(predictions, ground_truths):
    """
    计算多种准确率指标
    
    Args:
        predictions: 预测文本列表
        ground_truths: 真实文本列表
    
    Returns:
        metrics: 包含多种指标的字典
    """
    exact_match = 0
    char_correct = 0
    char_total = 0
    total_bleu = 0
    total_wer = 0
    
    # 部分匹配统计
    partial_50 = 0  # 50%以上字符匹配
    partial_80 = 0  # 80%以上字符匹配
    
    for pred, gt in zip(predictions, ground_truths):
        pred = pred.strip()
        gt = gt.strip()
        
        # 1. 完全匹配
        if pred == gt:
            exact_match += 1
        
        # 2. 字符级准确率
        match_count = 0
        max_len = max(len(pred), len(gt))
        min_len = min(len(pred), len(gt))
        
        for i in range(min_len):
            if pred[i] == gt[i]:
                match_count += 1
                char_correct += 1
            char_total += 1
        
        # 补齐长度差异
        char_total += abs(len(pred) - len(gt))
        
        # 计算该样本的字符匹配率
        if len(gt) > 0:
            char_match_rate = match_count / len(gt)
            if char_match_rate >= 0.5:
                partial_50 += 1
            if char_match_rate >= 0.8:
                partial_80 += 1
        
        # 3. BLEU分数
        bleu = compute_bleu(gt, pred)
        total_bleu += bleu
        
        # 4. WER
        wer = compute_wer(gt, pred)
        total_wer += wer
    
    n = len(predictions)
    
    metrics = {
        'exact_match': exact_match / n if n > 0 else 0,
        'char_accuracy': char_correct / char_total if char_total > 0 else 0,
        'partial_50': partial_50 / n if n > 0 else 0,  # 50%以上匹配
        'partial_80': partial_80 / n if n > 0 else 0,  # 80%以上匹配
        'bleu': total_bleu / n if n > 0 else 0,
        'wer': total_wer / n if n > 0 else 0,
        'total_samples': n,
        'exact_match_count': exact_match,
    }
    
    return metrics


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
            for key in ['body', 'left', 'right', 'face', 'attention_mask']:
                if key in src_input:
                    src_input[key] = src_input[key].to(config.device)
            
            # 生成翻译
            predictions = model.generate(
                src_input, 
                max_new_tokens=config.max_new_tokens
            )
            
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


def print_evaluation_report(metrics, results, num_samples=10):
    """打印详细的评估报告"""
    print("\n" + "=" * 60)
    print("评估报告")
    print("=" * 60)
    
    print(f"\n📊 总体统计:")
    print(f"  总样本数: {metrics['total_samples']}")
    print(f"  完全匹配: {metrics['exact_match_count']}")
    
    print(f"\n📈 准确率指标:")
    print(f"  完全匹配率:   {metrics['exact_match']:.2%}")
    print(f"  字符准确率:   {metrics['char_accuracy']:.2%}")
    print(f"  50%部分匹配: {metrics['partial_50']:.2%}")
    print(f"  80%部分匹配: {metrics['partial_80']:.2%}")
    
    print(f"\n📐 其他指标:")
    print(f"  BLEU分数:     {metrics['bleu']:.4f}")
    print(f"  字符错误率:   {metrics['wer']:.2%} (越低越好)")
    
    print(f"\n📝 预测示例 (前{num_samples}个):")
    for i, r in enumerate(results[:num_samples]):
        pred = r['prediction']
        gt = r['ground_truth']
        
        # 计算匹配情况
        match_chars = sum(1 for p, g in zip(pred, gt) if p == g)
        match_rate = match_chars / len(gt) if len(gt) > 0 else 0
        
        status = "✓" if pred == gt else f"({match_rate:.0%})"
        
        print(f"\n  [{i+1}] {status}")
        print(f"      预测: {pred}")
        print(f"      真实: {gt}")
    
    print("\n" + "=" * 60)
    
    # 分析常见错误
    print("\n🔍 错误分析:")
    errors = [r for r in results if r['prediction'] != r['ground_truth']]
    
    if errors:
        # 统计预测长度偏差
        length_diffs = [len(r['prediction']) - len(r['ground_truth']) for r in errors]
        avg_diff = sum(length_diffs) / len(length_diffs) if length_diffs else 0
        
        print(f"  错误样本数: {len(errors)}")
        print(f"  预测长度平均偏差: {avg_diff:+.1f} 字符")
        
        # 统计是否有空预测
        empty_preds = sum(1 for r in results if len(r['prediction'].strip()) == 0)
        if empty_preds > 0:
            print(f"  空预测数量: {empty_preds}")
    else:
        print("  无错误！所有预测完全匹配！")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='手语翻译推理与评估')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth',
                        help='模型权重文件路径')
    parser.add_argument('--label_path', type=str, default=None,
                        help='标签文件路径 (默认使用测试集)')
    parser.add_argument('--output', type=str, default='inference_results.txt',
                        help='输出结果文件路径')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批量大小')
    parser.add_argument('--num_beams', type=int, default=4,
                        help='Beam search 数量')
    
    args = parser.parse_args()
    
    # 配置
    config = InferenceConfig()
    config.batch_size = args.batch_size
    config.num_beams = args.num_beams
    
    # 处理模型路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(base_dir, model_path)
    
    # 加载模型
    model = load_model(model_path, config)
    
    # 加载数据
    label_path = args.label_path if args.label_path else test_label_path
    
    print(f"加载测试数据: {label_path}")
    
    try:
        dataset = SignLanguageDataset(label_path, config, phase='test')
        
        if len(dataset) == 0:
            print("错误: 数据集为空")
            return
        
        print(f"测试样本数: {len(dataset)}")
        
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )
        
        # 推理
        output_path = os.path.join(base_dir, args.output)
        results = inference_batch(model, dataloader, config, output_path)
        
        # 评估
        if results:
            predictions = [r['prediction'] for r in results]
            ground_truths = [r['ground_truth'] for r in results]
            
            metrics = compute_accuracy_metrics(predictions, ground_truths)
            print_evaluation_report(metrics, results)
            
    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
