"""
TensorBoard 日志记录工具
用于可视化训练过程
"""
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """TensorBoard 日志记录器"""
    
    def __init__(self, log_dir='runs', experiment_name=None):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称（可选，默认使用时间戳）
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            log_dir,
            experiment_name
        )
        
        os.makedirs(self.log_path, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_path)
        print(f"TensorBoard 日志目录: {self.log_path}")
        print(f"启动 TensorBoard: tensorboard --logdir={os.path.dirname(self.log_path)}")
    
    def log_scalar(self, tag, value, step):
        """记录标量值"""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag, tag_scalar_dict, step):
        """记录多个标量值"""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_loss(self, loss, step, prefix='train'):
        """记录损失值"""
        self.writer.add_scalar(f'{prefix}/loss', loss, step)
    
    def log_learning_rate(self, lr, step):
        """记录学习率"""
        self.writer.add_scalar('train/learning_rate', lr, step)
    
    def log_metrics(self, metrics_dict, step, prefix='eval'):
        """记录评估指标"""
        for name, value in metrics_dict.items():
            self.writer.add_scalar(f'{prefix}/{name}', value, step)
    
    def log_epoch_summary(self, epoch, train_loss, val_loss=None, metrics=None, lr=None):
        """记录epoch摘要"""
        self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
        
        if val_loss is not None:
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalars('epoch/loss_comparison', {
                'train': train_loss,
                'validation': val_loss
            }, epoch)
        
        if lr is not None:
            self.writer.add_scalar('epoch/learning_rate', lr, epoch)
        
        if metrics is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(f'epoch/{name}', value, epoch)
    
    def log_histogram(self, tag, values, step):
        """记录直方图"""
        self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag, text, step):
        """记录文本"""
        self.writer.add_text(tag, text, step)
    
    def log_prediction_samples(self, predictions, ground_truths, step, num_samples=5):
        """记录预测样本对比"""
        text = "| 预测 | 真实 |\n|------|------|\n"
        for pred, gt in zip(predictions[:num_samples], ground_truths[:num_samples]):
            text += f"| {pred} | {gt} |\n"
        self.writer.add_text('predictions/samples', text, step)
    
    def log_model_params(self, model, step):
        """记录模型参数分布"""
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.writer.add_histogram(f'params/{name}', param.data, step)
                self.writer.add_histogram(f'grads/{name}', param.grad, step)
    
    def flush(self):
        """刷新写入"""
        self.writer.flush()
    
    def close(self):
        """关闭写入器"""
        self.writer.close()


def get_experiment_name(config):
    """根据配置生成实验名称"""
    name = f"lr{config.learning_rate}_bs{config.batch_size}"
    if hasattr(config, 'label_smoothing') and config.label_smoothing > 0:
        name += f"_ls{config.label_smoothing}"
    name += "_" + datetime.now().strftime('%m%d_%H%M')
    return name


if __name__ == '__main__':
    # 测试代码
    logger = TensorBoardLogger(experiment_name='test')
    
    for i in range(100):
        logger.log_loss(10 / (i + 1), i)
        logger.log_learning_rate(0.001 * (0.99 ** i), i)
    
    logger.close()
    print("测试完成！运行以下命令查看:")
    print("tensorboard --logdir=runs")
