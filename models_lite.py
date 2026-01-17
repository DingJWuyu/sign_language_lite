import torch
import torch.nn as nn
import os

# 尝试导入 transformers，如果失败则提供友好提示
try:
    from transformers import MT5ForConditionalGeneration, T5Tokenizer
except ImportError:
    print("错误: 请安装 transformers 库")
    print("运行: pip install transformers")
    raise

# 尝试导入 einops
try:
    from einops import rearrange
except ImportError:
    print("错误: 请安装 einops 库")
    print("运行: pip install einops")
    raise


class LightweightGCN(nn.Module):
    """轻量级图卷积网络"""
    
    def __init__(self, in_channels, out_channels, num_joints):
        super().__init__()
        self.num_joints = num_joints
        
        # 简化的空间卷积
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # 简化的时序卷积
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, 
                                        kernel_size=(3, 1), padding=(1, 0))
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # x: (B, C, T, V) - batch, channels, time, vertices
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SignLanguageLite(nn.Module):
    """轻量化手语识别模型"""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # 标签平滑
        self.label_smoothing = getattr(args, 'label_smoothing', 0.1)
        
        # 关键点配置 (简化版)
        self.body_joints = 9      # 上半身关键点
        self.hand_joints = 21     # 单手关键点
        
        # 输入特征维度: 关键点坐标(2) + 置信度(1) = 3
        self.input_dim = 3
        
        # 姿态嵌入层
        self.pose_embed = nn.Linear(self.input_dim, 64)  # 3D输入 -> 64维特征
        
        # 轻量GCN编码器
        self.body_gcn = nn.Sequential(
            LightweightGCN(64, 128, self.body_joints),
            LightweightGCN(128, 256, self.body_joints),
        )
        
        self.hand_gcn = nn.Sequential(
            LightweightGCN(64, 128, self.hand_joints),
            LightweightGCN(128, 256, self.hand_joints),
        )
        
        # 特征融合
        self.fusion = nn.Linear(256 * 3, 512)  # body + left_hand + right_hand
        
        # 加载轻量语言模型
        self.mt5_tokenizer = T5Tokenizer.from_pretrained(args.mt5_path, legacy=True)
        self.mt5_model = MT5ForConditionalGeneration.from_pretrained(args.mt5_path)
        
        # 获取mT5的实际hidden_size并调整投影层
        mt5_hidden_size = self.mt5_model.config.d_model  # 通常是512
        self.proj = nn.Linear(512, mt5_hidden_size)
        
        # 添加层归一化以稳定训练
        self.layer_norm = nn.LayerNorm(mt5_hidden_size)
        
        print(f"mT5 hidden_size: {mt5_hidden_size}")
        
        # 冻结部分语言模型参数以节省显存
        self._freeze_mt5_layers()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化新添加层的权重"""
        for module in [self.pose_embed, self.fusion, self.proj]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def _freeze_mt5_layers(self):
        """冻结mT5的部分层 - 解冻更多层以提高学习能力"""
        # 解冻最后3层decoder和最后2层encoder
        for name, param in self.mt5_model.named_parameters():
            # 解冻 decoder的最后3层 (block.3, block.4, block.5)
            if any(f'decoder.block.{i}' in name for i in [3, 4, 5]):
                param.requires_grad = True
            # 解冻 encoder的最后2层 (block.4, block.5)
            elif any(f'encoder.block.{i}' in name for i in [4, 5]):
                param.requires_grad = True
            # 解冻 lm_head
            elif 'lm_head' in name:
                param.requires_grad = True
            # 解冻 final_layer_norm
            elif 'final_layer_norm' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def encode_pose(self, pose_data):
        """编码姿态数据"""
        # pose_data: dict with 'body', 'left', 'right' keys
        # 每个: (B, T, V, 3) - 坐标x, y + 置信度
        
        B, T = pose_data['body'].shape[:2]
        
        # 确保数据类型正确
        body = pose_data['body'].float()
        left = pose_data['left'].float()
        right = pose_data['right'].float()
        
        # 处理NaN和Inf值
        body = torch.nan_to_num(body, nan=0.0, posinf=1.0, neginf=-1.0)
        left = torch.nan_to_num(left, nan=0.0, posinf=1.0, neginf=-1.0)
        right = torch.nan_to_num(right, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 裁剪到合理范围
        body = torch.clamp(body, -10.0, 10.0)
        left = torch.clamp(left, -10.0, 10.0)
        right = torch.clamp(right, -10.0, 10.0)
        
        # 嵌入坐标 (输入维度是3: x, y, confidence)
        body_feat = self.pose_embed(body)  # (B, T, V, 64)
        left_feat = self.pose_embed(left)
        right_feat = self.pose_embed(right)
        
        # 调整维度为 (B, C, T, V)
        body_feat = rearrange(body_feat, 'b t v c -> b c t v')
        left_feat = rearrange(left_feat, 'b t v c -> b c t v')
        right_feat = rearrange(right_feat, 'b t v c -> b c t v')
        
        # GCN编码
        body_feat = self.body_gcn(body_feat)  # (B, 256, T, V)
        left_feat = self.hand_gcn(left_feat)
        right_feat = self.hand_gcn(right_feat)
        
        # 全局池化
        body_feat = body_feat.mean(dim=-1).permute(0, 2, 1)  # (B, T, 256)
        left_feat = left_feat.mean(dim=-1).permute(0, 2, 1)
        right_feat = right_feat.mean(dim=-1).permute(0, 2, 1)
        
        # 融合
        fused = torch.cat([body_feat, left_feat, right_feat], dim=-1)
        fused = self.fusion(fused)  # (B, T, 512)
        
        # 投影并归一化
        output = self.proj(fused)
        output = self.layer_norm(output)
        
        return output
    
    def forward(self, src_input, tgt_input):
        """训练前向传播"""
        # 编码姿态
        pose_embeds = self.encode_pose(src_input)
        
        # 准备注意力掩码
        attention_mask = src_input['attention_mask']
        
        # 编码目标文本
        tgt_tokens = self.mt5_tokenizer(
            tgt_input['gt_sentence'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=50
        )
        
        labels = tgt_tokens['input_ids'].to(pose_embeds.device)
        labels[labels == self.mt5_tokenizer.pad_token_id] = -100
        
        # 确保数据类型匹配 - 不使用autocast，直接用float32
        pose_embeds = pose_embeds.float()
        attention_mask = attention_mask.float()
        
        # mT5前向传播 - 使用标签平滑
        outputs = self.mt5_model(
            inputs_embeds=pose_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        # 如果启用标签平滑，重新计算loss
        if self.label_smoothing > 0:
            # 获取logits并计算平滑后loss
            logits = outputs.logits
            vocab_size = logits.size(-1)
            
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算平滑的交叉熵损失
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=self.label_smoothing
            )
            loss = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )
            return loss
        
        return outputs.loss
    
    @torch.no_grad()
    def generate(self, src_input, max_new_tokens=50, num_beams=4, 
                 length_penalty=1.0, no_repeat_ngram_size=2):
        """推理生成"""
        pose_embeds = self.encode_pose(src_input)
        attention_mask = src_input['attention_mask']
        
        # 确保类型匹配
        pose_embeds = pose_embeds.float()
        attention_mask = attention_mask.float()
        
        outputs = self.mt5_model.generate(
            inputs_embeds=pose_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
            do_sample=False,  # 使用确定性解码
        )
        
        # 解码
        texts = self.mt5_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return texts