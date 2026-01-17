import torch
import torch.nn as nn
import os
import math

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


class PositionalEncoding(nn.Module):
    """注入时序信息的位置编码"""
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, T, C) -> (T, B, C) for indexing
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x.permute(1, 0, 2))


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
        
        # 位置编码 (增强前端时序能力)
        self.pos_encoder = PositionalEncoding(mt5_hidden_size, max_len=512)
        
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
        """
        冻结策略调整:
        1. 必须解冻 Encoder: 因为输入是姿态而非文本，Encoder必须重新学习
        2. 冻结部分 Decoder: 保留预训练的语言生成能力，节省显存
        """
        print("应用优化后的冻结策略: 解冻 Encoder, 冻结部分 Decoder")
        
        # 1. 解冻整个 Encoder
        for param in self.mt5_model.encoder.parameters():
            param.requires_grad = True
            
        # 2. 冻结 Decoder 的前半部分 (假设有8层，冻结前4层)
        # mT5-small config: num_decoder_layers=8, num_heads=6, d_model=512
        for name, param in self.mt5_model.decoder.named_parameters():
            if 'block.0.' in name or 'block.1.' in name or 'block.2.' in name or 'block.3.' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
        # 3. 始终解冻 lm_head
        for param in self.mt5_model.lm_head.parameters():
            param.requires_grad = True
    
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

        # 注入位置信息
        output = self.pos_encoder(output)
        
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