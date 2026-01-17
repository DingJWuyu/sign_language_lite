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


class TemporalModule(nn.Module):
    """时序建模模块 - Bi-GRU + 下采样"""
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3, downsample_factor=2):
        super().__init__()
        self.downsample_factor = downsample_factor
        
        # Bi-GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出投影 (Bi-GRU 输出是 2*hidden_dim)
        self.proj = nn.Linear(hidden_dim * 2, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            output: (B, T//downsample_factor, C)
        """
        B, T, C = x.shape
        
        # Bi-GRU
        gru_out, _ = self.gru(x)  # (B, T, hidden*2)
        
        # 下采样: 均匀取样
        if self.downsample_factor > 1:
            new_T = T // self.downsample_factor
            indices = torch.linspace(0, T-1, new_T, dtype=torch.long, device=x.device)
            gru_out = gru_out[:, indices, :]  # (B, new_T, hidden*2)
        
        # 投影回原始维度
        output = self.proj(gru_out)
        output = self.norm(output)
        output = self.dropout(output)
        
        return output


class GlossHead(nn.Module):
    """
    Gloss 预测头 - 用于 CTC Loss
    在视觉编码器之后预测 Gloss 序列
    """
    def __init__(self, input_dim, vocab_size, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C)
        Returns:
            logits: (B, T, vocab_size) - 用于 CTC
        """
        return self.fc(x)


class SignLanguageLite(nn.Module):
    """轻量化手语识别模型 - Phase 2: 增强版"""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # 标签平滑
        self.label_smoothing = getattr(args, 'label_smoothing', 0.1)
        
        # 关键点配置 (新增面部)
        self.body_joints = 9       # 上半身关键点
        self.hand_joints = 21      # 单手关键点
        self.face_joints = 32      # 面部关键点 (眉毛+眼睛+嘴唇)
        
        # Gloss 词表大小 (从 args 获取，默认 2000)
        self.gloss_vocab_size = getattr(args, 'gloss_vocab_size', 2000)
        self.use_ctc = getattr(args, 'use_ctc', True)
        
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
        
        # 面部 GCN (新增)
        self.face_gcn = nn.Sequential(
            LightweightGCN(64, 128, self.face_joints),
            LightweightGCN(128, 256, self.face_joints),
        )
        
        # 特征融合: body(256) + left(256) + right(256) + face(256) = 1024
        self.fusion = nn.Linear(256 * 4, 512)
        
        # 时序模块 (Phase 2 新增: Bi-GRU + 下采样)
        self.temporal_module = TemporalModule(
            input_dim=512,
            hidden_dim=256,
            num_layers=2,
            dropout=0.3,
            downsample_factor=2  # 128帧 -> 64帧
        )
        
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
        
        # Gloss Head (CTC 辅助监督 - Phase 2 新增)
        if self.use_ctc:
            self.gloss_head = GlossHead(
                input_dim=512,
                vocab_size=self.gloss_vocab_size,
                hidden_dim=256,
                dropout=0.3
            )
        
        print(f"mT5 hidden_size: {mt5_hidden_size}")
        print(f"Gloss vocab size: {self.gloss_vocab_size}, CTC enabled: {self.use_ctc}")
        
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
    
    def encode_pose(self, pose_data, return_pre_temporal=False):
        """
        编码姿态数据 (Phase 2: 增加面部 + 时序模块)
        
        Args:
            pose_data: dict with 'body', 'left', 'right', 'face' keys
            return_pre_temporal: 是否返回时序模块之前的特征 (用于CTC)
        
        Returns:
            output: (B, T', C) 编码后的特征, T' = T // downsample_factor
            pre_temporal: (B, T, C) 时序模块之前的特征 (如果 return_pre_temporal=True)
        """
        B, T = pose_data['body'].shape[:2]
        
        # 确保数据类型正确
        body = pose_data['body'].float()
        left = pose_data['left'].float()
        right = pose_data['right'].float()
        
        # 面部数据 (Phase 2 新增)
        if 'face' in pose_data and pose_data['face'] is not None:
            face = pose_data['face'].float()
            face = torch.nan_to_num(face, nan=0.0, posinf=1.0, neginf=-1.0)
            face = torch.clamp(face, -10.0, 10.0)
            has_face = True
        else:
            # 兼容旧数据: 创建零张量
            face = torch.zeros(B, T, self.face_joints, 3, device=body.device)
            has_face = False
        
        # 处理NaN和Inf值
        body = torch.nan_to_num(body, nan=0.0, posinf=1.0, neginf=-1.0)
        left = torch.nan_to_num(left, nan=0.0, posinf=1.0, neginf=-1.0)
        right = torch.nan_to_num(right, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 裁剪到合理范围
        body = torch.clamp(body, -10.0, 10.0)
        left = torch.clamp(left, -10.0, 10.0)
        right = torch.clamp(right, -10.0, 10.0)
        
        # 嵌入坐标 (输入维度是3: x, y, confidence)
        body_feat = self.pose_embed(body)   # (B, T, V, 64)
        left_feat = self.pose_embed(left)
        right_feat = self.pose_embed(right)
        face_feat = self.pose_embed(face)   # (B, T, 32, 64)
        
        # 调整维度为 (B, C, T, V)
        body_feat = rearrange(body_feat, 'b t v c -> b c t v')
        left_feat = rearrange(left_feat, 'b t v c -> b c t v')
        right_feat = rearrange(right_feat, 'b t v c -> b c t v')
        face_feat = rearrange(face_feat, 'b t v c -> b c t v')
        
        # GCN编码
        body_feat = self.body_gcn(body_feat)   # (B, 256, T, V)
        left_feat = self.hand_gcn(left_feat)
        right_feat = self.hand_gcn(right_feat)
        face_feat = self.face_gcn(face_feat)   # (B, 256, T, 32)
        
        # 全局池化
        body_feat = body_feat.mean(dim=-1).permute(0, 2, 1)   # (B, T, 256)
        left_feat = left_feat.mean(dim=-1).permute(0, 2, 1)
        right_feat = right_feat.mean(dim=-1).permute(0, 2, 1)
        face_feat = face_feat.mean(dim=-1).permute(0, 2, 1)   # (B, T, 256)
        
        # 融合 (Phase 2: 增加面部)
        fused = torch.cat([body_feat, left_feat, right_feat, face_feat], dim=-1)  # (B, T, 1024)
        fused = self.fusion(fused)  # (B, T, 512)
        
        # 保存时序模块之前的特征 (用于CTC)
        pre_temporal = fused
        
        # 时序模块 (Phase 2 新增: Bi-GRU + 下采样)
        temporal_out = self.temporal_module(fused)  # (B, T//2, 512)
        
        # 投影并归一化
        output = self.proj(temporal_out)
        output = self.layer_norm(output)

        # 注入位置信息
        output = self.pos_encoder(output)
        
        if return_pre_temporal:
            return output, pre_temporal
        return output
    
    def forward(self, src_input, tgt_input, gloss_labels=None, gloss_lengths=None):
        """
        训练前向传播 (Phase 2: 支持 CTC Loss)
        
        Args:
            src_input: 源输入 (姿态数据)
            tgt_input: 目标输入 (文本)
            gloss_labels: Gloss 标签 (用于 CTC), shape (B, max_gloss_len)
            gloss_lengths: Gloss 长度, shape (B,)
        
        Returns:
            loss: 总损失 (Translation Loss + CTC Loss)
            或 dict: {'loss': total_loss, 'trans_loss': trans_loss, 'ctc_loss': ctc_loss}
        """
        # 编码姿态 (同时返回时序模块之前的特征用于CTC)
        if self.use_ctc and gloss_labels is not None:
            pose_embeds, pre_temporal = self.encode_pose(src_input, return_pre_temporal=True)
        else:
            pose_embeds = self.encode_pose(src_input)
            pre_temporal = None
        
        # 准备注意力掩码 (需要下采样以匹配 pose_embeds)
        attention_mask = src_input['attention_mask']
        B, T_new, _ = pose_embeds.shape
        T_orig = attention_mask.shape[1]
        if T_new != T_orig:
            # 下采样 attention_mask
            indices = torch.linspace(0, T_orig-1, T_new, dtype=torch.long, device=attention_mask.device)
            attention_mask = attention_mask[:, indices]
        
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
            trans_loss = loss_fct(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )
        else:
            trans_loss = outputs.loss
        
        # CTC Loss (Phase 2 新增)
        ctc_loss = torch.tensor(0.0, device=pose_embeds.device)
        if self.use_ctc and pre_temporal is not None and gloss_labels is not None:
            # Gloss 预测
            gloss_logits = self.gloss_head(pre_temporal)  # (B, T, vocab_size)
            
            # CTC Loss 需要 (T, B, C) 格式
            gloss_logits = gloss_logits.permute(1, 0, 2)  # (T, B, vocab_size)
            gloss_logits = gloss_logits.log_softmax(dim=-1)
            
            # 输入长度 (所有样本使用相同的序列长度)
            input_lengths = torch.full(
                (gloss_logits.size(1),), 
                gloss_logits.size(0), 
                dtype=torch.long, 
                device=pose_embeds.device
            )
            
            # 计算 CTC Loss
            ctc_loss_fn = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
            ctc_loss = ctc_loss_fn(
                gloss_logits, 
                gloss_labels, 
                input_lengths, 
                gloss_lengths
            )
            
            # 处理 NaN
            if torch.isnan(ctc_loss) or torch.isinf(ctc_loss):
                ctc_loss = torch.tensor(0.0, device=pose_embeds.device)
        
        # 总损失 = Translation Loss + λ * CTC Loss
        ctc_weight = getattr(self.args, 'ctc_weight', 0.5)
        total_loss = trans_loss + ctc_weight * ctc_loss
        
        return total_loss
    
    @torch.no_grad()
    def generate(self, src_input, max_new_tokens=50, num_beams=4, 
                 length_penalty=1.0, no_repeat_ngram_size=2):
        """推理生成"""
        pose_embeds = self.encode_pose(src_input)
        attention_mask = src_input['attention_mask']
        
        # 下采样后需要调整 attention_mask
        B, T_new, _ = pose_embeds.shape
        T_orig = attention_mask.shape[1]
        if T_new != T_orig:
            # 下采样 attention_mask
            indices = torch.linspace(0, T_orig-1, T_new, dtype=torch.long, device=attention_mask.device)
            attention_mask = attention_mask[:, indices]
        
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