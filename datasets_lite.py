"""
轻量化手语数据集
适配 CSL-Daily 数据集格式
"""
import torch
import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
import os
import random
import numpy as np
import pickle
import gzip
import copy


def load_dataset_file(filename):
    """加载gzip压缩的pickle文件"""
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


# ============ 面部关键点索引 (COCO-WholeBody 格式) ============
# 眉毛: 左眉 [33-37], 右眉 [43-47] -> 共10点
# 眼睛: 左眼 [38-42], 右眼 [48-52] -> 共10点  
# 嘴唇: 外轮廓 [76-87] -> 共12点
# 总计: 32点面部关键点
FACE_EYEBROW_LEFT = list(range(33, 38))   # 5点
FACE_EYEBROW_RIGHT = list(range(43, 48))  # 5点
FACE_EYE_LEFT = list(range(38, 43))       # 5点  
FACE_EYE_RIGHT = list(range(48, 53))      # 5点
FACE_MOUTH = list(range(76, 88))          # 12点
FACE_INDICES = FACE_EYEBROW_LEFT + FACE_EYEBROW_RIGHT + FACE_EYE_LEFT + FACE_EYE_RIGHT + FACE_MOUTH  # 32点


def load_part_kp(skeletons, confs, force_ok=False, use_face=True):
    """
    从全身关键点中提取各部位关键点 (Phase 2: 增加面部)
    
    Args:
        skeletons: 关键点坐标列表, 每个元素形状 (1, 133, 2)
        confs: 置信度列表, 每个元素形状 (1, 133)
        force_ok: 是否强制返回有效结果
        use_face: 是否提取面部关键点
    
    Returns:
        kps_with_scores: dict, 包含 body, left, right, face 的关键点
    """
    thr = 0.3  # 置信度阈值
    kps_with_scores = {}
    scale = None
    
    # 处理各部分关键点
    parts = ['body', 'left', 'right']
    if use_face:
        parts.append('face')
    
    for part in parts:
        kps = []
        confidences = []
        
        for skeleton, conf in zip(skeletons, confs):
            skeleton = skeleton[0] if len(skeleton.shape) == 3 else skeleton
            conf = conf[0] if len(conf.shape) == 2 else conf
            
            if part == 'body':
                # 上半身9个关键点: 鼻子 + 肩膀到臀部
                hand_kp2d = skeleton[[0] + [i for i in range(5, 13)], :]
                confidence = conf[[0] + [i for i in range(5, 13)]]
            elif part == 'left':
                # 左手21个关键点 (索引91-111)
                hand_kp2d = skeleton[91:112, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]  # 相对于手腕归一化
                confidence = conf[91:112]
            elif part == 'right':
                # 右手21个关键点 (索引112-132)
                hand_kp2d = skeleton[112:133, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]  # 相对于手腕归一化
                confidence = conf[112:133]
            elif part == 'face':
                # 面部32个关键点: 眉毛+眼睛+嘴唇
                # 注意: COCO-WholeBody 中面部点从索引23开始
                face_offset = 23  # 面部关键点在133点中的起始偏移
                face_indices = [face_offset + i for i in FACE_INDICES if face_offset + i < 91]
                if len(face_indices) < 10:  # 兜底: 使用简化的面部点
                    face_indices = list(range(23, 55))  # 取前32个面部点
                hand_kp2d = skeleton[face_indices, :]
                # 以鼻尖(索引0)为中心归一化
                nose_pos = skeleton[0, :]
                hand_kp2d = hand_kp2d - nose_pos
                confidence = conf[face_indices]
            else:
                raise NotImplementedError(f"Unknown part: {part}")
            
            kps.append(hand_kp2d)
            confidences.append(confidence)
            
        kps = np.stack(kps, axis=0)  # (T, V, 2)
        confidences = np.stack(confidences, axis=0)  # (T, V)
        
        if part == 'body':
            # 对身体部分进行归一化
            result, scale, _ = crop_scale(
                np.concatenate([kps, confidences[..., None]], axis=-1), 
                thr
            )
        elif part == 'face':
            # 面部独立归一化 (位移较小，需要放大)
            result = np.concatenate([kps, confidences[..., None]], axis=-1)
            # 面部使用固定的缩放因子，因为位移很小
            face_scale = 0.2 if scale and scale > 0 else 1.0
            result[..., :2] = result[..., :2] / face_scale
            result = np.clip(result, -1, 1)
            result[result[..., 2] <= thr] = 0
        else:
            # 手部已经相对于手腕归一化
            assert scale is not None
            result = np.concatenate([kps, confidences[..., None]], axis=-1)
            if scale == 0:
                result = np.zeros(result.shape)
            else:
                result[..., :2] = result[..., :2] / scale
                result = np.clip(result, -1, 1)
                result[result[..., 2] <= thr] = 0
            
        kps_with_scores[part] = torch.tensor(result, dtype=torch.float32)
        
    return kps_with_scores


def crop_scale(motion, thr):
    """
    将关键点归一化到 [-1, 1] 范围
    
    Args:
        motion: (T, V, 3) 关键点坐标和置信度
        thr: 置信度阈值
    
    Returns:
        result: 归一化后的关键点
        scale: 缩放因子
        offset: 偏移量
    """
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2] > thr][:, :2]
    
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None
    
    xmin = min(valid_coords[:, 0])
    xmax = max(valid_coords[:, 0])
    ymin = min(valid_coords[:, 1])
    ymax = max(valid_coords[:, 1])
    
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros(motion.shape), 0, None
    
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    result[result[..., 2] <= thr] = 0
    
    return result, scale, [xs, ys]


class SignLanguageDataset(data.Dataset):
    """
    轻量化手语翻译数据集
    
    数据格式要求:
    - 标签文件: gzip压缩的pickle文件，包含字典
    - 每个样本包含: name, video_path, text, gloss(可选)
    - 姿态文件: pickle文件，包含 keypoints 和 scores
    """
    
    def __init__(self, label_path, config, phase='train'):
        """
        Args:
            label_path: 标签文件路径
            config: 配置对象，包含 max_length, pose_dir 等
            phase: 'train', 'dev' 或 'test'
        """
        super().__init__()
        self.config = config
        self.phase = phase
        self.max_length = config.max_length
        
        # 获取姿态文件目录
        # 尝试从 config 中获取，否则使用默认路径
        if hasattr(config, 'pose_dir'):
            self.pose_dir = config.pose_dir
        else:
            # 从标签路径推断
            base_dir = os.path.dirname(label_path)
            self.pose_dir = os.path.join(base_dir, 'pose_format')
        
        # 加载数据
        self.raw_data = self._load_data(label_path)
        self.data_list = list(self.raw_data.keys())
        
        print(f"[{phase}] 加载了 {len(self.data_list)} 个样本")
    
    def _load_data(self, label_path):
        """加载标签数据"""
        try:
            # 尝试作为 gzip pickle 文件加载
            return load_dataset_file(label_path)
        except Exception as e:
            print(f"警告: 无法加载 {label_path}, 错误: {e}")
            return {}
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        """
        获取单个样本
        
        Returns:
            name: 样本名称
            pose_data: 姿态数据字典，包含 body, left, right
            text: 翻译文本
            gloss: 手语词汇序列
        """
        key = self.data_list[index]
        sample = self.raw_data[key]
        
        # 获取文本标注
        text = sample.get('text', '')
        
        # 获取 gloss (如果有)
        if 'gloss' in sample and sample['gloss']:
            if isinstance(sample['gloss'], list):
                gloss = ' '.join(sample['gloss'])
            else:
                gloss = sample['gloss']
        else:
            gloss = ''
        
        # 获取名称
        name = sample.get('name', key)
        
        # 加载姿态数据
        video_path = sample.get('video_path', f"{key}.mp4")
        pose_data = self._load_pose(video_path)
        
        return name, pose_data, text, gloss
    
    def _load_pose(self, video_path):
        """
        加载姿态数据
        
        Args:
            video_path: 视频路径（用于定位对应的pkl文件）
        
        Returns:
            pose_data: 包含 body, left, right 的字典
        """
        # 构建姿态文件路径
        pose_filename = video_path.replace('.mp4', '.pkl')
        pose_path = os.path.join(self.pose_dir, pose_filename)
        
        try:
            with open(pose_path, 'rb') as f:
                pose = pickle.load(f)
        except FileNotFoundError:
            print(f"警告: 找不到姿态文件 {pose_path}")
            # 返回空的姿态数据
            return self._empty_pose()
        
        # 获取时间范围
        if 'start' in pose and 'end' in pose:
            start = pose['start']
            end = pose['end']
            duration = end - start
        else:
            duration = len(pose['scores'])
            start = 0
        
        # 采样帧
        if duration > self.max_length:
            # 随机采样
            if self.phase == 'train':
                tmp = sorted(random.sample(range(duration), k=self.max_length))
            else:
                # 测试时均匀采样
                tmp = np.linspace(0, duration - 1, self.max_length, dtype=int).tolist()
        else:
            tmp = list(range(duration))
        
        # 获取采样后的关键点
        skeletons = pose['keypoints']
        confs = pose['scores']
        
        skeletons_tmp = []
        confs_tmp = []
        for idx in tmp:
            frame_idx = idx + start
            if frame_idx < len(skeletons):
                skeletons_tmp.append(skeletons[frame_idx])
                confs_tmp.append(confs[frame_idx])
        
        if len(skeletons_tmp) == 0:
            return self._empty_pose()
        
        # 提取各部位关键点
        pose_data = load_part_kp(skeletons_tmp, confs_tmp, force_ok=True)
        
        return pose_data
    
    def _empty_pose(self):
        """返回空的姿态数据 (包含面部)"""
        return {
            'body': torch.zeros(1, 9, 3),
            'left': torch.zeros(1, 21, 3),
            'right': torch.zeros(1, 21, 3),
            'face': torch.zeros(1, 32, 3),
        }
    
    def collate_fn(self, batch):
        """
        批次整理函数
        
        将不同长度的序列填充到相同长度
        """
        names, poses, texts, glosses = [], [], [], []
        
        for name, pose, text, gloss in batch:
            names.append(name)
            # 确保 pose 有 'face' 键
            if 'face' not in pose:
                T = pose['body'].shape[0]
                pose['face'] = torch.zeros(T, 32, 3)
            poses.append(pose)
            texts.append(text)
            glosses.append(gloss)
        
        # 构建源输入
        src_input = {}
        
        # 填充各部位的姿态数据 (包含面部)
        for key in ['body', 'left', 'right', 'face']:
            # 获取最大长度
            max_len = max(pose[key].shape[0] for pose in poses)
            
            # 获取各样本的长度
            lengths = [pose[key].shape[0] for pose in poses]
            
            # 填充到相同长度
            padded = []
            for pose in poses:
                seq = pose[key]
                T = seq.shape[0]
                if T < max_len:
                    # 用最后一帧填充
                    padding = seq[-1:].expand(max_len - T, -1, -1)
                    seq = torch.cat([seq, padding], dim=0)
                padded.append(seq)
            
            src_input[key] = torch.stack(padded, dim=0)  # (B, T, V, 3)
            
            # 第一个key时创建attention mask
            if 'attention_mask' not in src_input:
                masks = []
                for length in lengths:
                    mask = torch.zeros(max_len)
                    mask[:length] = 1
                    masks.append(mask)
                src_input['attention_mask'] = torch.stack(masks, dim=0)
                src_input['src_length'] = torch.tensor(lengths)
        
        src_input['names'] = names
        
        # 构建目标输入
        tgt_input = {
            'gt_sentence': texts,
            'gt_gloss': glosses
        }
        
        return src_input, tgt_input


class SignLanguageDatasetSimple(data.Dataset):
    """
    简化版数据集 - 用于没有原始数据时的测试
    使用JSON格式的标签文件
    """
    
    def __init__(self, label_path, config, phase='train'):
        import json
        
        self.config = config
        self.phase = phase
        self.max_length = getattr(config, 'max_length', 128)
        
        # 尝试加载不同格式
        self.data_list = []
        
        try:
            # 尝试JSON格式
            with open(label_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self.raw_data = data
                    self.data_list = list(data.keys())
                elif isinstance(data, list):
                    self.raw_data = {str(i): item for i, item in enumerate(data)}
                    self.data_list = list(self.raw_data.keys())
        except:
            try:
                # 尝试gzip pickle格式
                self.raw_data = load_dataset_file(label_path)
                self.data_list = list(self.raw_data.keys())
            except:
                print(f"警告: 无法加载数据文件 {label_path}")
                self.raw_data = {}
        
        # 姿态目录
        base_dir = os.path.dirname(label_path)
        self.pose_dir = os.path.join(base_dir, 'pose_format')
        
        print(f"[{phase}] 加载了 {len(self.data_list)} 个样本")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        if len(self.data_list) == 0:
            return 'empty', self._empty_pose(), '', ''
        
        key = self.data_list[index]
        sample = self.raw_data[key]
        
        text = sample.get('text', '')
        gloss = sample.get('gloss', '')
        if isinstance(gloss, list):
            gloss = ' '.join(gloss)
        name = sample.get('name', key)
        
        # 尝试加载姿态
        video_path = sample.get('video_path', f"{key}.mp4")
        pose_data = self._load_pose(video_path)
        
        return name, pose_data, text, gloss
    
    def _load_pose(self, video_path):
        """加载姿态数据"""
        pose_filename = video_path.replace('.mp4', '.pkl')
        pose_path = os.path.join(self.pose_dir, pose_filename)
        
        if os.path.exists(pose_path):
            try:
                with open(pose_path, 'rb') as f:
                    pose = pickle.load(f)
                
                # 处理姿态数据
                skeletons = pose['keypoints']
                confs = pose['scores']
                
                # 采样
                T = len(skeletons)
                if T > self.max_length:
                    indices = np.linspace(0, T-1, self.max_length, dtype=int)
                else:
                    indices = range(T)
                
                skeletons_tmp = [skeletons[i] for i in indices]
                confs_tmp = [confs[i] for i in indices]
                
                return load_part_kp(skeletons_tmp, confs_tmp, force_ok=True)
            except Exception as e:
                print(f"加载姿态文件失败: {e}")
        
        return self._empty_pose()
    
    def _empty_pose(self):
        return {
            'body': torch.zeros(1, 9, 3),
            'left': torch.zeros(1, 21, 3),
            'right': torch.zeros(1, 21, 3),
            'face': torch.zeros(1, 32, 3),
        }
    
    def collate_fn(self, batch):
        """批次整理函数"""
        names, poses, texts, glosses = [], [], [], []
        
        for name, pose, text, gloss in batch:
            names.append(name)
            # 确保 pose 有 'face' 键
            if 'face' not in pose:
                T = pose['body'].shape[0]
                pose['face'] = torch.zeros(T, 32, 3)
            poses.append(pose)
            texts.append(text)
            glosses.append(gloss)
        
        src_input = {}
        
        for key in ['body', 'left', 'right', 'face']:
            max_len = max(pose[key].shape[0] for pose in poses)
            lengths = [pose[key].shape[0] for pose in poses]
            
            padded = []
            for pose in poses:
                seq = pose[key]
                T = seq.shape[0]
                if T < max_len:
                    padding = seq[-1:].expand(max_len - T, -1, -1)
                    seq = torch.cat([seq, padding], dim=0)
                padded.append(seq)
            
            src_input[key] = torch.stack(padded, dim=0)
            
            if 'attention_mask' not in src_input:
                masks = []
                for length in lengths:
                    mask = torch.zeros(max_len)
                    mask[:length] = 1
                    masks.append(mask)
                src_input['attention_mask'] = torch.stack(masks, dim=0)
                src_input['src_length'] = torch.tensor(lengths)
        
        src_input['names'] = names
        
        tgt_input = {
            'gt_sentence': texts,
            'gt_gloss': glosses
        }
        
        return src_input, tgt_input


# 测试代码
if __name__ == '__main__':
    from config_lite import TrainConfig, train_label_path
    
    config = TrainConfig()
    
    # 测试数据集
    try:
        dataset = SignLanguageDataset(train_label_path, config, phase='train')
        print(f"数据集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"样本名称: {sample[0]}")
            print(f"身体关键点形状: {sample[1]['body'].shape}")
            print(f"文本: {sample[2]}")
    except Exception as e:
        print(f"测试失败: {e}")
