import cv2
import torch
import numpy as np
from collections import deque
import sys
sys.path.append('./demo/rtmlib-main')

from rtmlib import Wholebody
from models_lite import SignLanguageLite
from config_lite import InferenceConfig, mt5_path


class RealtimeSignRecognizer:
    """实时手语识别器"""
    
    def __init__(self, model_path='best_model.pth'):
        self.config = InferenceConfig()
        
        # 初始化姿态估计器 (轻量模式)
        self.pose_estimator = Wholebody(
            mode='lightweight',  # 使用轻量模式
            backend='onnxruntime',
            device='cuda'
        )
        
        # 加载手语识别模型
        class Args:
            mt5_path = mt5_path
            max_length = self.config.max_length
        
        self.model = SignLanguageLite(Args())
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.config.device)
        self.model.eval()
        
        # 帧缓冲区
        self.frame_buffer = deque(maxlen=self.config.max_length)
        
        # 关键点索引 (简化版)
        self.body_indices = [0] + list(range(5, 13))  # 9个上半身点
        self.left_hand_indices = list(range(91, 112))  # 21个左手点
        self.right_hand_indices = list(range(112, 133))  # 21个右手点
        
    def extract_pose(self, frame):
        """提取姿态关键点"""
        keypoints, scores = self.pose_estimator(frame)
        
        if len(keypoints) == 0:
            return None
            
        kp = keypoints[0]  # 取第一个人
        score = scores[0]
        
        # 提取各部分关键点
        pose_data = {
            'body': kp[self.body_indices, :2],
            'left': kp[self.left_hand_indices, :2] - kp[91, :2],  # 相对于手腕
            'right': kp[self.right_hand_indices, :2] - kp[112, :2],
            'score': score
        }
        
        return pose_data
    
    def process_frame(self, frame):
        """处理单帧"""
        pose_data = self.extract_pose(frame)
        
        if pose_data is not None:
            self.frame_buffer.append(pose_data)
        
        return pose_data
    
    @torch.no_grad()
    def recognize(self):
        """识别当前缓冲区中的手语"""
        if len(self.frame_buffer) < 30:  # 最少需要30帧
            return None
        
        # 构建输入
        poses = list(self.frame_buffer)
        
        body = np.stack([p['body'] for p in poses])
        left = np.stack([p['left'] for p in poses])
        right = np.stack([p['right'] for p in poses])
        
        # 转换为张量
        src_input = {
            'body': torch.FloatTensor(body).unsqueeze(0).to(self.config.device),
            'left': torch.FloatTensor(left).unsqueeze(0).to(self.config.device),
            'right': torch.FloatTensor(right).unsqueeze(0).to(self.config.device),
            'attention_mask': torch.ones(1, len(poses)).to(self.config.device)
        }
        
        # 生成翻译
        result = self.model.generate(src_input)[0]
        
        return result
    
    def run(self):
        """运行实时识别"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        last_result = ""
        frame_count = 0
        
        print("按 'q' 退出, 按 'r' 识别当前手语")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理帧
            pose_data = self.process_frame(frame)
            
            # 可视化
            if pose_data is not None:
                self._draw_skeleton(frame, pose_data)
            
            # 显示信息
            cv2.putText(frame, f"Frames: {len(self.frame_buffer)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Result: {last_result}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Sign Language Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                result = self.recognize()
                if result:
                    last_result = result
                    print(f"识别结果: {result}")
                    self.frame_buffer.clear()  # 清空缓冲区
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _draw_skeleton(self, frame, pose_data):
        """绘制骨架"""
        # 简化的可视化，仅绘制关键点
        for part in ['body']:
            points = pose_data[part]
            for point in points:
                x, y = int(point[0]), int(point[1])
                if 0 < x < frame.shape[1] and 0 < y < frame.shape[0]:
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)


if __name__ == '__main__':
    recognizer = RealtimeSignRecognizer()
    recognizer.run()