"""
红外图像预处理模块
包含噪声抑制、对比度增强、温度数据转换等功能
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

class InfraredProcessor:
    """红外图像处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.clahe = None
        self.temporal_buffer = None
        self.temporal_index = 0
        self.min_temp = None
        self.max_temp = None
        
    def setup_clahe(self, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)):
        """设置自适应直方图均衡化"""
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit, 
            tileGridSize=tile_size
        )
        self.logger.info(f"CLAHE已配置: clip_limit={clip_limit}, tile_size={tile_size}")
    
    def setup_temporal_filter(self, buffer_size: int = 3):
        """设置时域滤波器"""
        self.temporal_buffer = None
        self.temporal_index = 0
        self.logger.info(f"时域滤波器已配置: buffer_size={buffer_size}")
    
    def convert_temperature_to_8bit(self, infrared_frame: np.ndarray) -> np.ndarray:
        """将16位温度数据转换为8位可视化图像
        
        Args:
            infrared_frame: 16位温度数据，范围通常为-40°C到300°C
            
        Returns:
            8位灰度图像，0-255范围
        """
        if infrared_frame is None:
            return None
            
        # 计算温度范围（假设16位数据代表温度值）
        if self.min_temp is None or self.max_temp is None:
            # 动态计算min/max，忽略异常值
            valid_data = infrared_frame[infrared_frame > 100]  # 过滤掉明显异常值
            if len(valid_data) > 0:
                self.min_temp = np.percentile(valid_data, 1)
                self.max_temp = np.percentile(valid_data, 99)
            else:
                self.min_temp = infrared_frame.min()
                self.max_temp = infrared_frame.max()
        
        # 温度范围限制
        temp_range = max(self.max_temp - self.min_temp, 1)
        
        # 线性映射到8位
        normalized = (infrared_frame - self.min_temp) / temp_range
        normalized = np.clip(normalized, 0, 1)
        
        return (normalized * 255).astype(np.uint8)
    
    def apply_temporal_filter(self, frame: np.ndarray) -> np.ndarray:
        """应用时域滤波
        
        Args:
            frame: 输入图像
            
        Returns:
            滤波后的图像
        """
        if self.temporal_buffer is None:
            # 初始化时域缓冲区
            self.temporal_buffer = np.zeros(
                (3, frame.shape[0], frame.shape[1]), dtype=frame.dtype
            )
            self.temporal_buffer[:] = frame
            return frame
        
        # 更新缓冲区
        self.temporal_buffer[self.temporal_index] = frame
        self.temporal_index = (self.temporal_index + 1) % 3
        
        # 计算时域平均
        filtered = np.mean(self.temporal_buffer, axis=0).astype(frame.dtype)
        
        return filtered
    
    def apply_gaussian_blur(self, frame: np.ndarray, kernel_size: Tuple[int, int] = (5, 5)) -> np.ndarray:
        """应用高斯模糊去噪
        
        Args:
            frame: 输入图像
            kernel_size: 高斯核大小
            
        Returns:
            去噪后的图像
        """
        return cv2.GaussianBlur(frame, kernel_size, 0)
    
    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """应用自适应直方图均衡化
        
        Args:
            frame: 输入图像
            
        Returns:
            增强后的图像
        """
        if self.clahe is None:
            self.setup_clahe()
        
        if self.clahe is None:
            return frame  # 安全返回
        
        return self.clahe.apply(frame)
    
    def process_frame(self, infrared_frame: np.ndarray, config: Optional[dict] = None) -> dict:
        """完整的图像处理流程
        
        Args:
            infrared_frame: 16位红外原始数据
            config: 处理配置
            
        Returns:
            包含处理结果的字典
        """
        if infrared_frame is None:
            return None
            
        result = {
            'original': infrared_frame,
            'temperature_range': None,
            'visual_8bit': None,
            'denoised': None,
            'enhanced': None
        }
        
        try:
            # 转换为8位可视化图像
            visual_8bit = self.convert_temperature_to_8bit(infrared_frame)
            result['visual_8bit'] = visual_8bit
            
            if config:
                self.setup_clahe(
                    clip_limit=config.get('clahe_clip_limit', 2.0),
                    tile_size=config.get('clahe_tile_size', (8, 8))
                )
                self.setup_temporal_filter(
                    buffer_size=config.get('temporal_filter_size', 3)
                )
            
            # 时域滤波
            temp_filtered = self.apply_temporal_filter(visual_8bit)
            result['temporal_filtered'] = temp_filtered
            
            # 高斯去噪
            denoised = self.apply_gaussian_blur(
                temp_filtered, 
                config.get('gaussian_kernel', (5, 5)) if config else (5, 5)
            )
            result['denoised'] = denoised
            
            # 对比度增强
            enhanced = self.apply_clahe(denoised)
            result['enhanced'] = enhanced
            
            # 记录温度范围
            result['temperature_range'] = {
                'min': self.min_temp,
                'max': self.max_temp,
                'range': self.max_temp - self.min_temp
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"图像处理失败: {e}")
            return result
    
    def reset_temperature_range(self):
        """重置温度范围计算"""
        self.min_temp = None
        self.max_temp = None
    
    def get_temperature_at_point(self, infrared_frame: np.ndarray, x: int, y: int) -> Optional[float]:
        """获取指定坐标的温度值
        
        Args:
            infrared_frame: 16位温度数据
            x, y: 坐标
            
        Returns:
            温度值或None
        """
        if infrared_frame is None or not (0 <= x < infrared_frame.shape[1] and 0 <= y < infrared_frame.shape[0]):
            return None
            
        raw_value = infrared_frame[y, x]
        
        # 简单的温度映射（实际应用中需要根据相机校准参数调整）
        # 假设16位数据线性映射到-20°C到150°C
        temperature = (raw_value / 65535.0) * 170 - 20
        
        return temperature
    
    def apply_roi(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        """提取感兴趣区域
        
        Args:
            frame: 输入图像
            roi: (x, y, width, height)
            
        Returns:
            ROI区域图像
        """
        x, y, w, h = roi
        return frame[y:y+h, x:x+w]