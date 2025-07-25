"""
边缘检测算法模块
支持多种边缘检测算法：Canny、Sobel、Laplacian、自适应算法
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

class EdgeDetector:
    """边缘检测器类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def detect_canny(self, frame: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
        """Canny边缘检测
        
        Args:
            frame: 输入灰度图像
            low_threshold: 低阈值
            high_threshold: 高阈值
            
        Returns:
            边缘检测结果，二值图像
        """
        if frame is None:
            return None
            
        edges = cv2.Canny(frame, low_threshold, high_threshold)
        return edges.astype(np.uint8)
    
    def detect_sobel(self, frame: np.ndarray, kernel_size: int = 3) -> dict:
        """Sobel边缘检测
        
        Args:
            frame: 输入灰度图像
            kernel_size: Sobel核大小
            
        Returns:
            包含水平和垂直边缘的字典
        """
        if frame is None:
            return None
            
        sobel_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=kernel_size)
        sobel_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=kernel_size)
        
        # 计算梯度幅值和方向
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        direction = np.arctan2(sobel_y, sobel_x)
        
        # 归一化到8位
        magnitude_norm = np.uint8(255 * magnitude / np.max(magnitude))
        
        return {
            'x': np.abs(sobel_x).astype(np.uint8),
            'y': np.abs(sobel_y).astype(np.uint8),
            'magnitude': magnitude_norm,
            'direction': direction
        }
    
    def detect_laplacian(self, frame: np.ndarray) -> np.ndarray:
        """Laplacian边缘检测
        
        Args:
            frame: 输入灰度图像
            
        Returns:
            Laplacian边缘检测结果
        """
        if frame is None:
            return None
            
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        return np.abs(laplacian).astype(np.uint8)
    
    def detect_adaptive(self, frame: np.ndarray, method: str = 'otsu') -> np.ndarray:
        """自适应边缘检测
        
        Args:
            frame: 输入灰度图像
            method: 自适应方法 ('otsu', 'mean', 'gaussian')
            
        Returns:
            自适应边缘检测结果
        """
        if frame is None:
            return None
            
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        
        if method == 'otsu':
            _, edges = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'mean':
            edges = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
            )
        elif method == 'gaussian':
            edges = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            edges = self.detect_canny(frame)
            
        return edges
    
    def detect_temperature_gradient(self, infrared_frame: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        """基于温度梯度的边缘检测（红外特有）
        
        Args:
            infrared_frame: 16位温度数据
            threshold: 温度梯度阈值
            
        Returns:
            温度梯度边缘检测结果
        """
        if infrared_frame is None:
            return None
            
        # 计算温度梯度
        grad_x = cv2.Sobel(infrared_frame.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(infrared_frame.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        
        # 计算梯度幅值
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 应用温度梯度阈值
        edges = gradient_magnitude > threshold
        
        return edges.astype(np.uint8) * 255
    
    def detect_edges(self, frame: np.ndarray, method: str = 'canny', **kwargs) -> dict:
        """统一的边缘检测接口
        
        Args:
            frame: 输入灰度图像
            method: 检测方法 ('canny', 'sobel', 'laplacian', 'adaptive', 'temperature')
            **kwargs: 各方法特定参数
            
        Returns:
            包含边缘检测结果的字典
        """
        if frame is None:
            return None
            
        result = {
            'method': method,
            'edges': None,
            'confidence': None,
            'parameters': kwargs
        }
        
        try:
            if method == 'canny':
                edges = self.detect_canny(
                    frame,
                    low_threshold=kwargs.get('low_threshold', 50),
                    high_threshold=kwargs.get('high_threshold', 150)
                )
                
            elif method == 'sobel':
                edges_dict = self.detect_sobel(
                    frame,
                    kernel_size=kwargs.get('kernel_size', 3)
                )
                edges = edges_dict['magnitude']
                
            elif method == 'laplacian':
                edges = self.detect_laplacian(frame)
                
            elif method == 'adaptive':
                edges = self.detect_adaptive(
                    frame,
                    method=kwargs.get('adaptive_method', 'otsu')
                )
                
            elif method == 'temperature':
                edges = self.detect_temperature_gradient(
                    frame,
                    threshold=kwargs.get('temperature_threshold', 5.0)
                )
                
            else:
                edges = self.detect_canny(frame)
                
            result['edges'] = edges
            
            # 计算边缘密度作为置信度
            edge_density = np.sum(edges > 0) / edges.size
            result['confidence'] = min(edge_density * 10, 1.0)  # 归一化置信度
            
            return result
            
        except Exception as e:
            self.logger.error(f"边缘检测失败: {e}")
            return result
    
    def apply_edge_overlay(self, original_frame: np.ndarray, edges: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), alpha: float = 0.8) -> np.ndarray:
        """边缘叠加显示
        
        Args:
            original_frame: 原始图像（灰度或彩色）
            edges: 边缘检测结果
            color: 边缘颜色 (B, G, R)
            alpha: 透明度
            
        Returns:
            边缘叠加后的彩色图像
        """
        if original_frame is None or edges is None:
            return None
            
        # 确保原始图像是彩色
        if len(original_frame.shape) == 2:
            display_frame = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2BGR)
        else:
            display_frame = original_frame.copy()
        
        # 创建边缘掩码
        edge_mask = edges > 0
        display_frame[edge_mask] = color
        
        return display_frame
    
    def refine_edges(self, edges: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """边缘细化
        
        Args:
            edges: 原始边缘图像
            kernel_size: 形态学核大小
            
        Returns:
            细化后的边缘图像
        """
        if edges is None:
            return None
            
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
        
        # 形态学操作细化边缘
        edges_refined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges_refined