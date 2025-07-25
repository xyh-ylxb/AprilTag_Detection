"""
系统配置管理模块
支持YAML配置文件和命令行参数
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field

@dataclass
class CameraConfig:
    """相机配置"""
    device_id: int = 0
    width: int = 640
    height: int = 480
    fps: int = 10
    format: str = "Y16"  # 16位灰度红外格式
    buffer_size: int = 4
    
@dataclass
class ProcessingConfig:
    """图像处理配置"""
    gaussian_kernel: tuple = (5, 5)
    clahe_clip_limit: float = 2.0
    clahe_tile_size: tuple = (8, 8)
    temporal_filter_size: int = 3
    
@dataclass
class EdgeDetectionConfig:
    """边缘检测配置"""
    algorithm: str = "canny"  # canny, sobel, laplacian, adaptive
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    sobel_kernel: int = 3
    temperature_threshold: float = 5.0  # 温度梯度阈值
    hough_threshold: int = 80           # 霍夫变换累加器阈值
    hough_min_line_length: int = 50     # 霍夫变换最短直线长度
    hough_max_line_gap: int = 10        # 霍夫变换最大间隔
    
@dataclass
class SystemConfig:
    """系统配置"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    edge_detection: EdgeDetectionConfig = field(default_factory=EdgeDetectionConfig)
    log_level: str = "INFO"
    max_memory_mb: int = 512
    enable_gpu: bool = False
    output_dir: str = "./output"
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'SystemConfig':
        """从YAML文件加载配置"""
        if not os.path.exists(config_path):
            return cls()
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            
        return cls._from_dict(config_dict)
    
    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfig':
        """从字典创建配置对象"""
        camera_config = CameraConfig(**config_dict.get('camera', {}))
        processing_config = ProcessingConfig(**config_dict.get('processing', {}))
        edge_config = EdgeDetectionConfig(**config_dict.get('edge_detection', {}))
        
        system_config = cls(
            camera=camera_config,
            processing=processing_config,
            edge_detection=edge_config,
            log_level=config_dict.get('log_level', 'INFO'),
            max_memory_mb=config_dict.get('max_memory_mb', 512),
            enable_gpu=config_dict.get('enable_gpu', False),
            output_dir=config_dict.get('output_dir', './output')
        )
        
        return system_config
    
    def save_to_file(self, config_path: str):
        """保存配置到YAML文件"""
        config_dict = {
            'camera': asdict(self.camera),
            'processing': asdict(self.processing),
            'edge_detection': asdict(self.edge_detection),
            'log_level': self.log_level,
            'max_memory_mb': self.max_memory_mb,
            'enable_gpu': self.enable_gpu,
            'output_dir': self.output_dir
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def validate(self) -> bool:
        """验证配置有效性"""
        if self.camera.width <= 0 or self.camera.height <= 0:
            return False
        if self.camera.fps <= 0 or self.camera.fps > 30:
            return False
        if self.max_memory_mb < 128:
            return False
        return True

# 默认配置文件路径
DEFAULT_CONFIG_PATH = "./config/infrared_config.yaml"