"""
红外相机边缘检测系统
基于树莓派 + OpenCV + 嵌入式Linux
"""

__version__ = "1.0.0"
__author__ = "红外边缘检测系统"

from .camera import InfraredCamera
from .processor import InfraredProcessor
from .edge_detector import EdgeDetector
from .monitor import PerformanceMonitor
from .config import SystemConfig

__all__ = [
    'InfraredCamera',
    'InfraredProcessor', 
    'EdgeDetector',
    'PerformanceMonitor',
    'SystemConfig'
]