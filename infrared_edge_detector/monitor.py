"""
系统性能监控模块
监控FPS、内存使用、CPU占用等性能指标
"""

import psutil
import time
import logging
import threading
from typing import Dict, Optional
from collections import deque
import numpy as np

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.logger = logging.getLogger(__name__)
        
        # 性能数据存储
        self.fps_history = deque(maxlen=max_history)
        self.memory_history = deque(maxlen=max_history)
        self.cpu_history = deque(maxlen=max_history)
        self.latency_history = deque(maxlen=max_history)
        
        # 时间记录
        self.last_frame_time = None
        self.start_time = None
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        
        # 统计数据
        self.stats = {
            'fps': 0.0,
            'avg_fps': 0.0,
            'min_fps': float('inf'),
            'max_fps': 0.0,
            'memory_mb': 0.0,
            'cpu_percent': 0.0,
            'avg_latency_ms': 0.0,
            'frame_count': 0,
            'drop_count': 0
        }
        
    def start_monitoring(self):
        """开始性能监控"""
        if self.is_monitoring:
            return
            
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        self.is_monitoring = True
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("性能监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 更新系统信息
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                self.memory_history.append(memory_mb)
                self.cpu_history.append(cpu_percent)
                
                # 更新统计数据
                self.stats.update({
                    'memory_mb': memory_mb,
                    'cpu_percent': cpu_percent
                })
                
                time.sleep(1)  # 每秒更新一次
                
            except Exception as e:
                self.logger.warning(f"监控循环异常: {e}")
                time.sleep(1)
    
    def record_frame(self, frame_size: Optional[tuple] = None):
        """记录帧处理信息
        
        Args:
            frame_size: 帧尺寸 (width, height)
        """
        if not self.is_monitoring:
            return
            
        current_time = time.time()
        
        if self.last_frame_time is None:
            self.last_frame_time = current_time
            return
            
        # 计算FPS
        time_diff = current_time - self.last_frame_time
        fps = 1.0 / max(time_diff, 1e-6)
        
        self.fps_history.append(fps)
        self.last_frame_time = current_time
        
        # 更新统计数据
        self.stats['frame_count'] += 1
        self.stats['fps'] = fps
        
        if fps < self.stats['min_fps']:
            self.stats['min_fps'] = fps
        if fps > self.stats['max_fps']:
            self.stats['max_fps'] = fps
            
        # 计算平均FPS
        if len(self.fps_history) > 0:
            self.stats['avg_fps'] = np.mean(list(self.fps_history))
    
    def record_latency(self, latency_ms: float):
        """记录处理延迟
        
        Args:
            latency_ms: 处理延迟（毫秒）
        """
        self.latency_history.append(latency_ms)
        
        if len(self.latency_history) > 0:
            self.stats['avg_latency_ms'] = np.mean(list(self.latency_history))
    
    def record_frame_drop(self):
        """记录丢帧"""
        self.stats['drop_count'] += 1
    
    def get_current_stats(self) -> Dict[str, float]:
        """获取当前性能统计
        
        Returns:
            性能统计字典
        """
        return self.stats.copy()
    
    def get_detailed_stats(self) -> Dict[str, any]:  # type: ignore
        """获取详细性能统计
        
        Returns:
            包含历史数据的详细统计字典
        """
        return {
            'current': self.stats.copy(),
            'history': {
                'fps': list(self.fps_history),
                'memory_mb': list(self.memory_history),
                'cpu_percent': list(self.cpu_history),
                'latency_ms': list(self.latency_history)
            },
            'summary': {
                'total_runtime_seconds': time.time() - self.start_time if self.start_time else 0,
                'total_frames': self.stats['frame_count'],
                'total_drops': self.stats['drop_count'],
                'drop_rate': self.stats['drop_count'] / max(self.stats['frame_count'], 1)
            }
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.fps_history.clear()
        self.memory_history.clear()
        self.cpu_history.clear()
        self.latency_history.clear()
        
        self.stats = {
            'fps': 0.0,
            'avg_fps': 0.0,
            'min_fps': float('inf'),
            'max_fps': 0.0,
            'memory_mb': 0.0,
            'cpu_percent': 0.0,
            'avg_latency_ms': 0.0,
            'frame_count': 0,
            'drop_count': 0
        }
        
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        
        self.logger.info("性能统计已重置")
    
    def check_performance_warning(self) -> Optional[str]:
        """检查性能警告
        
        Returns:
            警告信息，如果没有警告则返回None
        """
        warnings = []
        
        if self.stats['fps'] > 0 and self.stats['fps'] < 5:
            warnings.append(f"FPS过低: {self.stats['fps']:.1f}")
            
        if self.stats['memory_mb'] > 400:
            warnings.append(f"内存使用过高: {self.stats['memory_mb']:.1f}MB")
            
        if self.stats['cpu_percent'] > 90:
            warnings.append(f"CPU占用过高: {self.stats['cpu_percent']:.1f}%")
            
        if len(self.latency_history) > 0 and self.stats['avg_latency_ms'] > 100:
            warnings.append(f"处理延迟过高: {self.stats['avg_latency_ms']:.1f}ms")
            
        return "; ".join(warnings) if warnings else None
    
    def print_periodic_stats(self, interval_seconds: int = 10):
        """定期打印性能统计
        
        Args:
            interval_seconds: 打印间隔（秒）
        """
        def print_stats():
            while self.is_monitoring:
                time.sleep(interval_seconds)
                stats = self.get_current_stats()
                
                self.logger.info(
                    f"性能统计 - FPS: {stats['fps']:.1f} "
                    f"(avg: {stats['avg_fps']:.1f}, min: {stats['min_fps']:.1f}, max: {stats['max_fps']:.1f}), "
                    f"内存: {stats['memory_mb']:.1f}MB, "
                    f"CPU: {stats['cpu_percent']:.1f}%, "
                    f"延迟: {stats['avg_latency_ms']:.1f}ms"
                )
                
                warning = self.check_performance_warning()
                if warning:
                    self.logger.warning(f"性能警告: {warning}")
        
        thread = threading.Thread(target=print_stats, daemon=True)
        thread.start()
        
    def estimate_optimal_parameters(self, target_fps: float = 10.0, max_memory_mb: float = 512.0) -> dict:
        """估算最优参数
        
        Args:
            target_fps: 目标FPS
            max_memory_mb: 最大内存限制
            
        Returns:
            参数建议字典
        """
        current_stats = self.get_current_stats()
        
        recommendations = {
            'cpu_overloaded': current_stats['cpu_percent'] > 80,
            'memory_overloaded': current_stats['memory_mb'] > max_memory_mb,
            'fps_too_low': current_stats['avg_fps'] < target_fps * 0.8,
            'suggestions': []
        }
        
        if recommendations['cpu_overloaded']:
            recommendations['suggestions'].append("降低图像分辨率")
            recommendations['suggestions'].append("减少处理步骤")
            
        if recommendations['memory_overloaded']:
            recommendations['suggestions'].append("减少缓冲区大小")
            recommendations['suggestions'].append("优化内存使用")
            
        if recommendations['fps_too_low']:
            recommendations['suggestions'].append("启用多线程处理")
            recommendations['suggestions'].append("使用更快的算法")
            
        return recommendations

class FrameRateController:
    """帧率控制器"""
    
    def __init__(self, target_fps: float):
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps
        self.last_frame_time = None
        
    def wait_for_frame(self) -> float:
        """等待达到目标帧率
        
        Returns:
            实际等待时间（秒）
        """
        current_time = time.time()
        
        if self.last_frame_time is None:
            self.last_frame_time = current_time
            return 0.0
            
        elapsed = current_time - self.last_frame_time
        wait_time = max(0, self.target_frame_time - elapsed)
        
        if wait_time > 0:
            time.sleep(wait_time)
            
        self.last_frame_time = time.time()
        return self.last_frame_time - current_time