#!/usr/bin/env python3
"""
多线程红外相机边缘检测系统
实现生产者-消费者模式的完整应用
"""

import logging
import argparse
import time
import cv2
import numpy as np
from typing import Optional

from infrared_edge_detector import (
    InfraredCamera, InfraredProcessor, EdgeDetector, 
    PerformanceMonitor, SystemConfig
)
from infrared_edge_detector.threading_manager import ThreadingManager, ThreadConfig, memory_pool

class MultithreadInfraredApp:
    """多线程红外边缘检测应用"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = SystemConfig.load_from_file(config_path or "./config/infrared_config.yaml")
        self.running = False
        
        # 初始化组件
        self.camera = InfraredCamera(
            device_id=self.config.camera.device_id,
            width=self.config.camera.width,
            height=self.config.camera.height,
            fps=self.config.camera.fps
        )
        
        self.processor = InfraredProcessor()
        self.detector = EdgeDetector()
        self.monitor = PerformanceMonitor()
        
        # 多线程管理器
        self.thread_config = ThreadConfig(
            max_queue_size=8,
            processing_thread_count=2,
            buffer_size=3
        )
        self.thread_manager = ThreadingManager(self.thread_config)
        
        # 设置日志
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 设置回调
        self._setup_callbacks()
        
    def _setup_callbacks(self):
        """设置多线程回调"""
        self.thread_manager.set_capture_callback(self._capture_frame)
        self.thread_manager.set_processing_callback(self._process_frame)
        self.thread_manager.set_display_callback(self._display_frame)
        
    def _capture_frame(self) -> Optional[np.ndarray]:
        """采集线程回调"""
        if not self.camera.is_opened:
            if not self.camera.open():
                self.logger.error("无法打开相机")
                return None
                
        frame = self.camera.read_frame()
        if frame is not None:
            self.monitor.record_frame(frame.shape)
        else:
            self.monitor.record_frame_drop()
            
        return frame
        
    def _process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """处理线程回调"""
        try:
            # 从内存池获取缓冲区
            processed = self.processor.process_frame(frame, config={
                'clahe_clip_limit': self.config.processing.clahe_clip_limit,
                'clahe_tile_size': tuple(self.config.processing.clahe_tile_size),
                'temporal_filter_size': self.config.processing.temporal_filter_size,
                'gaussian_kernel': tuple(self.config.processing.gaussian_kernel)
            })
            
            if processed and processed['enhanced'] is not None:
                # 边缘检测
                edges = self.detector.detect_edges(
                    processed['enhanced'],
                    method=self.config.edge_detection.algorithm,
                    low_threshold=self.config.edge_detection.canny_low_threshold,
                    high_threshold=self.config.edge_detection.canny_high_threshold
                )
                
                if edges and edges['edges'] is not None:
                    # 创建显示结果
                    overlay = self.detector.apply_edge_overlay(
                        processed['enhanced'], edges['edges'], color=(0, 255, 0), alpha=0.7
                    )
                    
                    # 添加性能信息
                    stats = self.thread_manager.get_performance_stats()
                    fps_text = f"FPS: {stats['processed_frames'] / max(time.time() - self.start_time, 1):.1f}"
                    cv2.putText(overlay, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    return overlay
            
            return None
            
        except Exception as e:
            self.logger.error(f"处理错误: {e}")
            return None
            
    def _display_frame(self, frame_data):
        """显示线程回调"""
        try:
            if frame_data.frame is not None:
                cv2.imshow("红外边缘检测 - 多线程", frame_data.frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop()
                elif key == ord('s'):
                    self.save_frame(frame_data.frame)
                    
        except Exception as e:
            self.logger.error(f"显示错误: {e}")
            
    def save_frame(self, frame):
        """保存当前帧"""
        import cv2
        import time
        import os
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"multithread_frame_{timestamp}.jpg"
        output_path = os.path.join(self.config.output_dir, filename)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        cv2.imwrite(output_path, frame)
        self.logger.info(f"已保存帧: {output_path}")
        
    def setup_processing_parameters(self):
        """设置处理参数"""
        self.processor.setup_clahe(
            clip_limit=self.config.processing.clahe_clip_limit,
            tile_size=tuple(self.config.processing.clahe_tile_size)
        )
        
        self.processor.setup_temporal_filter(
            buffer_size=self.config.processing.temporal_filter_size
        )
        
    def run(self):
        """运行多线程应用"""
        self.logger.info("启动多线程红外边缘检测系统")
        
        try:
            # 设置处理参数
            self.setup_processing_parameters()
            
            # 启动性能监控
            self.monitor.start_monitoring()
            
            # 启动多线程系统
            self.start_time = time.time()
            self.thread_manager.start()
            
            # 主循环
            self.running = True
            while self.running:
                time.sleep(0.1)
                
                # 显示性能统计
                if int(time.time() - self.start_time) % 10 == 0:
                    stats = self.thread_manager.get_performance_stats()
                    self.logger.info(
                        f"性能: 处理{stats['processed_frames']}帧, "
                        f"丢弃{stats['dropped_frames']}帧, "
                        f"内存{stats['memory_usage_mb']:.1f}MB"
                    )
                    
        except KeyboardInterrupt:
            self.logger.info("用户中断")
        except Exception as e:
            self.logger.error(f"运行错误: {e}", exc_info=True)
        finally:
            self.stop()
            
    def stop(self):
        """停止应用"""
        self.running = False
        self.thread_manager.stop()
        self.monitor.stop_monitoring()
        
        # 清理内存池
        memory_pool.clear_pool()
        
        # 关闭相机
        self.camera.close()
        
        # 关闭窗口
        import cv2
        cv2.destroyAllWindows()
        
        # 最终统计
        final_stats = self.thread_manager.get_performance_stats()
        self.logger.info(
            f"运行结束: 共处理{final_stats['processed_frames']}帧, "
            f"内存使用{final_stats['memory_usage_mb']:.1f}MB"
        )

def main():
    parser = argparse