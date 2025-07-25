"""
多线程管理模块
实现生产者-消费者模式，分离采集、处理、显示线程
"""

import threading
import queue
import time
import logging
import numpy as np
from typing import Optional, Callable, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import psutil

@dataclass
class ThreadConfig:
    """线程配置"""
    max_queue_size: int = 10
    camera_thread_priority: int = 10
    processing_thread_count: int = 2
    display_thread_priority: int = 5
    buffer_size: int = 3

class FrameData:
    """帧数据结构"""
    def __init__(self, frame: np.ndarray, timestamp: float, frame_id: int, metadata: Optional[dict] = None):
        self.frame = frame
        self.timestamp = timestamp
        self.frame_id = frame_id
        self.metadata = metadata or {}
        self.processing_start_time = None
        self.processing_end_time = None

class ThreadingManager:
    """多线程管理器"""
    
    def __init__(self, config: ThreadConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 线程安全队列
        self.capture_queue = queue.Queue(maxsize=config.max_queue_size)
        self.processing_queue = queue.Queue(maxsize=config.max_queue_size)
        self.display_queue = queue.Queue(maxsize=config.max_queue_size)
        
        # 线程控制
        self.running = False
        self.threads = []
        self.executor = ThreadPoolExecutor(max_workers=config.processing_thread_count + 3)
        
        # 同步锁
        self.lock = threading.Lock()
        self.frame_id_counter = 0
        
        # 性能监控
        self.dropped_frames = 0
        self.processed_frames = 0
        
    def start(self):
        """启动所有线程"""
        if self.running:
            return
            
        self.running = True
        self.threads.clear()
        
        # 启动采集线程
        self.threads.append(threading.Thread(
            target=self._capture_thread,
            name="CameraThread",
            daemon=True
        ))
        
        # 启动处理线程
        for i in range(self.config.processing_thread_count):
            self.threads.append(threading.Thread(
                target=self._processing_thread,
                name=f"ProcessingThread-{i}",
                daemon=True
            ))
        
        # 启动显示线程
        self.threads.append(threading.Thread(
            target=self._display_thread,
            name="DisplayThread",
            daemon=True
        ))
        
        for thread in self.threads:
            thread.start()
            
        self.logger.info("多线程系统已启动")
        
    def stop(self):
        """停止所有线程"""
        self.running = False
        
        # 清空队列信号
        for q in [self.capture_queue, self.processing_queue, self.display_queue]:
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
                
        # 等待线程结束
        for thread in self.threads:
            thread.join(timeout=1.0)
            
        self.executor.shutdown(wait=True)
        self.logger.info("多线程系统已停止")
        
    def _capture_thread(self):
        """相机采集线程"""
        while self.running:
            try:
                # 采集新帧
                frame = self._capture_frame()
                if frame is not None:
                    
                    with self.lock:
                        frame_id = self.frame_id_counter
                        self.frame_id_counter += 1
                    
                    frame_data = FrameData(
                        frame=frame,
                        timestamp=time.time(),
                        frame_id=frame_id
                    )
                    
                    # 放入处理队列
                    try:
                        self.processing_queue.put(frame_data, timeout=0.1)
                    except queue.Full:
                        self.dropped_frames += 1
                        self.logger.warning("处理队列已满，丢弃帧")
                        
            except Exception as e:
                self.logger.error(f"采集线程错误: {e}")
                time.sleep(0.1)
                
    def _processing_thread(self):
        """图像处理线程"""
        while self.running:
            try:
                # 获取待处理帧
                frame_data = self.processing_queue.get(timeout=0.1)
                frame_data.processing_start_time = time.time()
                
                # 处理图像
                processed_frame = self._process_frame(frame_data.frame)
                
                if processed_frame is not None:
                    frame_data.processing_end_time = time.time()
                    frame_data.frame = processed_frame
                    
                    # 放入显示队列
                    try:
                        self.display_queue.put(frame_data, timeout=0.1)
                    except queue.Full:
                        self.logger.warning("显示队列已满，丢弃帧")
                        
                self.processed_frames += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"处理线程错误: {e}")
                time.sleep(0.1)
                
    def _display_thread(self):
        """显示线程"""
        while self.running:
            try:
                frame_data = self.display_queue.get(timeout=0.1)
                
                # 显示处理结果
                self._display_frame(frame_data)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"显示线程错误: {e}")
                time.sleep(0.1)
                
    def set_capture_callback(self, callback: Callable[[], Optional[np.ndarray]]):
        """设置采集回调"""
        self._capture_frame = callback
        
    def set_processing_callback(self, callback: Callable[[np.ndarray], Optional[np.ndarray]]):
        """设置处理回调"""
        self._process_frame = callback
        
    def set_display_callback(self, callback: Callable[[FrameData], None]):
        """设置显示回调"""
        self._display_frame = callback
        
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        with self.lock:
            return {
                'dropped_frames': self.dropped_frames,
                'processed_frames': self.processed_frames,
                'queue_sizes': {
                    'capture': self.capture_queue.qsize(),
                    'processing': self.processing_queue.qsize(),
                    'display': self.display_queue.qsize()
                },
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'thread_count': len(self.threads)
            }

class MemoryPool:
    """内存池管理器"""
    
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.pool = []
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
    def get_buffer(self, shape: tuple, dtype=np.uint8) -> np.ndarray:
        """从内存池获取缓冲区"""
        with self.lock:
            for i, buffer in enumerate(self.pool):
                if buffer.shape == shape and buffer.dtype == dtype:
                    return self.pool.pop(i)
            
            # 创建新缓冲区
            if len(self.pool) >= self.max_size:
                self.logger.warning("内存池已满，创建新缓冲区")
            return np.zeros(shape, dtype=dtype)
    
    def return_buffer(self, buffer: np.ndarray):
        """返回缓冲区到内存池"""
        if buffer is None:
            return
            
        with self.lock:
            if len(self.pool) < self.max_size:
                self.pool.append(buffer)
            else:
                # 清理最旧的缓冲区
                if self.pool:
                    self.pool.pop(0)
    
    def clear_pool(self):
        """清空内存池"""
        with self.lock:
            self.pool.clear()
        self.logger.info("内存池已清空")
    
    def get_pool_stats(self) -> Dict:
        """获取内存池统计"""
        with self.lock:
            return {
                'pool_size': len(self.pool),
                'max_size': self.max_size,
                'total_memory_mb': sum(buf.nbytes for buf in self.pool) / 1024 / 1024
            }

# 全局内存池实例
memory_pool = MemoryPool(max_size=30)