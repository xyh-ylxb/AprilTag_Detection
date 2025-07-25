"""
红外相机采集模块
支持V4L2接口的红外相机，处理Y16格式数据
"""

import cv2
import numpy as np
import logging
import time
from typing import Optional, Tuple
from threading import Lock, Event
import os

class InfraredCamera:
    """红外相机采集类"""
    
    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480, fps: int = 10):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_opened = False
        self.lock = Lock()
        
        self.logger = logging.getLogger(__name__)
        
    def open(self) -> bool:
        """打开相机"""
        try:
            self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.logger.error(f"无法打开相机设备 {self.device_id}")
                return False
                
            # 设置相机参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 尝试设置Y16格式（16位灰度）
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', '1', '6', ' '))
            
            # 验证实际参数
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            # self.logger.info(f"相机已打开: {actual_width}x{actual_height}@{actual_fps}fps")
            
            self.is_opened = True
            return True
            
        except Exception as e:
            self.logger.error(f"打开相机失败: {e}")
            return False
    
    def close(self):
        """关闭相机"""
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.is_opened = False
            # self.logger.info("相机已关闭")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧图像
        
        Returns:
            np.ndarray: 16位灰度图像数据，形状为(height, width)，dtype为uint16
            None: 读取失败
        """
        if not self.is_opened or self.cap is None:
            self.logger.warning("相机未打开")
            return None
            
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.logger.warning("读取帧失败")
                return None
                
            # 处理不同格式的输入数据
            if frame.dtype == np.uint16:
                # 已经是16位格式
                infrared_frame = frame
            elif frame.dtype == np.uint8:
                if len(frame.shape) == 3:
                    # 彩色图像转换为灰度
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # 扩展到16位
                    infrared_frame = gray.astype(np.uint16) * 256
                else:
                    # 8位灰度扩展到16位
                    infrared_frame = frame.astype(np.uint16) * 256
            else:
                # 其他格式转换为16位
                infrared_frame = frame.astype(np.uint16)
                
            return infrared_frame
            
        except Exception as e:
            self.logger.error(f"读取帧异常: {e}")
            return None
    
    def get_camera_info(self) -> dict:
        """获取相机信息"""
        if not self.is_opened or self.cap is None:
            return {}
            
        try:
            info = {
                'device_id': self.device_id,
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
                'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
                'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
                'hue': self.cap.get(cv2.CAP_PROP_HUE),
                'gain': self.cap.get(cv2.CAP_PROP_GAIN),
                'exposure': self.cap.get(cv2.CAP_PROP_EXPOSURE),
                'fourcc': int(self.cap.get(cv2.CAP_PROP_FOURCC))
            }
            return info
        except Exception as e:
            self.logger.warning(f"获取相机信息失败: {e}")
            return {}
    
    def auto_detect_devices(self) -> list:
        """自动检测可用的相机设备"""
        devices = []
        # Linux系统下检查 /dev/video* 设备
        video_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
        
        for device in video_devices:
            device_path = f"/dev/{device}"
            device_id = int(device.replace('video', ''))
            
            try:
                test_cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    if ret and frame is not None:
                        devices.append({
                            'device_id': device_id,
                            'device_path': device_path,
                            'width': int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            'height': int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        })
                    test_cap.release()
            except Exception as e:
                self.logger.debug(f"检测设备 {device_id} 失败: {e}")
                
        return devices
    
    def reconnect(self) -> bool:
        """重新连接相机"""
        self.close()
        time.sleep(1)
        return self.open()
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()