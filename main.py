#!/usr/bin/env python3
"""
红外相机边缘检测系统主程序
"""

import logging
import argparse
import os
import sys
import signal
import time
from typing import Optional

from infrared_edge_detector import (
    InfraredCamera, InfraredProcessor, EdgeDetector, PerformanceMonitor, SystemConfig
)

class InfraredEdgeDetectorApp:
    """主应用程序"""
    
    def __init__(self, config_path: Optional[str] = None, no_gui: bool = False):
        self.config = SystemConfig.load_from_file(config_path or "./config/infrared_config.yaml")
        self.running = False
        self.no_gui = no_gui
        
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
        
        # 设置日志
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info("收到退出信号，正在关闭系统...")
        self.running = False
    
    def setup_processing_parameters(self):
        """设置处理参数"""
        self.processor.setup_clahe(
            clip_limit=self.config.processing.clahe_clip_limit,
            tile_size=self.config.processing.clahe_tile_size
        )
        
        self.processor.setup_temporal_filter(
            buffer_size=self.config.processing.temporal_filter_size
        )
    
    def run_single_threaded(self):
        """单线程运行模式"""
        self.logger.info("启动单线程模式")
        self.running = True
        
        try:
            self.camera.open()
            self.setup_processing_parameters()
            self.monitor.start_monitoring()
            
            self.logger.info("系统启动完成")
            
            frame_count = 0
            
            while self.running:
                start_time = time.time()
                
                # 采集图像
                infrared_frame = self.camera.read_frame()
                if infrared_frame is None:
                    self.logger.warning("无法读取图像，重试中...")
                    time.sleep(0.1)
                    self.monitor.record_frame_drop()
                    continue
                
                # 新增：打印帧的类型、shape、dtype、最小值、最大值
                import numpy as np
                self.logger.info("即将打印帧信息")
                if isinstance(infrared_frame, np.ndarray):
                    self.logger.debug(
                        f"Frame type: {type(infrared_frame)}, shape: {infrared_frame.shape}, "
                        f"dtype: {infrared_frame.dtype}, min: {infrared_frame.min()}, max: {infrared_frame.max()}"
                    )
                else:
                    self.logger.debug(f"Frame type: {type(infrared_frame)}")
                self.logger.info("帧信息打印完毕")
                
                frame_count += 1
                
                # 处理图像
                processed = self.processor.process_frame(
                    infrared_frame,
                    config={
                        'clahe_clip_limit': self.config.processing.clahe_clip_limit,
                        'clahe_tile_size': self.config.processing.clahe_tile_size,
                        'temporal_filter_size': self.config.processing.temporal_filter_size,
                        'gaussian_kernel': tuple(self.config.processing.gaussian_kernel)
                    }
                )
                
                if processed is None or processed['enhanced'] is None:
                    self.logger.warning("图像处理失败")
                    continue
                
                # 边缘检测
                edges = self.detector.detect_edges(
                    processed['enhanced'],
                    method=self.config.edge_detection.algorithm,
                    low_threshold=self.config.edge_detection.canny_low_threshold,
                    high_threshold=self.config.edge_detection.canny_high_threshold,
                    kernel_size=self.config.edge_detection.sobel_kernel
                )
                
                # 性能记录
                latency_ms = (time.time() - start_time) * 1000
                self.monitor.record_frame()
                self.monitor.record_latency(latency_ms)
                
                # 显示结果
                self.display_results(processed, edges)
                
                # 检查性能警告
                warning = self.monitor.check_performance_warning()
                if warning:
                    self.logger.warning(warning)
                
                # 控制帧率
                target_frame_time = 1.0 / self.config.camera.fps
                actual_time = time.time() - start_time
                if actual_time < target_frame_time:
                    time.sleep(target_frame_time - actual_time)
                
        except KeyboardInterrupt:
            self.logger.info("用户中断")
        except Exception as e:
            self.logger.error(f"运行错误: {e}", exc_info=True)
        finally:
            self.cleanup()
    
    def display_results(self, processed: dict, edges: dict):
        if getattr(self, 'no_gui', False):
            self.logger.debug("GUI已关闭，跳过显示。")
            return
        import cv2
        import numpy as np
        
        if edges and edges['edges'] is not None:
            # 创建显示图像
            enhanced = processed['enhanced']
            edges_frame = edges['edges']
            
            # # 裁剪ROI（四周各去掉5%）
            # h, w = enhanced.shape
            # margin_h = int(h * 0.05)
            # margin_w = int(w * 0.05)
            # roi = (margin_w, margin_h, w - margin_w, h - margin_h)
            # # 裁剪增强图像和边缘图像
            # enhanced_roi = enhanced[roi[1]:roi[3], roi[0]:roi[2]]
            # edges_frame_roi = edges_frame[roi[1]:roi[3], roi[0]:roi[2]]
            
            # # 边缘叠加（先用原图，后面只画ROI内直线）
            # overlay = self.detector.apply_edge_overlay(
            #     enhanced, edges_frame, color=(0, 255,0), alpha=0.7
            # )
            
            # 曲线特征检测：Canny + findContours
            import cv2
            contours, _ = cv2.findContours(edges_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)  # 绿色显示所有曲线
            self.logger.info(f"检测到曲线（轮廓）数量: {len(contours)}")
            
            # # Hough直线检测（如需恢复可取消注释）
            # lines = cv2.HoughLinesP(
            #     edges_frame_roi,
            #     rho=1,
            #     theta=np.pi / 180,
            #     threshold=self.config.edge_detection.hough_threshold,
            #     minLineLength=self.config.edge_detection.hough_min_line_length,
            #     maxLineGap=self.config.edge_detection.hough_max_line_gap
            # )
            # 只保留最长的那根直线
            # if lines is not None and len(lines) > 0:
            #     longest_line = None
            #     max_length = 0
            #     for line in lines:
            #         x1, y1, x2, y2 = line[0]
            #         length = np.hypot(x2 - x1, y2 - y1)
            #         if length > max_length:
            #             max_length = length
            #             longest_line = (x1, y1, x2, y2)
            #     if longest_line is not None:
            #         x1, y1, x2, y2 = longest_line
            #         # 坐标映射回原图
            #         x1_full, y1_full = x1 + margin_w, y1 + margin_h
            #         x2_full, y2_full = x2 + margin_w, y2 + margin_h
            #         cv2.line(overlay, (x1_full, y1_full), (x2_full, y2_full), (0, 0, 255), 3)
            #         self.logger.info(f"最长直线坐标(ROI映射回原图): ({x1_full}, {y1_full}) -> ({x2_full}, {y2_full}), 长度: {max_length:.1f}")
            
            # 显示信息
            stats = self.monitor.get_current_stats()
            info_text = f"FPS: {stats['fps']:.1f} | Latency: {stats['avg_latency_ms']:.1f}ms"
            cv2.putText(overlay, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Infrared Edge Detection", overlay)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('s'):
                self.save_frame(overlay)
    
    def save_frame(self, frame):
        """保存当前帧"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}.jpg"
        output_path = os.path.join(self.config.output_dir, filename)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        import cv2
        cv2.imwrite(output_path, frame)
        self.logger.info(f"已保存帧到: {output_path}")
    
    def test_camera_connection(self):
        """测试相机连接"""
        self.logger.info("开始相机连接测试...")
        
        devices = self.camera.auto_detect_devices()
        if not devices:
            self.logger.error("未检测到相机设备")
            return False
        
        self.logger.info(f"检测到 {len(devices)} 个相机设备:")
        for device in devices:
            self.logger.info(f"  - Device {device['device_id']}: {device['width']}x{device['height']}")
        
        if self.camera.open():
            info = self.camera.get_camera_info()
            self.logger.info(f"相机信息: {info}")
            self.camera.close()
            return True
        else:
            self.logger.error("无法打开相机")
            return False
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("正在清理资源...")
        self.monitor.stop_monitoring()
        self.camera.close()
        
        # 保存最终统计
        final_stats = self.monitor.get_detailed_stats()
        self.logger.info("最终性能统计:")
        self.logger.info(f"总帧数: {final_stats['summary']['total_frames']}")
        self.logger.info(f"运行时间: {final_stats['summary']['total_runtime_seconds']:.1f}秒")
        self.logger.info(f"平均FPS: {final_stats['current']['avg_fps']:.1f}")
        self.logger.info(f"丢帧率: {final_stats['summary']['drop_rate']:.2%}")
        
        import cv2
        cv2.destroyAllWindows()
        
        self.logger.info("系统已关闭")

def main():
    parser = argparse.ArgumentParser(description='红外相机边缘检测系统')
    parser.add_argument('-c', '--config', type=str, default='./config/infrared_config.yaml',
                        help='配置文件路径')
    parser.add_argument('-t', '--test', action='store_true',
                        help='测试相机连接')
    parser.add_argument('-d', '--device', type=int, default=0,
                        help='相机设备ID')
    parser.add_argument('-w', '--width', type=int, default=640,
                        help='图像宽度')
    parser.add_argument('-H', '--height', type=int, default=480,
                        help='图像高度')
    parser.add_argument('-f', '--fps', type=int, default=10,
                        help='目标帧率')
    parser.add_argument('--no-gui', action='store_true', help='不启动可视化界面')
    
    args = parser.parse_args()
    
    # 创建应用实例
    app = InfraredEdgeDetectorApp(config_path=args.config, no_gui=args.no_gui)
    
    if args.test:
        success = app.test_camera_connection()
        sys.exit(0 if success else 1)
    else:
        app.run_single_threaded()

if __name__ == "__main__":
    main()