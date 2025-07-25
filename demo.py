#!/usr/bin/env python3
"""
红外边缘检测系统演示脚本
使用模拟数据进行演示
"""

import numpy as np
import cv2
import time
from infrared_edge_detector import InfraredProcessor, EdgeDetector

def create_mock_infrared_pattern():
    """创建模拟红外图案"""
    height, width = 480, 640
    
    # 创建基础温度背景
    base_temp = np.random.normal(20000, 1000, (height, width)).astype(np.uint16)
    
    # 添加热源物体（模拟人体或设备）
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height//2, width//2
    
    # 圆形热源
    circle1 = ((y - center_y + 50)**2 + (x - center_x - 100)**2) < 2000
    circle2 = ((y - center_y - 100)**2 + (x - center_x + 100)**2) < 1500
    
    base_temp[circle1] += 15000
    base_temp[circle2] += 10000
    
    # 添加边缘效应（模拟温度梯度）
    gradient_x = np.abs(x - center_x) / width
    gradient_y = np.abs(y - center_y) / height
    edge_effect = (gradient_x + gradient_y) * 5000
    
    mock_data = base_temp + edge_effect.astype(np.uint16)
    
    return mock_data

def run_demo():
    """运行演示"""
    print("红外边缘检测系统演示")
    print("=" * 50)
    
    processor = InfraredProcessor()
    detector = EdgeDetector()
    
    processor.setup_clahe(clip_limit=2.0, tile_size=(8, 8))
    processor.setup_temporal_filter(buffer_size=3)
    
    # 创建窗口
    window_names = ['原始红外', '处理后', 'Canny边缘', '温度梯度边缘', '叠加显示']
    for name in window_names:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # 生成模拟红外数据
            infrared_frame = create_mock_infrared_pattern()
            
            # 添加一些变化来模拟动态场景
            if frame_count % 30 == 0:
                infrared_frame += np.random.randint(-1000, 1000, infrared_frame.shape, dtype=np.uint16)
                infrared_frame = np.clip(infrared_frame, 5000, 60000)
            
            # 处理红外图像
            processed = processor.process_frame(infrared_frame, config={
                'clahe_clip_limit': 2.0,
                'clahe_tile_size': (8, 8),
                'temporal_filter_size': 3,
                'gaussian_kernel': (5, 5)
            })
            
            if processed is None:
                continue
            
            enhanced = processed['enhanced']
            
            # 应用不同的边缘检测算法
            canny_edges = detector.detect_edges(
                enhanced, method='canny',
                low_threshold=50, high_threshold=150
            )['edges']
            
            temp_edges = detector.detect_temperature_gradient(
                infrared_frame, threshold=2000
            )
            
            # 创建叠加显示
            overlay = detector.apply_edge_overlay(
                enhanced, canny_edges, color=(0, 255, 0), alpha=0.7
            )
            
            # 温度信息显示
            if processed['temperature_range']:
                temp_range = processed['temperature_range']
                temp_text = f"温度范围: {temp_range['min']:.1f} - {temp_range['max']:.1f}°C"
                cv2.putText(overlay, temp_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 计算FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(overlay, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示结果
            cv2.imshow('原始红外', processed['visual_8bit'])
            cv2.imshow('处理后', enhanced)
            cv2.imshow('Canny边缘', canny_edges)
            cv2.imshow('温度梯度边缘', temp_edges)
            cv2.imshow('叠加显示', overlay)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f'demo_frame_{timestamp}.jpg', overlay)
                print(f"已保存帧: demo_frame_{timestamp}.jpg")
    
    except KeyboardInterrupt:
        print("\n演示已中断")
    
    finally:
        cv2.destroyAllWindows()
        print(f"\n演示结束 - 总帧数: {frame_count}, 平均FPS: {fps:.1f}")

if __name__ == "__main__":
    run_demo()