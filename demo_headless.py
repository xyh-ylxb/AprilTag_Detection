#!/usr/bin/env python3
"""
无头红外边缘检测系统演示
不依赖显示环境
"""

import numpy as np
import time
import os
import cv2
from infrared_edge_detector import InfraredProcessor, EdgeDetector, PerformanceMonitor

def create_mock_infrared_pattern():
    """创建模拟红外图案"""
    height, width = 480, 640
    
    # 创建基础温度背景
    base_temp = np.random.normal(20000, 1000, (height, width)).astype(np.uint16)
    
    # 添加热源物体
    y, x = np.ogrid[:height, :width]
    
    # 多个热源区域
    for i in range(5):
        center_y = np.random.randint(100, height-100)
        center_x = np.random.randint(100, width-100)
        radius = np.random.randint(30, 80)
        
        circle = ((y - center_y)**2 + (x - center_x)**2) < radius**2
        base_temp[circle] += np.random.randint(5000, 15000)
    
    return base_temp

def run_headless_demo(duration_seconds=10):
    """运行无头演示"""
    print("红外边缘检测系统无头演示")
    print("=" * 50)
    print(f"运行时长: {duration_seconds}秒")
    
    processor = InfraredProcessor()
    detector = EdgeDetector()
    monitor = PerformanceMonitor()
    
    processor.setup_clahe(clip_limit=2.0, tile_size=(8, 8))
    processor.setup_temporal_filter(buffer_size=3)
    monitor.start_monitoring()
    
    output_dir = "./demo_output"
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration_seconds:
            frame_start = time.time()
            
            # 生成模拟红外数据
            infrared_frame = create_mock_infrared_pattern()
            
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
            
            # 应用多种边缘检测算法
            algorithms = ['canny', 'sobel', 'laplacian', 'temperature']
            results = {}
            
            for algo in algorithms:
                if algo == 'temperature':
                    edges = detector.detect_temperature_gradient(
                        infrared_frame, threshold=2000
                    )
                else:
                    edges = detector.detect_edges(
                        enhanced, method=algo,
                        low_threshold=50, high_threshold=150
                    )['edges']
                
                results[algo] = edges
            
            # 性能记录
            monitor.record_frame()
            latency_ms = (time.time() - frame_start) * 1000
            monitor.record_latency(latency_ms)
            
            frame_count += 1
            
            # 每5帧保存一次结果
            if frame_count % 5 == 0:
                timestamp = time.strftime("%H%M%S")
                
                # 保存原始和处理后的图像
                cv2.imwrite(f"{output_dir}/frame_{frame_count:04d}_original.jpg", processed['visual_8bit'])
                cv2.imwrite(f"{output_dir}/frame_{frame_count:04d}_enhanced.jpg", enhanced)
                
                # 保存边缘检测结果
                for algo, edges in results.items():
                    if edges is not None:
                        cv2.imwrite(f"{output_dir}/frame_{frame_count:04d}_{algo}.jpg", edges)
                
                # 创建组合图像
                combined_height = max(processed['visual_8bit'].shape[0], 
                                    enhanced.shape[0],
                                    *[edges.shape[0] for edges in results.values() if edges is not None])
                
                combined_width = sum([
                    processed['visual_8bit'].shape[1],
                    enhanced.shape[1],
                    max([edges.shape[1] for edges in results.values() if edges is not None])
                ])
                
                combined = np.zeros((combined_height, combined_width), dtype=np.uint8)
                
                # 组合图像
                x_offset = 0
                combined[:processed['visual_8bit'].shape[0], x_offset:x_offset+processed['visual_8bit'].shape[1]] = processed['visual_8bit']
                x_offset += processed['visual_8bit'].shape[1]
                
                combined[:enhanced.shape[0], x_offset:x_offset+enhanced.shape[1]] = enhanced
                x_offset += enhanced.shape[1]
                
                # 添加边缘检测结果
                edge_samples = [edges for edges in results.values() if edges is not None]
                if edge_samples:
                    max_edge_width = max([edges.shape[1] for edges in edge_samples])
                    for edges in edge_samples:
                        if edges.shape[1] <= max_edge_width:
                            combined[:edges.shape[0], x_offset:x_offset+edges.shape[1]] = edges
                            break
                
                cv2.imwrite(f"{output_dir}/frame_{frame_count:04d}_combined.jpg", combined)
                
                # 显示进度
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"处理第 {frame_count} 帧 - FPS: {fps:.1f} - 已运行: {elapsed:.1f}s")
            
            # 控制帧率
            time.sleep(0.1)  # 约10fps
            
    except KeyboardInterrupt:
        print("\n演示被中断")
    
    finally:
        monitor.stop_monitoring()
        
        # 输出最终统计
        final_stats = monitor.get_current_stats()
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        
        print(f"\n演示完成!")
        print(f"总帧数: {frame_count}")
        print(f"运行时间: {elapsed:.1f}秒")
        print(f"平均FPS: {fps:.1f}")
        print(f"平均延迟: {final_stats['avg_latency_ms']:.1f}ms")
        print(f"输出目录: {output_dir}")
        print(f"已生成图像: {len(os.listdir(output_dir))} 张")

if __name__ == "__main__":
    run_headless_demo(duration_seconds=5)