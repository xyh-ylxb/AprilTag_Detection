#!/usr/bin/env python3
"""
基础功能测试脚本
用于验证各个模块的基本功能
"""

import numpy as np
import cv2
import logging
from infrared_edge_detector import InfraredProcessor, EdgeDetector, SystemConfig

def test_infrared_processing():
    """测试红外处理模块"""
    print("测试红外图像处理模块...")
    
    processor = InfraredProcessor()
    
    # 创建模拟红外数据
    mock_infrared = np.random.randint(10000, 40000, (480, 640), dtype=np.uint16)
    
    # 测试温度转换
    visual_8bit = processor.convert_temperature_to_8bit(mock_infrared)
    print(f"温度转换完成: {mock_infrared.shape} -> {visual_8bit.shape}")
    
    # 测试完整处理流程
    config = SystemConfig.load_from_file('./config/infrared_config.yaml')
    processed = processor.process_frame(mock_infrared, config={
        'clahe_clip_limit': config.processing.clahe_clip_limit,
        'clahe_tile_size': tuple(config.processing.clahe_tile_size),
        'temporal_filter_size': config.processing.temporal_filter_size
    })
    
    if processed and 'enhanced' in processed:
        print("处理流程测试成功")
        return True
    else:
        print("处理流程测试失败")
        return False

def test_edge_detection():
    """测试边缘检测模块"""
    print("测试边缘检测模块...")
    
    detector = EdgeDetector()
    
    # 创建测试图像
    test_image = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(test_image, (50, 50), (150, 150), 255, -1)
    
    # 测试各种算法
    algorithms = ['canny', 'sobel', 'laplacian', 'adaptive']
    
    success_count = 0
    for algo in algorithms:
        try:
            result = detector.detect_edges(test_image, method=algo)
            if result and result['edges'] is not None:
                print(f"算法 {algo} 测试成功")
                success_count += 1
            else:
                print(f"算法 {algo} 测试失败")
        except Exception as e:
            print(f"算法 {algo} 异常: {e}")
    
    return success_count == len(algorithms)

def test_temperature_gradient():
    """测试温度梯度边缘检测"""
    print("测试温度梯度边缘检测...")
    
    detector = EdgeDetector()
    
    # 创建模拟温度数据
    temp_data = np.zeros((100, 100), dtype=np.uint16)
    temp_data[25:75, 25:75] = 500  # 中间区域温度较高
    temp_data[40:60, 40:60] = 1000  # 核心区域更高
    
    try:
        edges = detector.detect_temperature_gradient(temp_data, threshold=50)
        if edges is not None:
            edge_pixels = np.sum(edges > 0)
            print(f"温度梯度边缘检测成功，边缘像素数: {edge_pixels}")
            return edge_pixels > 0
    except Exception as e:
        print(f"温度梯度检测异常: {e}")
    
    return False

def run_all_tests():
    """运行所有测试"""
    print("开始红外边缘检测系统基础功能测试...\n")
    
    tests = [
        test_infrared_processing,
        test_edge_detection,
        test_temperature_gradient
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"测试异常: {e}")
    
    print(f"\n测试完成: {passed}/{total} 项通过")
    return passed == total

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = run_all_tests()
    exit(0 if success else 1)