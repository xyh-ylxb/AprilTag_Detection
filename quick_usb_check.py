#!/usr/bin/env python3
"""
USB红外相机快速检测脚本
一键检测设备状态
"""

import os
import subprocess
from infrared_edge_detector import InfraredCamera

def quick_usb_check():
    """快速USB相机检测"""
    print("🔍 USB红外相机快速检测")
    print("=" * 40)
    
    # 检测设备
    video_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
    
    if not video_devices:
        print("❌ 未检测到USB视频设备")
        print("\n解决方案:")
        print("1. 检查相机是否正确连接")
        print("2. 运行: lsusb")
        print("3. 检查权限: sudo chmod 666 /dev/video*")
        return
    
    print(f"✅ 找到 {len(video_devices)} 个USB视频设备")
    
    for device in video_devices:
        device_id = int(device.replace('video', ''))
        print(f"\n设备 {device_id} (/dev/video{device_id}):")
        
        try:
            # 获取设备名称
            cmd = f"v4l2-ctl -d {device_id} --info"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'Card type' in line:
                    print(f"  名称: {line.split(':')[1].strip()}")
                    break
            
            # 测试连接
            camera = InfraredCamera(device_id=device_id)
            if camera.open():
                info = camera.get_camera_info()
                print(f"  状态: ✅ 正常")
                print(f"  分辨率: {info.get('width', '未知')}x{info.get('height', '未知')}")
                print(f"  帧率: {info.get('fps', '未知')}fps")
                camera.close()
            else:
                print(f"  状态: ❌ 无法打开")
                
        except Exception as e:
            print(f"  状态: ❌ 错误 - {e}")
    
    print(f"\n{'='*40}")
    print("📋 使用命令:")
    print(f"python main.py --device 0  # 使用设备0")
    print(f"python main.py --test      # 测试所有设备")
    print(f"python interactive_usb_detector.py  # 交互式检测")

if __name__ == "__main__":
    quick_usb_check()