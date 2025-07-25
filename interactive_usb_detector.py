#!/usr/bin/env python3
"""
交互式USB红外相机识别和检测程序
"""

import os
import sys
import time
import subprocess
from typing import List, Dict
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infrared_edge_detector import InfraredCamera

class InteractiveUSBDetector:
    def __init__(self):
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def clear_screen(self):
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def print_header(self, title: str):
        print("\n" + "="*60)
        print(f"  {title}")
        print("="*60 + "\n")
    
    def detect_usb_devices(self) -> List[Dict]:
        devices = []
        try:
            video_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
            for device in video_devices:
                device_id = int(device.replace('video', ''))
                info = self._get_device_info(device_id)
                if info:
                    devices.append(info)
        except Exception as e:
            self.logger.error(f"检测设备时出错: {e}")
        return devices
    
    def _get_device_info(self, device_id: int) -> Dict:
        info = {
            'device_id': device_id,
            'device_path': f"/dev/video{device_id}",
            'name': f"Camera {device_id}",
            'formats': [],
            'resolutions': []
        }
        
        try:
            cmd_name = f"v4l2-ctl -d {device_id} --info"
            result_name = subprocess.run(cmd_name.split(), capture_output=True, text=True)
            for line in result_name.stdout.split('\n'):
                if 'Card type' in line:
                    info['name'] = line.split(':')[1].strip()
                    
            cmd_formats = f"v4l2-ctl -d {device_id} --list-formats-ext"
            result_formats = subprocess.run(cmd_formats.split(), capture_output=True, text=True)
            
            for line in result_formats.stdout.split('\n'):
                line = line.strip()
                if 'Pixel Format' in line and "'" in line:
                    fmt = line.split("'")[1]
                    if fmt not in info['formats']:
                        info['formats'].append(fmt)
                elif 'Size' in line and 'x' in line:
                    parts = line.split()
                    for part in parts:
                        if 'x' in part and part.count('x') == 1:
                            info['resolutions'].append(part)
        except:
            pass
        return info
    
    def test_camera_connection(self, device_id: int) -> Dict:
        result = {
            'device_id': device_id,
            'connected': False,
            'width': 0,
            'height': 0,
            'fps': 0,
            'error': None
        }
        
        try:
            test_camera = InfraredCamera(device_id=device_id)
            if test_camera.open():
                info = test_camera.get_camera_info()
                result.update({
                    'connected': True,
                    'width': info.get('width', 640),
                    'height': info.get('height', 480),
                    'fps': info.get('fps', 30)
                })
                test_camera.close()
            else:
                result['error'] = "无法打开相机"
        except Exception as e:
            result['error'] = str(e)
        return result
    
    def run_detection(self):
        self.clear_screen()
        self.print_header("USB红外相机实时检测系统")
        
        print("🔍 正在扫描USB视频设备...\n")
        devices = self.detect_usb_devices()
        
        if not devices:
            print("⚠️  未检测到USB视频设备")
            print("\n可能的原因:")
            print("- 相机未连接或已损坏")
            print("- 权限不足 (尝试: sudo chmod 666 /dev/video*)")
            print("- 驱动未正确安装")
            input("\n按回车键继续...")
            return
            
        print(f"📊 发现 {len(devices)} 个USB视频设备")
        print("-" * 50)
        
        for i, device in enumerate(devices, 1):
            print(f"\n{i}. 【设备{device['device_id']}】")
            print(f"   名称: {device['name']}")
            print(f"   路径: {device['device_path']}")
            if device['formats']:
                print(f"   支持格式: {', '.join(device['formats'])}")
            if device['resolutions']:
                print(f"   支持分辨率: {len(device['resolutions'])}种")
        
        print(f"\n🔄 开始连接测试...")
        for device in devices:
            print(f"\n   测试设备 {device['device_id']} ... ", end="")
            result = self.test_camera_connection(device['device_id'])
            
            if result['connected']:
                print("✅ 成功")
                print(f"      - 分辨率: {result['width']}x{result['height']}")
                print(f"      - 帧率: {result['fps']}fps")
            else:
                print("❌ 失败")
                print(f"      - 错误: {result['error']}")
        
        print("\n" + "="*60)
        print("💡 下一步操作:")
        print("1. 运行: python main.py --device <设备ID>")
        print("2. 测试: python main.py --test")
        print("3. 配置: 编辑 config/infrared_config.yaml")
        
        device_choice = input("\n输入设备ID直接启动边缘检测(回车跳过): ").strip()
        if device_choice.isdigit():
            device_id = int(device_choice)
            if any(d['device_id'] == device_id for d in devices):
                print(f"\n🚀 启动设备{device_id}的边缘检测...")
                time.sleep(1)
                self._run_edge_detection(device_id)
    
    def _run_edge_detection(self, device_id: int):
        """运行边缘检测"""
        from main import InfraredEdgeDetectorApp
        app = InfraredEdgeDetectorApp()
        app.camera.device_id = device_id
        app.run_single_threaded()
    
    def run_menu(self):
        """运行主菜单"""
        while True:
            self.clear_screen()
            self.print_header("红外相机USB设备识别系统")
            
            print("请选择操作:")
            print("1. 🔍 扫描并测试USB相机设备")
            print("2. 📊 查看所有视频设备详情")
            print("3. 🎯 直接启动边缘检测")
            print("4. ❌ 退出")
            
            choice = input("\n请输入选项 (1-4): ").strip()
            
            if choice == '1':
                self.run_detection()
            elif choice == '2':
                self.show_all_devices()
            elif choice == '3':
                device_id = input("请输入设备ID(默认为0): ").strip()
                try:
                    device_id = int(device_id) if device_id else 0
                    self._run_edge_detection(device_id)
                except ValueError:
                    print("❌ 无效的设备ID")
                    input("按回车键继续...")
            elif choice == '4':
                print("👋 感谢使用！")
                break
            else:
                print("❌ 无效选项")
                time.sleep(1)
    
    def show_all_devices(self):
        """显示所有设备详情"""
        self.clear_screen()
        self.print_header("系统视频设备详情")
        
        devices = self.detect_usb_devices()
        if devices:
            for device in devices:
                print(f"\n设备 {device['device_id']}:")
                print(f"  名称: {device['name']}")
                print(f"  路径: {device['device_path']}")
                if device['formats']:
                    print(f"  格式: {', '.join(device['formats'])}")
                if device['resolutions']:
                    print(f"  分辨率: {', '.join(device['resolutions'][:5])}")
                    if len(device['resolutions']) > 5:
                        print(f"  ... 共{len(device['resolutions'])}种")
        else:
            print("未检测到视频设备")
        
        input("\n按回车键返回...")

def main():
    logging.basicConfig(level=logging.INFO)
    detector = InteractiveUSBDetector()
    
    try:
        detector.run_menu()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()