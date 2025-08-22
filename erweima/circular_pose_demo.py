#!/usr/bin/env python3
"""
机器人位姿估计完整演示程序
集成现有检测程序，实现实时位姿估计
"""

import cv2
import numpy as np
import json
import math
import os
import sys
import time
from typing import List, Tuple, Dict, Optional

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from infrared_edge_detector.camera import InfraredCamera
from erweima.pose_estimator import PoseEstimator, TagDetection, RobotPose

class CircularPoseDemo:
    """圆形板位姿估计演示系统"""
    
    def __init__(self):
        self.estimator = PoseEstimator()
        self.camera = None
        self.device_id = 0
        
    def setup_camera(self, device_id: int = 0, width: int = 640, height: int = 480, fps: int = 10):
        """设置相机参数"""
        self.device_id = device_id
        self.camera = InfraredCamera(
            device_id=device_id,
            width=width,
            height=height,
            fps=fps
        )
        
    def detect_tags_manual(self, image: np.ndarray) -> List[TagDetection]:
        """手动实现二维码检测（兼容现有系统）"""
        try:
            import apriltag
        except ImportError:
            print("请先安装 apriltag 库: pip install apriltag")
            return []
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 创建检测器
        detector = apriltag.Detector()
        results = detector.detect(gray)
        
        detections = []
        for result in results:
            if result.tag_id in self.estimator.tag_coordinates:
                # 计算角度
                pt0, pt1 = result.corners[0], result.corners[1]
                dx, dy = pt1[0] - pt0[0], pt1[1] - pt0[1]
                angle = math.degrees(math.atan2(dy, dx))
                angle = (angle + 360) % 360
                
                detection = TagDetection(
                    tag_id=result.tag_id,
                    corners=result.corners,
                    center=np.mean(result.corners, axis=0),
                    angle=angle
                )
                detections.append(detection)
        
        return detections
    
    def run_realtime_demo(self):
        """实时位姿估计演示"""
        if not self.camera:
            self.setup_camera()
        
        if not self.camera.open():
            print(f"无法打开相机设备 {self.device_id}")
            return
        
        print("开始实时位姿估计演示...")
        print("按 'Q' 退出，按 'S' 保存当前帧")
        
        frame_count = 0
        
        try:
            while True:
                frame = self.camera.read_frame()
                if frame is None:
                    print("无法读取相机帧")
                    time.sleep(1)
                    continue
                
                # 转换为8位图像
                if frame.dtype == np.uint16:
                    img8 = (frame / 256).astype(np.uint8)
                elif frame.dtype == np.uint8:
                    img8 = frame
                else:
                    img8 = frame.astype(np.uint8)
                
                # 转换为3通道
                if len(img8.shape) == 2:
                    img_bgr = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
                else:
                    img_bgr = img8
                
                # 检测二维码
                detections = self.detect_tags_manual(img_bgr)
                
                # 估计位姿
                if detections:
                    pose = self.estimator.estimate_pose_multi_tags(detections)
                    vis_img = self.visualize_results(img_bgr, detections, pose)
                else:
                    vis_img = img_bgr.copy()
                    cv2.putText(vis_img, "No tags detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 显示结果
                cv2.imshow("Circular Board Pose Estimation", vis_img)
                
                # 按键处理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户退出")
                    break
                elif key == ord('s'):
                    filename = f"pose_capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, vis_img)
                    print(f"已保存: {filename}")
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"处理帧数: {frame_count}")
                    
        except KeyboardInterrupt:
            print("用户中断")
        finally:
            self.camera.close()
            cv2.destroyAllWindows()
    
    def visualize_results(self, image: np.ndarray, detections: List[TagDetection], 
                         pose: Optional[RobotPose]) -> np.ndarray:
        """可视化检测结果和位姿"""
        vis_img = image.copy()
        
        # 绘制检测到的二维码
        for detection in detections:
            if detection.tag_id in self.estimator.tag_coordinates:
                # 绘制边框
                corners = detection.corners.astype(np.int32)
                cv2.polylines(vis_img, [corners], True, (0, 255, 0), 2)
                
                # 显示ID和世界坐标
                world_x, world_y = self.estimator.tag_coordinates[detection.tag_id]
                text = f"ID:{detection.tag_id}({world_x:.2f},{world_y:.2f})"
                cv2.putText(vis_img, text,
                           (int(detection.corners[0][0]), int(detection.corners[0][1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 显示位姿信息
        if pose:
            pose_text = f"X:{pose.x:.2f}m Y:{pose.y:.2f}m θ:{pose.theta:.1f}°"
            conf_text = f"Conf:{pose.confidence:.2f}"
            cv2.putText(vis_img, pose_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_img, conf_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(vis_img, "No pose estimation", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 绘制圆板轮廓
        h, w = vis_img.shape[:2]
        center_x, center_y = w // 2, h // 2
        scale = min(w, h) / (2.5 * 100)  # 像素/米比例
        
        # 绘制圆板边界
        radius_pixels = int(1.25 * scale)
        cv2.circle(vis_img, (center_x, center_y), radius_pixels, (255, 255, 255), 1)
        
        # 绘制网格
        for r in [0.5, 1.0]:
            r_pixels = int(r * scale)
            cv2.circle(vis_img, (center_x, center_y), r_pixels, (100, 100, 100), 1)
        
        return vis_img
    
    def process_image_file(self, image_path: str):
        """处理单张图片"""
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            return
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return
        
        detections = self.detect_tags_manual(image)
        
        if detections:
            pose = self.estimator.estimate_pose_multi_tags(detections)
            result_img = self.visualize_results(image, detections, pose)
            
            # 保存结果
            output_path = f"pose_result_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, result_img)
            print(f"结果已保存: {output_path}")
            
            if pose:
                print(f"估计位姿: X={pose.x:.3f}m, Y={pose.y:.3f}m, θ={pose.theta:.1f}°")
        else:
            print("未检测到二维码")
    
    def generate_system_report(self):
        """生成系统配置报告"""
        report = {
            "system_config": {
                "camera_height": 0.3,
                "camera_fov": 50,
                "tag_size": 0.012,
                "circle_radius": 1.25,
                "total_tags": len(self.estimator.tag_coordinates)
            },
            "tag_distribution": self._analyze_tag_distribution(),
            "coverage_analysis": self._analyze_coverage()
        }
        
        with open("system_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("系统报告已生成: system_report.json")
    
    def _analyze_tag_distribution(self) -> Dict:
        """分析二维码分布"""
        coords = list(self.estimator.tag_coordinates.values())
        x_coords = [c[0] for c in coords]
        y_coords = [c[1] for c in coords]
        
        return {
            "min_x": min(x_coords),
            "max_x": max(x_coords),
            "min_y": min(y_coords),
            "max_y": max(y_coords),
            "count": len(coords),
            "density": len(coords) / (math.pi * 1.25**2)
        }
    
    def _analyze_coverage(self) -> Dict:
        """分析覆盖率"""
        # 简化的覆盖率分析
        test_points = 0
        covered_points = 0
        
        for x in np.arange(-1.2, 1.21, 0.1):
            for y in np.arange(-1.2, 1.21, 0.1):
                if x**2 + y**2 <= 1.25**2:
                    test_points += 1
                    min_dist = min(
                        math.sqrt((x - tx)**2 + (y - ty)**2)
                        for tx, ty in self.estimator.tag_coordinates.values()
                    )
                    if min_dist <= 0.14:  # 视野半径
                        covered_points += 1
        
        return {
            "total_points": test_points,
            "covered_points": covered_points,
            "coverage_rate": covered_points / test_points * 100
        }

def main():
    """主函数"""
    demo = CircularPoseDemo()
    
    print("=== 圆形板位姿估计系统 ===")
    print("1. 实时演示")
    print("2. 处理图片文件")
    print("3. 生成系统报告")
    print("4. 导出打印文件")
    
    choice = input("请选择操作 (1-4): ").strip()
    
    if choice == "1":
        demo.run_realtime_demo()
    elif choice == "2":
        image_path = input("请输入图片路径: ").strip()
        demo.process_image_file(image_path)
    elif choice == "3":
        demo.generate_system_report()
    elif choice == "4":
        output_dir = input("请输入输出目录 (默认 print_tags): ").strip()
        if not output_dir:
            output_dir = "print_tags"
        estimator = PoseEstimator()
        estimator.export_for_printing(output_dir)
    else:
        print("无效选择")

if __name__ == "__main__":
    main()