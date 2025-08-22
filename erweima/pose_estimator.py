#!/usr/bin/env python3
"""
基于二维码的机器人位姿估计系统
集成现有apriltag检测程序，实现完整位姿计算
"""

import json
import numpy as np
import cv2
import math
import os
import sys
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import logging
import yaml

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pose_estimator")

@dataclass
class TagDetection:
    """二维码检测结果"""
    tag_id: int
    corners: np.ndarray  # 4x2数组，像素坐标
    center: np.ndarray   # 中心点像素坐标
    angle: float         # 角度（度）

@dataclass
class RobotPose:
    """机器人位姿"""
    x: float      # 世界坐标X (m)
    y: float      # 世界坐标Y (m)
    theta: float  # 朝向角度（度，0-360）
    confidence: float  # 置信度

class PoseEstimator:
    """位姿估计器"""
    
    def __init__(self, layout_file: str = ""):
        """初始化位姿估计器"""
        self.tag_coordinates = {}  # ID到世界坐标的映射
        self.camera_height = 0.25   # 相机高度(m)，默认0.25m
        self.camera_fov = 50       # 相机FOV(度)
        
        # 相机内参（需要标定）
        self.camera_matrix = np.array([
            [640, 0, 320],
            [0, 480, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = np.zeros((4, 1))  # 无畸变
        
        # 加载相机参数
        config_path = os.path.join(os.path.dirname(__file__), "../config/infrared_config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                camera_cfg = config.get('camera', {})
                self.pixel_size_um = camera_cfg.get('pixel_size_um', 3.4)
                self.focal_length_mm = camera_cfg.get('focal_length_mm', 2.6)
                self.camera_width = camera_cfg.get('width', 640)
                self.camera_height_px = camera_cfg.get('height', 480)
        else:
            self.pixel_size_um = 3.4
            self.focal_length_mm = 2.6
            self.camera_width = 640
            self.camera_height_px = 480
        
        # 优先读取同目录下的circular_tag_layout.json
        if not layout_file:
            layout_file = os.path.join(os.path.dirname(__file__), "circular_tag_layout.json")
        if layout_file and os.path.exists(layout_file):
            self.load_layout(layout_file)
        else:
            self.generate_default_layout()
    
    def load_layout(self, layout_file: str):
        """从文件加载二维码布局"""
        try:
            with open(layout_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.tag_coordinates = {
                    tag["id"]: (tag["x"], tag["y"])
                    for tag in data["tags"]
                }
            logger.info(f"已加载 {len(self.tag_coordinates)} 个二维码坐标")
        except Exception as e:
            logger.error(f"加载布局失败: {e}")
            self.generate_default_layout()
    
    def generate_default_layout(self):
        """生成默认二维码布局"""
        # 手动生成布局，避免循环导入
        tags = []
        tag_id = 0
        
        # 中心点
        tags.append({
            "id": tag_id,
            "x": 0.0,
            "y": 0.0,
            "radius": 0.0,
            "angle": 0.0
        })
        tag_id += 1
        
        # 7圈布局
        ring_configs = [
            {"radius": 0.20, "count": 6},
            {"radius": 0.35, "count": 12},
            {"radius": 0.50, "count": 18},
            {"radius": 0.65, "count": 24},
            {"radius": 0.80, "count": 24},
            {"radius": 0.95, "count": 12},
            {"radius": 1.10, "count": 12}
        ]
        
        for config in ring_configs:
            radius = config["radius"]
            count = config["count"]
            
            for i in range(count):
                angle = 2 * math.pi * i / count
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                
                tags.append({
                    "id": tag_id,
                    "x": round(x, 3),
                    "y": round(y, 3)
                })
                tag_id += 1
        
        self.tag_coordinates = {tag["id"]: (tag["x"], tag["y"]) for tag in tags}
        logger.info("已生成默认布局")
    
    def detect_tags(self, image: np.ndarray) -> List[TagDetection]:
        """检测图像中的二维码"""
        try:
            import apriltag
        except ImportError:
            logger.error("请先安装 apriltag 库: pip install apriltag")
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
            if result.tag_id in self.tag_coordinates:
                detection = TagDetection(
                    tag_id=result.tag_id,
                    corners=result.corners,
                    center=np.mean(result.corners, axis=0),
                    angle=self._calculate_tag_angle(result.corners)
                )
                detections.append(detection)
        
        return detections
    
    def _calculate_tag_angle(self, corners: np.ndarray) -> float:
        """计算二维码朝向角度"""
        # 计算上边向量
        pt0, pt1 = corners[0], corners[1]
        dx, dy = pt1[0] - pt0[0], pt1[1] - pt0[1]
        
        # 计算角度（度）
        angle = math.degrees(math.atan2(dy, dx))
        angle = (angle + 360) % 360  # 归一化到0-360
        
        return angle
    
    def estimate_pose_single_tag(self, detection: TagDetection) -> Optional[RobotPose]:
        """基于单个二维码估计位姿"""
        if detection.tag_id not in self.tag_coordinates:
            return None
        
        tag_world = self.tag_coordinates[detection.tag_id]
        
        # 计算距离（基于相机参数）
        # 取二维码对角线像素距离
        pixel_distance = np.linalg.norm(detection.corners[0] - detection.corners[2])
        # 计算像素在成像空间的实际长度
        # pixel_size_scene = (pixel_size_sensor * object_distance) / focal_length
        # 这里object_distance用self.camera_height近似
        pixel_size_sensor_m = self.pixel_size_um * 1e-6
        focal_length_m = self.focal_length_mm * 1e-3
        object_distance = self.camera_height
        pixel_size = (pixel_size_sensor_m * object_distance) / focal_length_m
        actual_distance = pixel_distance * pixel_size
        
        # 计算相对位置
        tag_center = np.array(tag_world)
        robot_pos = tag_center + np.array([
            -actual_distance * math.cos(math.radians(detection.angle)),
            -actual_distance * math.sin(math.radians(detection.angle))
        ])
        
        # 机器人朝向 = 二维码朝向 + 180°
        robot_angle = (detection.angle + 180) % 360
        
        return RobotPose(
            x=robot_pos[0],
            y=robot_pos[1],
            theta=robot_angle,
            confidence=0.8  # 单二维码置信度
        )
    
    def estimate_pose_multi_tags(self, detections: List[TagDetection]) -> Optional[RobotPose]:
        """基于多个二维码优化位姿估计"""
        if not detections:
            return None
        
        if len(detections) == 1:
            return self.estimate_pose_single_tag(detections[0])
        
        # 收集观测数据
        positions = []
        angles = []
        weights = []
        
        for detection in detections:
            pose = self.estimate_pose_single_tag(detection)
            if pose:
                positions.append([pose.x, pose.y])
                angles.append(pose.theta)
                weights.append(1.0)  # 可以基于距离加权
        
        if not positions:
            return None
        
        # 使用最小二乘法优化
        positions = np.array(positions)
        
        # 优化位置（取中位数或加权平均）
        x_opt = np.median(positions[:, 0])
        y_opt = np.median(positions[:, 1])
        
        # 优化角度（处理角度周期性）
        angles = np.array(angles)
        angles_rad = np.radians(angles)
        
        # 计算平均角度
        avg_sin = np.mean(np.sin(angles_rad))
        avg_cos = np.mean(np.cos(angles_rad))
        theta_opt = math.degrees(math.atan2(avg_sin, avg_cos))
        theta_opt = (theta_opt + 360) % 360
        
        # 计算置信度
        confidence = min(1.0, 0.8 + 0.1 * (len(detections) - 1))
        
        return RobotPose(
            x=x_opt,
            y=y_opt,
            theta=theta_opt,
            confidence=confidence
        )
    
    def visualize_pose(self, image: np.ndarray, pose: RobotPose, 
                      detections: List[TagDetection]) -> np.ndarray:
        """可视化位姿估计结果"""
        vis_img = image.copy()
        
        # 绘制检测到的二维码
        for detection in detections:
            if detection.tag_id in self.tag_coordinates:
                # 绘制边框
                corners = detection.corners.astype(np.int32)
                cv2.polylines(vis_img, [corners], True, (0, 255, 0), 2)
                
                # 显示ID
                text = f"ID:{detection.tag_id}"
                cv2.putText(vis_img, text, 
                           (int(detection.corners[0][0]), int(detection.corners[0][1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示位姿信息
        pose_text = f"Pose: ({pose.x:.2f}, {pose.y:.2f}, {pose.theta:.1f}°)"
        cv2.putText(vis_img, pose_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_img
    
    def export_for_printing(self, output_dir: str = "print_tags"):
        """导出用于打印的二维码文件"""
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
            from erweima.generate_apriltag_images import generate_apriltag
            
            os.makedirs(output_dir, exist_ok=True)
            
            for tag_id, (x, y) in self.tag_coordinates.items():
                try:
                    # 生成二维码图像
                    img = generate_apriltag('tag36h11', tag_id, size=300)
                    
                    # 保存文件
                    filename = f"tag_{tag_id}_x{x:.2f}_y{y:.2f}.png"
                    filepath = os.path.join(output_dir, filename)
                    cv2.imwrite(filepath, img)
                    
                    print(f"已生成: {filename}")
                except Exception as e:
                    print(f"生成二维码 {tag_id} 失败: {e}")
        except ImportError:
            print("无法导入generate_apriltag_images，跳过导出")

    def set_camera_height(self, height: float):
        """设置相机离地高度（单位：米）"""
        self.camera_height = height
    
    def get_camera_height(self) -> float:
        """获取当前相机离地高度（单位：米）"""
        return self.camera_height

    def get_camera_orientation_on_disk(self, detection: TagDetection) -> float:
        """
        计算相机在圆盘（世界）坐标系下的朝向角度
        apriltag检测角度以x轴正方向为0°逆时针，
        圆盘定义以y轴正方向为0°顺时针。
        转换公式：theta_disk = (90 - detection.angle) % 360
        """
        tag_angle_img = detection.angle  # apriltag检测角度
        # 若布局有angle字段可加上，这里假设为0
        theta_disk = (90 - tag_angle_img) % 360
        return theta_disk

def test_pose_estimation():
    """测试位姿估计功能"""
    estimator = PoseEstimator()
    
    # 创建测试图像
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 模拟检测结果
    detections = [
        TagDetection(
            tag_id=0,
            corners=np.array([[100, 100], [150, 100], [150, 150], [100, 150]]),
            center=np.array([125, 125]),
            angle=45.0
        )
    ]
    
    # 估计位姿
    pose = estimator.estimate_pose_multi_tags(detections)
    
    if pose:
        print(f"估计位姿: X={pose.x:.3f}m, Y={pose.y:.3f}m, θ={pose.theta:.1f}°")
        print(f"置信度: {pose.confidence:.2f}")
    else:
        print("位姿估计失败")

if __name__ == "__main__":
    # 测试位姿估计器
    test_pose_estimation()
    
    # 导出打印文件
    estimator = PoseEstimator()
    estimator.export_for_printing()